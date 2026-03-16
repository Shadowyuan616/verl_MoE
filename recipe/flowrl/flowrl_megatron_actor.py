# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FlowRL Megatron Actor that extends MegatronPPOActor with FlowRL-specific components.
"""

import itertools
import logging
import os
from functools import partial
from typing import Iterable

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.distributed import finalize_model_grads
from megatron.core.optimizer import DistributedOptimizer
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from torch import nn

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    merge_router_topk_indices,
    pp_gather,
    reorder_and_merge_vpp_layers,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy, vocab_parallel_log_probs_from_logits
from verl.utils.megatron_utils import get_model_config, unwrap_model
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.profiler.profile import Profiler
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor.megatron_actor import MegatronPPOActor

__all__ = ["FlowRLMegatronActor", "ProjZModule"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ProjZModule(torch.nn.Module):
    """Projection network for estimating log partition function Z in FlowRL.

    This module is used to estimate the log partition function from the hidden states
    of the prompt. It uses a simple MLP with GELU activation and LayerNorm.
    """

    def __init__(self, hidden_size: int, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []

        for i in range(num_layers - 1):
            layers.extend(
                [
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.LayerNorm(hidden_size),
                    torch.nn.Dropout(dropout),
                ]
            )

        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FlowRLMegatronActor(MegatronPPOActor):
    """
    FlowRL version of MegatronPPOActor that implements the trajectory balance loss.

    This actor extends MegatronPPOActor with:
    - FlowRL trajectory balance loss computation
    - Support for log_z estimation via ProjZModule
    - Modified update_policy using FlowRL's loss function
    """

    def __init__(
        self,
        config,
        model_config,
        hf_config,
        tf_config,
        actor_module: nn.ModuleList,
        actor_optimizer: DistributedOptimizer,
        proj_z_module: nn.Module = None,
    ):
        """
        Initialize FlowRLMegatronActor.

        Args:
            config: Actor configuration
            model_config: Model configuration
            hf_config: HuggingFace config
            tf_config: Transformer config
            actor_module: Megatron actor module (ModuleList for VPP)
            actor_optimizer: Megatron distributed optimizer
            proj_z_module: ProjZModule for log Z estimation (on last PP stage only)
        """
        super().__init__(
            config=config,
            model_config=model_config,
            hf_config=hf_config,
            tf_config=tf_config,
            actor_module=actor_module,
            actor_optimizer=actor_optimizer,
        )
        # FlowRL hyperparameters (hardcoded as per paper)
        self.flowrl_beta_coef = 15.0  # β coefficient for reward scaling in flowrl loss

        # ProjZ module is only used on the last PP stage
        self.proj_z_module = proj_z_module

    def compute_flowrl_loss(
        self,
        log_prob,
        ref_log_prob,
        old_log_prob,
        log_z,
        reward,
        response_mask,
    ):
        """
        Compute FlowRL trajectory balance loss.

        Args:
            log_prob: Current policy log probabilities (bs, response_len)
            ref_log_prob: Reference policy log probabilities (bs, response_len)
            old_log_prob: Old policy log probabilities for importance weighting (bs, response_len)
            log_z: Log partition function estimate (bs, 1)
            reward: Rewards/advantages (bs, response_len)
            response_mask: Mask for valid response tokens (bs, response_len)

        Returns:
            loss: FlowRL trajectory balance loss (scalar)
            metrics: Dictionary of metrics
        """
        # Cast all inputs to float32 for numerically stable loss computation.
        # log_prob / ref_log_prob / old_log_prob may arrive as bf16 from Megatron;
        # log_z is already float32 (cast before calling this function).
        log_prob = log_prob.float()
        ref_log_prob = ref_log_prob.float()
        old_log_prob = old_log_prob.float()
        reward = reward.float()

        # squeeze log_z to (B,)
        log_z = log_z.squeeze(-1)

        # Average token log-probs & rewards over valid positions
        avg_log_prob = verl_F.masked_mean(log_prob, response_mask, axis=1)
        avg_ref_log_prob = verl_F.masked_mean(ref_log_prob, response_mask, axis=1)
        seq_log_reward = verl_F.masked_mean(reward, response_mask, axis=1)

        # FlowRL residual: logZ + logpf - β*R - logpref
        delta = log_z + avg_log_prob - self.flowrl_beta_coef * seq_log_reward - avg_ref_log_prob

        # Importance ratio from current vs old policy (product of token ratios)
        log_w = verl_F.masked_sum(log_prob - old_log_prob, response_mask, axis=1)
        imp_w_raw = torch.exp(log_w).detach()
        imp_w = torch.clamp(imp_w_raw, max=10)

        # Loss: weighted squared residual with importance weights
        weighted_losses = imp_w * (delta**2)
        avg_loss = torch.mean(weighted_losses)

        # PPO KL: negative_approx_kl = log_prob - old_log_prob
        negative_approx_kl = log_prob - old_log_prob
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

        # Reference KL: approx_kl_ref = log_prob - ref_log_prob
        approx_kl_ref = log_prob - ref_log_prob
        ref_kl = verl_F.masked_mean(-approx_kl_ref, response_mask)

        # Metrics
        metrics = {
            "actor/log_prob": verl_F.masked_mean(log_prob, response_mask).detach().item(),
            "actor/old_log_prob": verl_F.masked_mean(old_log_prob, response_mask).detach().item(),
            "actor/ref_log_prob": verl_F.masked_mean(ref_log_prob, response_mask).detach().item(),
            "actor/log_z": log_z.mean().detach().item(),
            "actor/log_reward": verl_F.masked_mean(reward, response_mask).detach().item(),
            "actor/final_loss": avg_loss.detach().item(),
            "actor/importance_weight": imp_w.mean().detach().item(),
            "actor/ppo_kl": ppo_kl.detach().item(),
            "actor/ref_kl": ref_kl.detach().item(),
        }

        return avg_loss, metrics

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for FlowRL update.

        FlowRL requires ref_log_prob in addition to the standard PPO keys.
        """
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "response_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "ref_log_prob",  # FlowRL requires reference log prob
        ]

        # Include pre-computed IS weights if present in batch
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # router replay
        if self.enable_routing_replay:
            select_keys.append("routed_experts")
        if self.has_multi_modal_inputs:
            data = data.select(select_keys, ["multi_modal_inputs"])
        else:
            data = data.select(batch_keys=select_keys)

        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch_flowrl(
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        """
        Forward and backward pass for FlowRL with hidden state extraction for log_z.

        This method extends the base forward_backward_batch to:
        1. Extract hidden states from the last layer on the last PP stage via forward hook
        2. Compute log_z using the ProjZ module
        3. Use FlowRL loss instead of standard PPO loss
        """
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

        from verl.models.mcore.util import preprocess_packed_seqs, postprocess_packed_seqs

        # broadcast from last pp rank to all other pp ranks
        data.to(get_device_id())
        data.batch = data.batch.contiguous()
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)
        self.has_multi_modal_inputs = "multi_modal_inputs" in mini_batch.non_tensor_batch.keys()
        if self.has_multi_modal_inputs:
            mini_batch.batch["multi_modal_inputs"] = mini_batch.non_tensor_batch["multi_modal_inputs"]
            mini_batch.batch["multi_modal_inputs_idx"] = torch.Tensor(
                list(range(len(mini_batch.non_tensor_batch["multi_modal_inputs"])))
            ).to(torch.int64)

        if mini_batch.batch["position_ids"].dim() == 3:  # qwen2vl mrope [bs, 3, seq_len]
            mini_batch.batch["position_ids"] = mini_batch.batch["position_ids"][:, 0]

        indices = None
        temperature = data.meta_info["temperature"]
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0
            else:
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            total_seqlen = max_token_len
        else:
            assert micro_batch_size is not None
            micro_batches = mini_batch.batch.split(micro_batch_size)
            seq_len = micro_batches[0]["input_ids"].shape[1]
            total_seqlen = micro_batch_size * seq_len
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        # Store references for use in nested functions
        proj_z_module = self.proj_z_module
        compute_flowrl_loss_fn = self.compute_flowrl_loss

        # ==== FlowRL: Container to capture decoder hidden states via forward hook ====
        # The Megatron forward path (both fused and non-fused) does NOT return hidden_states
        # in its output dict. We use a forward hook on the decoder (TransformerBlock) to
        # capture the last-layer hidden states before they go to the output layer.
        captured_hidden_states = {}  # micro-batch index -> hidden_states tensor

        def _decoder_hook(module, input, output):
            """Forward hook to capture decoder output hidden states.

            IMPORTANT: Do NOT detach() here! In FlowRL, the loss gradient must flow
            back through log_z → proj_z → hidden_states → decoder → actor parameters.
            This is "gradient path B" in FlowRL's trajectory balance objective.
            The FSDP version (flowrl_actor.py) also keeps hidden_states attached
            (output.hidden_states[-1] without detach).
            Detaching would collapse log_z to a near-constant value and break
            FlowRL's flow-balance regularization.
            """
            # output from TransformerBlock is hidden_states tensor
            # Shape: (total_nnz_packed, 1, hidden_size) in SBH format
            #    or: (seq_len, bs, hidden_size) in SBH format
            # If sequence_parallel is enabled, the sequence dim is sharded across TP ranks
            if isinstance(output, tuple):
                output = output[0]
            captured_hidden_states["latest"] = output  # keep grad for FlowRL path B

        # Check if sequence_parallel is enabled (need to gather before using hidden states)
        is_seq_parallel = False
        if hasattr(self.tf_config, "sequence_parallel"):
            is_seq_parallel = self.tf_config.sequence_parallel
        # ============================================================================

        def flowrl_loss_func(output, data, meta_info):
            """FlowRL-specific loss function that computes trajectory balance loss."""
            # Handle different output formats
            log_probs = None

            if hasattr(output, "log_probs"):
                log_probs = output.log_probs
            elif isinstance(output, dict):
                log_probs = output.get("log_probs", output)
            else:
                # Output is the hidden states from non-last PP stage
                log_probs = output

            device = log_probs.device if hasattr(log_probs, "device") else get_device_id()
            metrics = {}

            if forward_only:
                if post_process_fn is not None:
                    if hasattr(output, "log_probs"):
                        stats = post_process_fn({"log_probs": output.log_probs}, data)
                    else:
                        stats = post_process_fn(output, data)
                    metrics.update(stats)
                return torch.tensor(1.0, device=device), [metrics, None]

            responses = data["responses"]
            response_length = responses.size(1)
            response_mask = data["response_mask"].to(bool)
            attention_mask = data["attention_mask"].to(bool)
            batch_size_local = attention_mask.shape[0]
            seq_len_local = attention_mask.shape[1]

            # Extract log_prob for response tokens only
            log_prob = log_probs[:, -response_length - 1 : -1].contiguous()

            old_log_prob = data["old_log_probs"]
            advantages = data["advantages"]
            ref_log_prob = data["ref_log_prob"]

            # ==== FlowRL: Compute log_z from captured hidden states ====
            batch_size = log_prob.size(0)
            if "latest" in captured_hidden_states and proj_z_module is not None:
                hidden_states = captured_hidden_states.pop("latest")

                # If sequence_parallel, gather hidden states across TP group first
                # gather_from_sequence_parallel_region gathers along dim 0 (the seq dim in SBH)
                if is_seq_parallel:
                    hidden_states = gather_from_sequence_parallel_region(hidden_states)

                # Megatron decoder outputs in SBH format: (seq, batch, hidden)
                # Convert to BSH format: (batch, seq, hidden) for downstream processing
                hidden_states = hidden_states.transpose(0, 1).contiguous()

                # Unpack from THD packed format to (bs, seq_len, hidden_size)
                # using the same postprocess_packed_seqs utility
                if self.config.megatron.use_remove_padding:
                    # After transpose, hidden_states is (1, total_nnz_packed, hidden_size) in BSH
                    # Re-compute packed_seq_params from the current micro-batch's attention_mask
                    _, packed_seq_params_local = preprocess_packed_seqs(
                        data["input_ids"], attention_mask, pre_process=False
                    )
                    hidden_states_unpacked = postprocess_packed_seqs(
                        hidden_states,
                        packed_seq_params_local,
                        attention_mask,
                        batch_size_local,
                        seq_len_local,
                        post_process=True,
                    )
                else:
                    # BSHD format: after transpose, hidden_states is (bs, seq_len, hidden_size)
                    hidden_states_unpacked = hidden_states

                # Extract prompt hidden states (exclude response tokens)
                prompt_hidden = hidden_states_unpacked[:, : -response_length - 1, :]
                prompt_mask = attention_mask[:, : -response_length - 1]

                # Average pooling over prompt tokens
                avg_hidden = verl_F.masked_mean(
                    prompt_hidden, prompt_mask.unsqueeze(-1), axis=1
                )  # (bs, hidden_size)

                # Cast to ProjZ module's dtype (decoder hidden_states may be fp32
                # after SP gather / recompute, but ProjZ weights are bf16)
                proj_z_dtype = next(proj_z_module.parameters()).dtype
                avg_hidden = avg_hidden.to(proj_z_dtype)

                # Compute log_z (requires grad for ProjZ backward)
                log_z = proj_z_module(avg_hidden)

                # Cast log_z to float32 for numerically stable loss computation
                log_z = log_z.float()
            else:
                # For non-last PP stages or when proj_z is not available
                log_z = torch.zeros(batch_size, 1, device=device)
            # ===========================================================

            # Compute FlowRL loss
            policy_loss, flowrl_metrics = compute_flowrl_loss_fn(
                log_prob=log_prob,
                ref_log_prob=ref_log_prob,
                old_log_prob=old_log_prob,
                log_z=log_z,
                reward=advantages,
                response_mask=response_mask,
            )

            metrics.update(flowrl_metrics)
            return policy_loss, [metrics, None]

        def forward_step(batch_iter, model, return_schedule_plan: bool = False):
            """Forward step with hidden state extraction for FlowRL."""
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"].to(bool)
            position_ids = batch["position_ids"]

            unwrapped_model = unwrap_model(model)
            if hasattr(unwrapped_model, "vp_stage"):
                vp_rank = unwrapped_model.vp_stage
            else:
                vp_rank = 0

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)
            responses = batch["responses"]
            response_length = responses.size(1)
            label = position_ids.clone()
            label[:, -response_length - 1 : -1] = responses
            label_mask = attention_mask.clone()
            label_mask[:, : -response_length - 1] = False
            label_mask[:, -1] = False

            if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                layers_topk_idx = batch["routed_experts"]
                set_router_replay_data(layers_topk_idx, attention_mask, self.tf_config, vp_rank)

            from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_fused_fn

            # ==== FlowRL: Register forward hook on decoder to capture hidden states ====
            hook_handle = None
            if proj_z_module is not None and mpu.is_pipeline_last_stage(ignore_virtual=True):
                # Find the GPTModel's decoder (TransformerBlock) to hook
                gpt_model = unwrapped_model
                if hasattr(gpt_model, "language_model") and isinstance(gpt_model.language_model, GPTModel):
                    gpt_model = gpt_model.language_model
                if isinstance(gpt_model, GPTModel) and hasattr(gpt_model, "decoder"):
                    hook_handle = gpt_model.decoder.register_forward_hook(_decoder_hook)
            # ==========================================================================

            if self.use_fused_kernels:
                forward_fn = get_mcore_forward_fused_fn(self.hf_config)
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    labels=label,
                    labels_mask=label_mask,
                    temperature=temperature,
                    multi_modal_inputs=multi_modal_inputs,
                )
            else:
                forward_fn = get_mcore_forward_fn(self.hf_config)

                def logits_processor(logits, label, label_mask):
                    assert logits.shape[:2] == label.shape[:2]
                    assert label.shape == label_mask.shape
                    logits.div_(temperature)
                    ret = {}
                    log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                    log_probs = log_probs.masked_fill(~label_mask, 0.0)
                    ret["log_probs"] = log_probs
                    return ret

                logits_processor_args = {"label": label, "label_mask": label_mask}
                output = forward_fn(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    multi_modal_inputs=multi_modal_inputs,
                    logits_processor=logits_processor,
                    logits_processor_args=logits_processor_args,
                    data_format="thd" if self.config.megatron.use_remove_padding else "bshd",
                )

            # ==== FlowRL: Remove hook after forward pass ====
            if hook_handle is not None:
                hook_handle.remove()
            # ================================================

            if forward_only:
                meta_info = None
            else:
                meta_info = {
                    "clip_ratio": self.config.clip_ratio,
                    "entropy_coeff": self.config.entropy_coeff,
                }

            if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
                merge_router_topk_indices(
                    attention_mask, input_ids, self.mini_layer_topk_idx_list, self.tf_config, vp_rank
                )

            if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
                router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
                for router in router_instance_list:
                    router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

            return output, partial(flowrl_loss_func, data=batch, meta_info=meta_info)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        if mpu.get_pipeline_model_parallel_world_size() > 1:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,
                micro_batch_size=1,
                forward_only=forward_only,
            )
        else:
            losses_reduced = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=n_micro_batch,
                seq_length=total_seqlen,
                micro_batch_size=1,
                forward_only=forward_only,
            )

        if self.has_multi_modal_inputs:
            data.batch.pop("multi_modal_inputs")
            data.batch.pop("multi_modal_inputs_idx")
            data.non_tensor_batch.pop("multi_modal_inputs")

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                vp_size = len(self.actor_module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                losses_reduced["mini_layer_topk_idx_tensor"] = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
                )
            else:
                losses_reduced["mini_layer_topk_idx_tensor"] = torch.cat(self.mini_layer_topk_idx_list, dim=0)
            self.mini_layer_topk_idx_list = []

        return losses_reduced

    def update_policy(self, dataloader: Iterable[DataProto]) -> dict:
        """
        Update actor policy using FlowRL's trajectory balance loss.

        This overrides the parent's update_policy to use FlowRL-specific loss computation.

        Args:
            dataloader: Iterator over DataProto mini-batches (from make_minibatch_iterator)

        Returns:
            Dictionary of aggregated metrics
        """
        for module in self.actor_module:
            module.train()
        if self.proj_z_module is not None:
            self.proj_z_module.train()

        metrics = {}

        for data in dataloader:
            # Setup router replay if needed
            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

            self.actor_optimizer.zero_grad()
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            for chunk in self.actor_module:
                chunk.zero_grad_buffer()

            # Get batch config
            if data.meta_info.get("micro_batch_size", None) is not None:
                micro_batch_size = data.meta_info["micro_batch_size"]
            else:
                micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
            max_token_len = None
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.config.megatron.context_parallel_size

            # Forward-backward with FlowRL loss
            metric_micro_batch = self.forward_backward_batch_flowrl(
                data,
                forward_only=False,
                use_dynamic_bsz=self.config.use_dynamic_bsz,
                micro_batch_size=micro_batch_size,
                max_token_len=max_token_len,
                mini_batch_size=self.config.ppo_mini_batch_size,
            )
            metric_micro_batch = metric_micro_batch["output"]
            for metric in metric_micro_batch:
                # Note that metric[0] is metrics, metric[1] is None for FlowRL
                if metric is not None and metric[0] is not None:
                    append_to_dict(metrics, metric[0])

            # Optimizer step for main actor
            update_successful, grad_norm, num_zeros_in_grad = self.actor_optimizer.step()
            grad_data = {"actor/grad_norm": grad_norm}
            append_to_dict(metrics, grad_data)

            if not update_successful:
                raise NotImplementedError("Update failed")

            # Clear router replay state
            if self.config.router_replay.mode in ["R2", "R3"]:
                RouterReplay.clear_global_router_replay_action()
                RouterReplay.clear_global_indices()

        get_torch_device().empty_cache()
        return metrics
