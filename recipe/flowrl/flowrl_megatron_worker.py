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

"""FlowRL Megatron Worker that uses FlowRLMegatronActor instead of standard MegatronPPOActor."""

import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

import psutil
import torch
import torch.distributed
from codetiming import Timer
from megatron.core import parallel_state as mpu, tensor_parallel
from omegaconf import DictConfig, OmegaConf

from recipe.flowrl.flowrl_megatron_actor import FlowRLMegatronActor, ProjZModule
from verl import DataProto
from verl.models.mcore import get_mcore_weight_converter
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_torch_device, set_expandable_segments
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction
from verl.utils.megatron_utils import (
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    per_tensor_generator,
)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.profiler import GPUMemoryLogger, log_gpu_memory_usage, simple_timer
from verl.utils.profiler.performance import reduce_timing, topk_reduce_ratio_min_max
from verl.utils.ray_utils import get_event_loop
from verl.utils.torch_functional import use_original_torch_compile
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

FLOWRL_PROJ_Z_CHECKPOINT_NAME = "flowrl_proj_z.pt"


def _ensure_model_list(model):
    return model if isinstance(model, list) else [model]


def _is_proj_z_param_name(name: str) -> bool:
    return name.startswith("proj_z.") or ".proj_z." in name


def _make_proj_z_injection_hook(hidden_size: int, proj_layers: int):
    def hook(model, *args, **kwargs):
        model_list = _ensure_model_list(model)
        if len(model_list) == 1 and "post_process" in kwargs:
            model_post_process = [bool(kwargs["post_process"])]
        elif (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and mpu.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            model_post_process = [
                mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
                for i in range(mpu.get_virtual_pipeline_model_parallel_world_size())
            ]
        else:
            model_post_process = [mpu.is_pipeline_last_stage()]

        assert len(model_post_process) == len(model_list), (
            f"Expected {len(model_post_process)} model chunks, got {len(model_list)}"
        )

        for index, model_chunk in enumerate(model_list):
            if not model_post_process[index] or hasattr(model_chunk, "proj_z"):
                continue

            proj_z_module = ProjZModule(hidden_size, num_layers=proj_layers)
            ref_param = next(model_chunk.parameters(), None)
            if ref_param is not None and ref_param.device.type != "meta":
                proj_z_module = proj_z_module.to(device=ref_param.device, dtype=ref_param.dtype)

            for param in proj_z_module.parameters():
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

            model_chunk.add_module("proj_z", proj_z_module)

        return model

    return hook


def _find_proj_z_module(actor_module) -> torch.nn.Module | None:
    from verl.utils.megatron_utils import unwrap_model

    modules = []
    for model_chunk in actor_module:
        unwrapped = unwrap_model(model_chunk)
        if hasattr(unwrapped, "proj_z"):
            modules.append(unwrapped.proj_z)

    if len(modules) > 1:
        raise RuntimeError(f"Expected a single proj_z module, found {len(modules)}")
    return modules[0] if modules else None


class FlowRLActorRolloutRefWorker(ActorRolloutRefWorker):
    """
    FlowRL version of Megatron ActorRolloutRefWorker.

    This worker adds FlowRL-specific modifications:
    - ProjZModule for log Z estimation (added on last PP stage)
    - FlowRLMegatronActor with trajectory balance loss (replaces standard MegatronPPOActor)
    - Proper handling of proj_z parameters during weight sync to rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        # FlowRL specific: track proj_z module
        self.proj_z_module = None

    def _filter_proj_z_rollout_params(self, params):
        items = params.items() if isinstance(params, dict) else params
        for name, value in items:
            if _is_proj_z_param_name(name):
                continue
            yield name, value

    def _patch_bridge_proj_z_filter(self):
        if self.bridge is None or getattr(self.bridge, "_flowrl_proj_z_filter_patched", False):
            return

        from verl.utils.megatron_utils import unwrap_model

        original_local_to_global = self.bridge._weight_name_mapping_mcore_local_to_global

        def filtered_local_to_global(model, consider_ep: bool = True):
            mapping = original_local_to_global(model, consider_ep=consider_ep)
            return {key: value for key, value in mapping.items() if not _is_proj_z_param_name(key)}

        @contextmanager
        def hide_proj_z_named_parameters(models):
            patched_models = [unwrap_model(model) for model in models]
            original_methods = []
            try:
                for model in patched_models:
                    original_named_parameters = model.named_parameters

                    def filtered_named_parameters(*args, _orig=original_named_parameters, **kwargs):
                        for name, param in _orig(*args, **kwargs):
                            if _is_proj_z_param_name(name):
                                continue
                            yield name, param

                    model.named_parameters = filtered_named_parameters
                    original_methods.append((model, original_named_parameters))
                yield
            finally:
                for model, original_named_parameters in original_methods:
                    model.named_parameters = original_named_parameters

        def patch_export_method(method_name: str):
            original_method = getattr(self.bridge, method_name, None)
            if original_method is None:
                return

            def filtered_export(models, *args, _orig=original_method, **kwargs):
                def filtered_generator():
                    with hide_proj_z_named_parameters(models):
                        yield from _orig(models, *args, **kwargs)

                return filtered_generator()

            setattr(self.bridge, method_name, filtered_export)

        self.bridge._weight_name_mapping_mcore_local_to_global = filtered_local_to_global
        patch_export_method("export_weights")
        patch_export_method("export_weights_without_gather")
        self.bridge._flowrl_proj_z_filter_patched = True

    def _is_proj_z_owner(self) -> bool:
        return self.proj_z_module is not None and mpu.is_pipeline_last_stage(ignore_virtual=True)

    def _is_proj_z_checkpoint_rank(self) -> bool:
        if not self._is_proj_z_owner():
            return False

        if mpu.get_data_parallel_rank() != 0 or mpu.get_tensor_model_parallel_rank() != 0:
            return False
        if hasattr(mpu, "get_context_parallel_rank") and mpu.get_context_parallel_rank() != 0:
            return False
        if hasattr(mpu, "get_expert_model_parallel_rank") and mpu.get_expert_model_parallel_rank() != 0:
            return False
        if hasattr(mpu, "get_expert_tensor_parallel_rank") and mpu.get_expert_tensor_parallel_rank() != 0:
            return False
        return True

    def _proj_z_checkpoint_path(self, checkpoint_path: str) -> str:
        return os.path.join(checkpoint_path, FLOWRL_PROJ_Z_CHECKPOINT_NAME)

    def _broadcast_proj_z_group(self, group, src_rank: int):
        if not self._is_proj_z_owner() or group is None:
            return
        if torch.distributed.get_world_size(group=group) <= 1:
            return
        for param in self.proj_z_module.parameters():
            torch.distributed.broadcast(param.data, src=src_rank, group=group)

    def _sync_proj_z_module_params(self):
        if not self._is_proj_z_owner():
            return

        self._broadcast_proj_z_group(
            group=mpu.get_data_parallel_group(),
            src_rank=mpu.get_data_parallel_src_rank(),
        )

        if hasattr(mpu, "get_context_parallel_group") and hasattr(mpu, "get_context_parallel_rank"):
            cp_group = mpu.get_context_parallel_group()
            cp_src_rank = torch.distributed.get_global_rank(group=cp_group, group_rank=0)
            self._broadcast_proj_z_group(group=cp_group, src_rank=cp_src_rank)

        if hasattr(mpu, "get_expert_model_parallel_group") and hasattr(mpu, "get_expert_model_parallel_world_size"):
            if mpu.get_expert_model_parallel_world_size() > 1:
                ep_group = mpu.get_expert_model_parallel_group()
                ep_src_rank = torch.distributed.get_global_rank(group=ep_group, group_rank=0)
                self._broadcast_proj_z_group(group=ep_group, src_rank=ep_src_rank)

        if hasattr(mpu, "get_expert_tensor_parallel_group") and hasattr(mpu, "get_expert_tensor_parallel_world_size"):
            if mpu.get_expert_tensor_parallel_world_size() > 1:
                etp_group = mpu.get_expert_tensor_parallel_group()
                etp_src_rank = torch.distributed.get_global_rank(group=etp_group, group_rank=0)
                self._broadcast_proj_z_group(group=etp_group, src_rank=etp_src_rank)

        self._broadcast_proj_z_group(
            group=mpu.get_tensor_model_parallel_group(),
            src_rank=mpu.get_tensor_model_parallel_src_rank(),
        )

    def _save_proj_z_sidecar(self, checkpoint_path: Optional[str], hdfs_path: Optional[str] = None):
        if checkpoint_path is None or not self._is_proj_z_owner():
            return

        local_proj_z_path = self._proj_z_checkpoint_path(checkpoint_path)
        if self._is_proj_z_checkpoint_rank():
            torch.save({"model": self.proj_z_module.state_dict()}, local_proj_z_path)
            if hdfs_path is not None:
                from verl.utils import hdfs_io

                hdfs_io.makedirs(hdfs_path, exist_ok=True)
                hdfs_io.copy(src=local_proj_z_path, dst=hdfs_path, dirs_exist_ok=True)

        torch.distributed.barrier()

    def _load_proj_z_sidecar(self, checkpoint_path: Optional[str], hdfs_path: Optional[str] = None):
        if checkpoint_path is None or not self._is_proj_z_owner():
            return

        proj_z_path = self._proj_z_checkpoint_path(checkpoint_path)
        if not os.path.exists(proj_z_path) and hdfs_path is not None:
            from verl.utils.fs import is_non_local

            if is_non_local(hdfs_path):
                from verl.utils import hdfs_io

                remote_proj_z_path = self._proj_z_checkpoint_path(hdfs_path)
                if hdfs_io.exists(remote_proj_z_path):
                    os.makedirs(checkpoint_path, exist_ok=True)
                    hdfs_io.copy(src=remote_proj_z_path, dst=checkpoint_path, dirs_exist_ok=True)

        if os.path.exists(proj_z_path):
            state_dict = torch.load(proj_z_path, map_location="cpu", weights_only=False)
            self.proj_z_module.load_state_dict(state_dict["model"])
        self._sync_proj_z_module_params()

    def _build_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config=None
    ):
        """Override to add ProjZModule for FlowRL."""
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer,
            get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config,
        )
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.actor.megatron if not self._is_ref else self.config.ref.megatron,
        )
        self.generation_config = get_generation_config(
            self.local_path,
            self.config.model.get("trust_remote_code", False),
        )
        self._patch_bridge_proj_z_filter()

        if self._is_actor or self._is_rollout:
            proj_z_hook = None
            if self._is_actor:
                proj_layers = getattr(self.config.actor, "proj_layer", 3)
                hidden_size = self.hf_config.hidden_size
                proj_z_hook = _make_proj_z_injection_hook(hidden_size=hidden_size, proj_layers=proj_layers)

            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # actor is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            print(f"[FlowRL DEBUG] rank={self.rank} Before make_megatron_module (actor branch)")
            actor_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
                override_ddp_config=override_ddp_config,
                peft_cls=self.peft_cls,
                peft_config=self.config.model.get("lora", None),
                post_model_creation_callbacks=[proj_z_hook] if proj_z_hook is not None else None,
            )
            self.tf_config = updated_tf_config
            print(f"actor_module: {len(actor_module)}")
            print(f"[FlowRL DEBUG] rank={self.rank} After make_megatron_module, before load_weight")
            if self.config.actor.load_weight:
                from verl.utils.model import get_hf_model_path, load_mcore_dist_weights, load_megatron_gptmodel_weights

                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        actor_module,
                        self.config.actor.megatron.dist_checkpointing_path,
                        is_value_model=False,
                        prefix=self.config.actor.megatron.dist_checkpointing_prefix,
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        print(f"[FlowRL DEBUG] rank={self.rank} Before bridge.load_weights/load_hf_weights (actor)")
                        if self.vanilla_bridge:
                            self.bridge.load_weights(actor_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(actor_module, local_model_path)
                        print(f"[FlowRL DEBUG] rank={self.rank} After bridge load (actor)")
                    else:
                        print(f"[FlowRL DEBUG] rank={self.rank} Before load_megatron_gptmodel_weights (actor)")
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False
                        )
                        print(f"[FlowRL DEBUG] rank={self.rank} After load_megatron_gptmodel_weights (actor)")

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After actor module init", logger=logger)

            print(f"[FlowRL DEBUG] rank={self.rank} Checking proj_z attachment")
            self.proj_z_module = _find_proj_z_module(actor_module) if self._is_actor else None
            if self.proj_z_module is not None:
                self._sync_proj_z_module_params()
                if self.rank == 0:
                    proj_z_params = sum(p.numel() for p in self.proj_z_module.parameters())
                    print(f"[FlowRL] Attached proj_z inside actor module with {proj_z_params:,} parameters")
            print(f"[FlowRL DEBUG] rank={self.rank} After ProjZ check")
            # ====================================================

            # === FlowRL FIX-1: Barrier before get_megatron_optimizer ===
            # PyTorch 2.8 uses lazy NCCL communicator initialization:
            # - bridge.load_hf_weights uses per-EP-group NCCL (EP groups independent)
            # - EP groups (g0016=EP0, g0025=EP1) may finish loading at different times
            # - get_megatron_optimizer requires DP-group NCCL (cross-EP-group pairs)
            # - With lazy init, if EP0 (g0016) is slower, EP1 (g0025) starts DP NCCL
            #   ops before EP0 calls the matching op → lazy init fails silently
            # - NCCL_ASYNC_ERROR_HANDLING=1 stores the error, surfaced at next NCCL op
            # Barrier ensures ALL 16 ranks (both EP groups) finish weight loading
            # BEFORE any rank enters get_megatron_optimizer's DP NCCL ops.
            print(f"[FlowRL DEBUG] rank={self.rank} Before barrier (pre-get_megatron_optimizer)")
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            print(f"[FlowRL DEBUG] rank={self.rank} After barrier (pre-get_megatron_optimizer)")
            # ===========================================================

        elif self._is_ref:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # ref is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )
            print(f"[FlowRL DEBUG] rank={self.rank} Before make_megatron_module (ref branch)")
            ref_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
            )
            self.tf_config = updated_tf_config
            print(f"[FlowRL DEBUG] rank={self.rank} After make_megatron_module (ref branch)")
            if self.config.ref.load_weight:
                from verl.utils.model import get_hf_model_path, load_mcore_dist_weights, load_megatron_gptmodel_weights

                assert self.config.actor.load_weight == self.config.ref.load_weight
                print(f"[FlowRL DEBUG] rank={self.rank} load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        ref_module,
                        self.config.ref.megatron.dist_checkpointing_path,
                        is_value_model=False,
                        prefix=self.config.ref.megatron.dist_checkpointing_prefix,
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        print(f"[FlowRL DEBUG] rank={self.rank} Before bridge.load_hf_weights (ref)")
                        if self.vanilla_bridge:
                            self.bridge.load_weights(ref_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(ref_module, local_model_path)
                        print(f"[FlowRL DEBUG] rank={self.rank} After bridge load (ref)")
                    else:
                        print(f"[FlowRL DEBUG] rank={self.rank} Before load_megatron_gptmodel_weights (ref)")
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False
                        )
                        print(f"[FlowRL DEBUG] rank={self.rank} After load_megatron_gptmodel_weights (ref)")
            log_gpu_memory_usage("After ref module init", logger=logger)
            print(f"[FlowRL DEBUG] rank={self.rank} _build_model_optimizer ref branch returning")
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config_megatron = init_megatron_optim_config(
                optim_config,
                use_distributed_optimizer=wrap_config.use_distributed_optimizer,
                fp16=self.dtype == torch.float16,
            )

            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config_megatron)
            actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=actor_optimizer, config=optim_config
            )
        else:
            optim_config = None
            actor_optimizer = None
            actor_optimizer_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        from verl.utils.megatron_utils import register_megatron_training_hooks

        print(f"[FlowRL DEBUG] rank={self.rank} Before register_megatron_training_hooks")
        register_megatron_training_hooks(actor_module, actor_optimizer)
        print(f"[FlowRL DEBUG] rank={self.rank} After register_megatron_training_hooks")

        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """Override init_model to use FlowRLMegatronActor instead of MegatronPPOActor."""
        if self.config.model.get("external_lib", None) is not None:
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        if self._is_actor:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_transformer_config", {}))
            )
            if self.enable_routing_replay:
                override_transformer_config["enable_routing_replay"] = True
            override_ddp_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_ddp_config", {}))
            )
        elif self._is_ref:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.ref.megatron.get("override_transformer_config", {}))
            )
        else:
            override_transformer_config = {}
        self.param_dtype = PrecisionType.to_dtype(self.config.actor.megatron.dtype)
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        if self._is_actor:
            optim_config = self.config.actor.optim if self._is_actor else None
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_optimizer_scheduler,
                self.actor_model_config,
                self.actor_optim_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
                override_ddp_config=override_ddp_config,
            )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            # ==== FlowRL: Remove proj_layer from config before dataclass conversion ====
            # proj_layer is FlowRL-specific and not part of McoreActorConfig
            actor_config = OmegaConf.to_container(self.config.actor, resolve=True)
            if 'proj_layer' in actor_config:
                actor_config.pop('proj_layer')
            actor_cfg = omega_conf_to_dataclass(OmegaConf.create(actor_config))
            # ===========================================================================

            # ==== FlowRL: Use FlowRLMegatronActor instead of MegatronPPOActor ====
            if self.rank == 0:
                print("[FlowRL] Using FlowRLMegatronActor instead of MegatronPPOActor")

            self.actor = FlowRLMegatronActor(
                config=actor_cfg,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
                proj_z_module=self.proj_z_module,  # FlowRL: pass ProjZ module
            )
            # ====================================================================

            print(f"routing replay layers: {len(RouterReplay.router_instances)}")
            log_gpu_memory_usage("After FlowRLMegatronActor init", logger=logger)

        if self._is_rollout:
            # === FlowRL FIX-2: Barrier before rollout init ===
            # Defense-in-depth: ensures ALL 16 ranks have completed the entire
            # actor build (including param/optimizer offload) before ANY rank
            # enters _build_rollout's vLLM init_process_group (global collective).
            print(f"[FlowRL DEBUG] rank={self.rank} Before barrier (pre-_build_rollout)")
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
            print(f"[FlowRL DEBUG] rank={self.rank} After barrier, Before _build_rollout")
            with use_original_torch_compile():
                self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            log_gpu_memory_usage("After rollout init", logger=logger)
            print(f"[FlowRL DEBUG] rank={self.rank} After _build_rollout")

        if self._is_ref:
            print(f"[FlowRL DEBUG] rank={self.rank} Before _build_model_optimizer (ref in init_model)")
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            print(f"[FlowRL DEBUG] rank={self.rank} Before MegatronPPOActor (ref)")
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            print(f"[FlowRL DEBUG] rank={self.rank} After MegatronPPOActor (ref)")
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

        if self._is_actor:
            from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
            from verl.utils.flops_counter import FlopsCounter

            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                checkpoint_config=self.config.actor.checkpoint,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                optimizer=self.actor_optimizer,
                optimizer_scheduler=self.actor_optimizer_scheduler,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
                bridge=self.bridge,
                provider=self.provider,
                use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
                peft_cls=self.peft_cls,
            )

            self.layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            self.weight_converter = None
            if not self.config.actor.megatron.use_mbridge:
                self.weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)

        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode.

        FlowRL: Filter out proj_z parameters before syncing to vLLM/SGLang.
        """
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params during rollout_mode", logger=logger)

        if self.bridge is not None:
            if self.vanilla_bridge:
                per_tensor_param = self.bridge.export_weights(self.actor.actor_module)
            else:
                per_tensor_param = self.bridge.export_hf_weights(self.actor.actor_module)
        else:
            per_tensor_param = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                self.weight_converter,
                self.tf_config,
                self.layer_name_mapping,
            )

        per_tensor_param = self._filter_proj_z_rollout_params(per_tensor_param)

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        await self.rollout.update_weights(per_tensor_param)

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor.actor_module)
        aggressive_empty_cache(force_sync=True)

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])

        # Important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    def update_actor(self, data: DataProto):
        """Update actor using FlowRL trajectory balance loss."""
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)

        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
            # Note: proj_z optimizer step is now handled inside update_policy per mini-batch

        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)

        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        super().load_checkpoint(
            checkpoint_path=checkpoint_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        if not self.config.actor.megatron.use_dist_checkpointing:
            self._load_proj_z_sidecar(checkpoint_path, hdfs_path=hdfs_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        super().save_checkpoint(
            checkpoint_path=checkpoint_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        if not self.config.actor.megatron.use_dist_checkpointing:
            self._save_proj_z_sidecar(checkpoint_path, hdfs_path=hdfs_path)


class FlowRLAsyncActorRolloutRefWorker(FlowRLActorRolloutRefWorker):
    """
    Async version of FlowRLActorRolloutRefWorker for async rollout mode.

    This worker adds async-specific methods required by vLLM/SGLang async server:
    - get_zeromq_address: Returns ZeroMQ address for vLLM distributed executor
    - wake_up/sleep: Context switching between rollout and training modes
    - chat_completion/generate: SGLang-specific async generation methods
    """

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret
