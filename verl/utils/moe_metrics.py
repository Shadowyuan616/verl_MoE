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
MoE Expert Routing Metrics Utilities.

This module provides utilities for computing and logging MoE (Mixture of Experts)
expert routing distribution metrics. These metrics help monitor expert load balance
during training and can help detect routing collapse issues.

Key metrics computed:
- Expert load distribution (mean, std, max, min)
- Expert utilization ratio
- Expert load entropy
- Router confidence scores
- Load balance auxiliary loss

Usage:
    from verl.utils.moe_metrics import compute_moe_metrics_from_router_logits
    
    # During forward pass, collect router_logits from model output
    metrics = compute_moe_metrics_from_router_logits(
        router_logits_list,
        attention_mask=attention_mask,
        top_k=config.num_experts_per_tok
    )
"""

from typing import Optional

import torch


def compute_expert_load_from_logits(
    router_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    top_k: int = 8,
) -> torch.Tensor:
    """
    Compute the token load for each expert from router logits.
    
    Args:
        router_logits: Router logits tensor of shape (num_tokens, num_experts) 
                      or (batch, seq_len, num_experts)
        attention_mask: Optional attention mask of shape (batch, seq_len) to exclude padding
        top_k: Number of experts selected per token
        
    Returns:
        expert_load: Tensor of shape (num_experts,) containing the number of tokens routed to each expert
    """
    if router_logits.dim() == 3:
        batch_size, seq_len, num_experts = router_logits.shape
        
        # Apply attention mask to filter out padding tokens
        if attention_mask is not None:
            # Flatten while respecting mask
            valid_mask = attention_mask.bool().view(-1)  # (batch * seq_len,)
            router_logits_flat = router_logits.view(-1, num_experts)  # (batch * seq_len, num_experts)
            router_logits_flat = router_logits_flat[valid_mask]  # (num_valid_tokens, num_experts)
        else:
            router_logits_flat = router_logits.view(-1, num_experts)
    elif router_logits.dim() == 2:
        router_logits_flat = router_logits
        num_experts = router_logits_flat.shape[-1]
    else:
        raise ValueError(f"Expected router_logits to have 2 or 3 dimensions, got {router_logits.dim()}")
    
    # Get routing probabilities via softmax
    routing_probs = torch.softmax(router_logits_flat.float(), dim=-1)  # (num_tokens, num_experts)
    
    # Get top-k expert indices for each token
    _, topk_indices = torch.topk(routing_probs, k=top_k, dim=-1)  # (num_tokens, top_k)
    
    # Count how many tokens are routed to each expert
    # Create one-hot encoding and sum
    expert_load = torch.zeros(num_experts, device=router_logits.device, dtype=torch.float32)
    for k in range(top_k):
        expert_indices = topk_indices[:, k]
        expert_load.scatter_add_(0, expert_indices, torch.ones_like(expert_indices, dtype=torch.float32))
    
    return expert_load


def compute_load_balance_loss(
    router_logits: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    top_k: int = 8,
) -> torch.Tensor:
    """
    Compute the load balance auxiliary loss for MoE routing.
    
    This loss encourages balanced expert utilization by penalizing uneven load distribution.
    The loss is computed as: loss = num_experts * sum(fraction_tokens * fraction_router_probs)
    
    Args:
        router_logits: Router logits tensor of shape (num_tokens, num_experts) 
                      or (batch, seq_len, num_experts)
        attention_mask: Optional attention mask of shape (batch, seq_len)
        top_k: Number of experts selected per token
        
    Returns:
        load_balance_loss: Scalar tensor representing the load balance loss
    """
    if router_logits.dim() == 3:
        batch_size, seq_len, num_experts = router_logits.shape
        
        if attention_mask is not None:
            valid_mask = attention_mask.bool().view(-1)
            router_logits_flat = router_logits.view(-1, num_experts)[valid_mask]
        else:
            router_logits_flat = router_logits.view(-1, num_experts)
    else:
        router_logits_flat = router_logits
        num_experts = router_logits_flat.shape[-1]
    
    num_tokens = router_logits_flat.shape[0]
    if num_tokens == 0:
        return torch.tensor(0.0, device=router_logits.device)
    
    # Compute routing probabilities
    routing_probs = torch.softmax(router_logits_flat.float(), dim=-1)
    
    # Get top-k routing mask
    _, topk_indices = torch.topk(routing_probs, k=top_k, dim=-1)
    routing_mask = torch.zeros_like(routing_probs)
    routing_mask.scatter_(1, topk_indices, 1.0)
    
    # Fraction of tokens routed to each expert
    fraction_tokens = routing_mask.sum(dim=0) / (num_tokens * top_k)
    
    # Average routing probability to each expert
    fraction_router_probs = routing_probs.mean(dim=0)
    
    # Load balance loss
    load_balance_loss = num_experts * (fraction_tokens * fraction_router_probs).sum()
    
    return load_balance_loss


def compute_moe_metrics_from_router_logits(
    router_logits_list: list[torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    top_k: int = 8,
    layer_aggregation: str = "mean",
) -> dict[str, float]:
    """
    Compute comprehensive MoE routing metrics from a list of router logits.
    
    Args:
        router_logits_list: List of router logits from each MoE layer, 
                           each tensor has shape (batch*seq_len, num_experts) or (batch, seq_len, num_experts)
        attention_mask: Optional attention mask of shape (batch, seq_len)
        top_k: Number of experts selected per token
        layer_aggregation: How to aggregate metrics across layers ("mean", "sum", "last")
        
    Returns:
        Dictionary containing MoE routing metrics:
        - moe/expert_load_mean: Average number of tokens per expert
        - moe/expert_load_std: Standard deviation of expert loads
        - moe/expert_load_max: Maximum expert load
        - moe/expert_load_min: Minimum expert load
        - moe/expert_load_entropy: Entropy of expert load distribution (higher = more balanced)
        - moe/expert_utilization_ratio: Fraction of experts receiving any tokens
        - moe/router_confidence_mean: Average confidence of top-k routing decisions
        - moe/router_confidence_std: Std of routing confidence
        - moe/load_balance_loss: Load balancing auxiliary loss
        - moe/num_layers: Number of MoE layers processed
    """
    if not router_logits_list or len(router_logits_list) == 0:
        return {}
    
    metrics_per_layer = []
    total_load_balance_loss = 0.0
    
    for layer_idx, router_logits in enumerate(router_logits_list):
        if router_logits is None:
            continue
            
        # Detach and move to CPU for metrics computation
        router_logits = router_logits.detach()
        
        # Get dimensions
        if router_logits.dim() == 3:
            batch_size, seq_len, num_experts = router_logits.shape
            if attention_mask is not None:
                valid_mask = attention_mask.bool().view(-1)
                router_logits_flat = router_logits.view(-1, num_experts)[valid_mask]
            else:
                router_logits_flat = router_logits.view(-1, num_experts)
        elif router_logits.dim() == 2:
            router_logits_flat = router_logits
            num_experts = router_logits_flat.shape[-1]
        else:
            continue
        
        num_tokens = router_logits_flat.shape[0]
        if num_tokens == 0:
            continue
        
        # Compute routing probabilities
        routing_probs = torch.softmax(router_logits_flat.float(), dim=-1)
        
        # Get top-k routing decisions
        topk_probs, topk_indices = torch.topk(routing_probs, k=min(top_k, num_experts), dim=-1)
        
        # Compute expert load
        expert_load = torch.zeros(num_experts, device=router_logits.device, dtype=torch.float32)
        for k in range(min(top_k, num_experts)):
            expert_indices = topk_indices[:, k]
            expert_load.scatter_add_(0, expert_indices, torch.ones_like(expert_indices, dtype=torch.float32))
        
        # Normalize expert load to get distribution
        total_selections = expert_load.sum()
        if total_selections > 0:
            expert_load_normalized = expert_load / total_selections
        else:
            expert_load_normalized = expert_load
        
        # Compute metrics for this layer
        layer_metrics = {}
        
        # Expert load statistics
        layer_metrics["expert_load_mean"] = expert_load.mean().item()
        layer_metrics["expert_load_std"] = expert_load.std().item()
        layer_metrics["expert_load_max"] = expert_load.max().item()
        layer_metrics["expert_load_min"] = expert_load.min().item()
        
        # Expert load entropy (higher = more balanced)
        eps = 1e-10
        expert_load_entropy = -(expert_load_normalized * torch.log(expert_load_normalized + eps)).sum()
        layer_metrics["expert_load_entropy"] = expert_load_entropy.item()
        
        # Maximum possible entropy for reference
        max_entropy = torch.log(torch.tensor(float(num_experts)))
        layer_metrics["expert_load_entropy_normalized"] = (expert_load_entropy / max_entropy).item()
        
        # Expert utilization ratio (fraction of experts that received any tokens)
        experts_used = (expert_load > 0).sum().float()
        layer_metrics["expert_utilization_ratio"] = (experts_used / num_experts).item()
        
        # Router confidence (average probability of selected experts)
        layer_metrics["router_confidence_mean"] = topk_probs.mean().item()
        layer_metrics["router_confidence_std"] = topk_probs.std().item()
        layer_metrics["router_confidence_max"] = topk_probs.max().item()
        
        # Load balance loss for this layer
        lb_loss = compute_load_balance_loss(router_logits, attention_mask, top_k)
        layer_metrics["load_balance_loss"] = lb_loss.item()
        total_load_balance_loss += lb_loss.item()
        
        metrics_per_layer.append(layer_metrics)
    
    if not metrics_per_layer:
        return {}
    
    # Aggregate metrics across layers
    aggregated_metrics = {}
    num_layers = len(metrics_per_layer)
    
    for key in metrics_per_layer[0].keys():
        values = [m[key] for m in metrics_per_layer]
        if layer_aggregation == "mean":
            aggregated_metrics[f"moe/{key}"] = sum(values) / len(values)
        elif layer_aggregation == "sum":
            aggregated_metrics[f"moe/{key}"] = sum(values)
        elif layer_aggregation == "last":
            aggregated_metrics[f"moe/{key}"] = values[-1]
        else:
            aggregated_metrics[f"moe/{key}"] = sum(values) / len(values)
    
    aggregated_metrics["moe/num_layers"] = float(num_layers)
    aggregated_metrics["moe/load_balance_loss_total"] = total_load_balance_loss
    
    return aggregated_metrics


def compute_moe_metrics_from_model_output(
    model_output,
    attention_mask: Optional[torch.Tensor] = None,
    top_k: int = 8,
) -> dict[str, float]:
    """
    Compute MoE metrics from a model output that contains router_logits.
    
    This is a convenience function that extracts router_logits from model output
    and computes metrics.
    
    Args:
        model_output: Model output object that may contain router_logits attribute
        attention_mask: Optional attention mask
        top_k: Number of experts per token
        
    Returns:
        Dictionary of MoE metrics, empty if no router_logits found
    """
    # Check if model output has router_logits
    router_logits = getattr(model_output, "router_logits", None)
    
    if router_logits is None:
        return {}
    
    # router_logits is typically a tuple of tensors, one per MoE layer
    if isinstance(router_logits, (tuple, list)):
        router_logits_list = list(router_logits)
    else:
        router_logits_list = [router_logits]
    
    # Filter out None values
    router_logits_list = [r for r in router_logits_list if r is not None]
    
    if not router_logits_list:
        return {}
    
    return compute_moe_metrics_from_router_logits(
        router_logits_list=router_logits_list,
        attention_mask=attention_mask,
        top_k=top_k,
    )


class MoEMetricsCollector:
    """
    A collector class for accumulating MoE metrics across multiple forward passes.
    
    This is useful for computing statistics over multiple micro-batches before
    aggregating and logging to wandb.
    
    Usage:
        collector = MoEMetricsCollector(top_k=8)
        
        for micro_batch in mini_batch:
            output = model(micro_batch)
            collector.add(output.router_logits, attention_mask)
        
        metrics = collector.compute_and_reset()
    """
    
    def __init__(self, top_k: int = 8, num_experts: Optional[int] = None):
        """
        Initialize the MoE metrics collector.
        
        Args:
            top_k: Number of experts selected per token
            num_experts: Optional number of experts (will be inferred if not provided)
        """
        self.top_k = top_k
        self.num_experts = num_experts
        self.reset()
    
    def reset(self):
        """Reset all accumulated statistics."""
        self.total_expert_load = None
        self.total_tokens = 0
        self.total_load_balance_loss = 0.0
        self.confidence_sum = 0.0
        self.confidence_sq_sum = 0.0
        self.confidence_max = 0.0
        self.num_samples = 0
        self.num_layers = 0
    
    def add(
        self,
        router_logits: list[torch.Tensor] | tuple[torch.Tensor] | torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Add router logits from a forward pass to the collector.
        
        Args:
            router_logits: Router logits from model output (list/tuple of tensors or single tensor)
            attention_mask: Optional attention mask
        """
        if router_logits is None:
            return
        
        if isinstance(router_logits, (tuple, list)):
            router_logits_list = [r for r in router_logits if r is not None]
        else:
            router_logits_list = [router_logits]
        
        if not router_logits_list:
            return
        
        for router_logits_layer in router_logits_list:
            router_logits_layer = router_logits_layer.detach()
            
            # Get dimensions
            if router_logits_layer.dim() == 3:
                batch_size, seq_len, num_experts = router_logits_layer.shape
                if attention_mask is not None:
                    valid_mask = attention_mask.bool().view(-1)
                    router_logits_flat = router_logits_layer.view(-1, num_experts)[valid_mask]
                else:
                    router_logits_flat = router_logits_layer.view(-1, num_experts)
            elif router_logits_layer.dim() == 2:
                router_logits_flat = router_logits_layer
                num_experts = router_logits_flat.shape[-1]
            else:
                continue
            
            if self.num_experts is None:
                self.num_experts = num_experts
            
            num_tokens = router_logits_flat.shape[0]
            if num_tokens == 0:
                continue
            
            # Initialize total_expert_load if needed
            if self.total_expert_load is None:
                self.total_expert_load = torch.zeros(num_experts, device=router_logits_layer.device)
            
            # Compute routing probabilities and top-k
            routing_probs = torch.softmax(router_logits_flat.float(), dim=-1)
            topk_probs, topk_indices = torch.topk(routing_probs, k=min(self.top_k, num_experts), dim=-1)
            
            # Accumulate expert load
            for k in range(min(self.top_k, num_experts)):
                expert_indices = topk_indices[:, k]
                self.total_expert_load.scatter_add_(
                    0, expert_indices, torch.ones_like(expert_indices, dtype=torch.float32)
                )
            
            # Accumulate confidence statistics
            self.confidence_sum += topk_probs.sum().item()
            self.confidence_sq_sum += (topk_probs ** 2).sum().item()
            self.confidence_max = max(self.confidence_max, topk_probs.max().item())
            
            # Accumulate load balance loss
            lb_loss = compute_load_balance_loss(router_logits_layer, attention_mask, self.top_k)
            self.total_load_balance_loss += lb_loss.item()
            
            self.total_tokens += num_tokens
            self.num_layers += 1
        
        self.num_samples += 1
    
    def compute_and_reset(self) -> dict[str, float]:
        """
        Compute aggregated metrics and reset the collector.
        
        Returns:
            Dictionary of aggregated MoE metrics
        """
        if self.total_expert_load is None or self.total_tokens == 0:
            self.reset()
            return {}
        
        metrics = {}
        num_experts = self.num_experts
        
        # Expert load statistics
        expert_load = self.total_expert_load
        metrics["moe/expert_load_mean"] = expert_load.mean().item()
        metrics["moe/expert_load_std"] = expert_load.std().item()
        metrics["moe/expert_load_max"] = expert_load.max().item()
        metrics["moe/expert_load_min"] = expert_load.min().item()
        
        # Load imbalance ratio (max / mean)
        mean_load = expert_load.mean().item()
        if mean_load > 0:
            metrics["moe/load_imbalance_ratio"] = expert_load.max().item() / mean_load
        else:
            metrics["moe/load_imbalance_ratio"] = 0.0
        
        # Expert load entropy
        total_selections = expert_load.sum()
        if total_selections > 0:
            expert_load_normalized = expert_load / total_selections
            eps = 1e-10
            entropy = -(expert_load_normalized * torch.log(expert_load_normalized + eps)).sum()
            metrics["moe/expert_load_entropy"] = entropy.item()
            
            max_entropy = torch.log(torch.tensor(float(num_experts)))
            metrics["moe/expert_load_entropy_normalized"] = (entropy / max_entropy).item()
        
        # Expert utilization
        experts_used = (expert_load > 0).sum().float()
        metrics["moe/expert_utilization_ratio"] = (experts_used / num_experts).item()
        metrics["moe/num_active_experts"] = experts_used.item()
        
        # Router confidence statistics
        total_confidence_samples = self.total_tokens * self.top_k * (self.num_layers // self.num_samples if self.num_samples > 0 else 1)
        if total_confidence_samples > 0:
            metrics["moe/router_confidence_mean"] = self.confidence_sum / total_confidence_samples
            variance = (self.confidence_sq_sum / total_confidence_samples) - (metrics["moe/router_confidence_mean"] ** 2)
            metrics["moe/router_confidence_std"] = max(0, variance) ** 0.5
        metrics["moe/router_confidence_max"] = self.confidence_max
        
        # Load balance loss
        if self.num_layers > 0:
            metrics["moe/load_balance_loss"] = self.total_load_balance_loss / self.num_layers
        metrics["moe/load_balance_loss_total"] = self.total_load_balance_loss
        
        # Meta info
        metrics["moe/num_layers_processed"] = float(self.num_layers)
        metrics["moe/total_tokens_processed"] = float(self.total_tokens)
        
        self.reset()
        return metrics
