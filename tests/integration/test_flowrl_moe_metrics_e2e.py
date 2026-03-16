#!/usr/bin/env python
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
End-to-end simulation test for FlowRLActor MoE metrics collection.

This script simulates the complete flow of MoE metrics collection during
FlowRL training without loading any actual models (CPU only).

Usage:
    python tests/integration/test_flowrl_moe_metrics_e2e.py
"""

import sys
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

import torch


# ============================================================================
# Mock Classes to Simulate FlowRLActor Behavior
# ============================================================================

@dataclass
class MockMoEModelConfig:
    """Mock config for MoE model (e.g., Qwen3-30B-A3B)."""
    model_type: str = "qwen3_moe"
    num_experts_per_tok: int = 8
    num_local_experts: int = 128
    hidden_size: int = 4096
    num_attention_heads: int = 32


@dataclass
class MockDenseModelConfig:
    """Mock config for dense model (e.g., Qwen3-8B)."""
    model_type: str = "qwen3"
    hidden_size: int = 4096
    num_attention_heads: int = 32


@dataclass 
class MockModelOutput:
    """Mock transformer model output with optional router_logits."""
    logits: torch.Tensor
    hidden_states: Optional[tuple] = None
    router_logits: Optional[tuple] = None


class MockFlowRLActor:
    """
    Simplified mock of FlowRLActor that demonstrates MoE metrics collection.
    
    This simulates the key logic from recipe/flowrl/flowrl_actor.py without
    requiring actual model loading.
    """
    
    def __init__(self, model_config, collect_moe_metrics=None, moe_top_k=8):
        """
        Initialize mock actor with MoE detection logic.
        
        Args:
            model_config: Mock model configuration
            collect_moe_metrics: None (auto), True (force enable), False (disable)
            moe_top_k: Number of experts per token
        """
        from verl.utils.moe_metrics import MoEMetricsCollector
        
        # Store config
        self.config = model_config
        
        # ====== MoE Detection Logic (from FlowRLActor.__init__) ======
        self.is_moe_model = False
        self.moe_top_k = moe_top_k
        
        # Check for MoE indicators in config
        if hasattr(model_config, "num_experts_per_tok") and model_config.num_experts_per_tok is not None:
            self.is_moe_model = True
            self.moe_top_k = model_config.num_experts_per_tok
        elif hasattr(model_config, "num_local_experts") and model_config.num_local_experts is not None:
            self.is_moe_model = True
        elif hasattr(model_config, "n_routed_experts") and model_config.n_routed_experts is not None:
            self.is_moe_model = True
        
        # Check model_type for known MoE architectures
        model_type = getattr(model_config, "model_type", "")
        if "moe" in model_type.lower() or "mixtral" in model_type.lower():
            self.is_moe_model = True
        
        # Determine collect_moe_metrics setting
        if collect_moe_metrics is not None:
            self.collect_moe_metrics = collect_moe_metrics and self.is_moe_model
        else:
            self.collect_moe_metrics = self.is_moe_model
        
        # Initialize collector
        if self.collect_moe_metrics:
            self.moe_metrics_collector = MoEMetricsCollector(top_k=self.moe_top_k)
            print(f"[MockFlowRLActor] MoE model detected. Metrics enabled with top_k={self.moe_top_k}")
        else:
            self.moe_metrics_collector = None
            if self.is_moe_model:
                print("[MockFlowRLActor] MoE model detected but metrics disabled by config")
            else:
                print("[MockFlowRLActor] Dense model detected. MoE metrics disabled")
    
    def _forward_micro_batch(self, model_output: MockModelOutput, collect_moe_metrics: bool = False):
        """
        Simulate forward pass and collect MoE metrics.
        
        Args:
            model_output: Mock model output
            collect_moe_metrics: Whether to collect metrics this pass
        """
        output_router_logits = collect_moe_metrics and self.collect_moe_metrics and self.is_moe_model
        
        if output_router_logits and hasattr(model_output, "router_logits") and model_output.router_logits is not None:
            self.moe_metrics_collector.add(model_output.router_logits, attention_mask=None)
    
    def update_policy(self, micro_batches: list[MockModelOutput]) -> dict:
        """
        Simulate policy update with MoE metrics collection.
        
        Args:
            micro_batches: List of mock model outputs
            
        Returns:
            Dictionary of metrics including MoE metrics
        """
        metrics = {}
        
        # Reset collector at start of update
        if self.collect_moe_metrics and self.moe_metrics_collector is not None:
            self.moe_metrics_collector.reset()
        
        # Process each micro-batch
        for model_output in micro_batches:
            self._forward_micro_batch(model_output, collect_moe_metrics=self.collect_moe_metrics)
            
            # Simulate some training metrics
            metrics.setdefault("actor/loss", []).append(0.5)
        
        # Compute MoE metrics after all micro-batches
        if self.collect_moe_metrics and self.moe_metrics_collector is not None:
            moe_metrics = self.moe_metrics_collector.compute_and_reset()
            if moe_metrics:
                metrics.update(moe_metrics)
        
        return metrics


# ============================================================================
# Test Functions
# ============================================================================

def generate_mock_moe_output(
    batch_size: int = 4,
    seq_len: int = 512,
    vocab_size: int = 151936,
    num_experts: int = 128,
    num_moe_layers: int = 28,
    routing_pattern: str = "balanced"
) -> MockModelOutput:
    """
    Generate mock model output with router_logits.
    
    Args:
        routing_pattern: "balanced", "collapsed", or "random"
    """
    num_tokens = batch_size * seq_len
    
    # Generate logits
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Generate router logits for each MoE layer
    router_logits_list = []
    for _ in range(num_moe_layers):
        if routing_pattern == "balanced":
            # Roughly uniform distribution
            router_logits = torch.randn(num_tokens, num_experts)
        elif routing_pattern == "collapsed":
            # Routing collapse: all tokens go to first 8 experts
            router_logits = torch.zeros(num_tokens, num_experts)
            router_logits[:, :8] = 10.0 + torch.randn(num_tokens, 8) * 0.1
        else:  # random
            router_logits = torch.randn(num_tokens, num_experts)
        
        router_logits_list.append(router_logits)
    
    return MockModelOutput(
        logits=logits,
        router_logits=tuple(router_logits_list)
    )


def generate_mock_dense_output(
    batch_size: int = 4,
    seq_len: int = 512,
    vocab_size: int = 151936,
) -> MockModelOutput:
    """Generate mock model output for dense model (no router_logits)."""
    logits = torch.randn(batch_size, seq_len, vocab_size)
    return MockModelOutput(logits=logits, router_logits=None)


def test_moe_model_metrics_collection():
    """Test MoE metrics collection for MoE model (Qwen3-30B-A3B)."""
    print("\n" + "=" * 70)
    print("Test: MoE Model (Qwen3-30B-A3B) Metrics Collection")
    print("=" * 70)
    
    # Create mock actor with MoE config
    config = MockMoEModelConfig()
    actor = MockFlowRLActor(model_config=config)
    
    assert actor.is_moe_model == True, "Should detect MoE model"
    assert actor.collect_moe_metrics == True, "Should enable metrics collection"
    assert actor.moe_top_k == 8, "Should detect top_k from config"
    
    # Generate mock micro-batches with balanced routing
    num_micro_batches = 4
    micro_batches = [
        generate_mock_moe_output(
            batch_size=4,
            seq_len=512,
            num_experts=config.num_local_experts,
            num_moe_layers=28,
            routing_pattern="balanced"
        )
        for _ in range(num_micro_batches)
    ]
    
    # Run update and collect metrics
    metrics = actor.update_policy(micro_batches)
    
    # Verify MoE metrics are present
    moe_metric_keys = [k for k in metrics.keys() if k.startswith("moe/")]
    assert len(moe_metric_keys) > 0, "Should have MoE metrics"
    
    print(f"\n📊 Collected {len(moe_metric_keys)} MoE metrics:")
    for key in sorted(moe_metric_keys):
        print(f"  {key}: {metrics[key]:.6f}")
    
    # Verify key metrics for balanced routing
    assert metrics["moe/expert_utilization_ratio"] > 0.9, "All experts should be used"
    assert metrics["moe/expert_load_entropy_normalized"] > 0.9, "Entropy should be high"
    
    print("\n✅ MoE model metrics collection test PASSED")
    return True


def test_dense_model_no_metrics():
    """Test that dense model doesn't collect MoE metrics."""
    print("\n" + "=" * 70)
    print("Test: Dense Model (Qwen3-8B) No MoE Metrics")
    print("=" * 70)
    
    # Create mock actor with dense config
    config = MockDenseModelConfig()
    actor = MockFlowRLActor(model_config=config)
    
    assert actor.is_moe_model == False, "Should detect dense model"
    assert actor.collect_moe_metrics == False, "Should disable metrics"
    assert actor.moe_metrics_collector is None, "Should not create collector"
    
    # Generate mock micro-batches (dense output)
    num_micro_batches = 4
    micro_batches = [
        generate_mock_dense_output(batch_size=4, seq_len=512)
        for _ in range(num_micro_batches)
    ]
    
    # Run update
    metrics = actor.update_policy(micro_batches)
    
    # Verify no MoE metrics
    moe_metric_keys = [k for k in metrics.keys() if k.startswith("moe/")]
    assert len(moe_metric_keys) == 0, "Should not have MoE metrics"
    
    print(f"\n📊 Collected metrics: {list(metrics.keys())}")
    print("   (No MoE metrics as expected for dense model)")
    
    print("\n✅ Dense model no-metrics test PASSED")
    return True


def test_routing_collapse_detection():
    """Test detection of routing collapse scenario."""
    print("\n" + "=" * 70)
    print("Test: Routing Collapse Detection")
    print("=" * 70)
    
    config = MockMoEModelConfig()
    actor = MockFlowRLActor(model_config=config)
    
    # Generate micro-batches with collapsed routing
    num_micro_batches = 4
    micro_batches = [
        generate_mock_moe_output(
            batch_size=4,
            seq_len=512,
            num_experts=config.num_local_experts,
            num_moe_layers=28,
            routing_pattern="collapsed"  # All tokens to few experts
        )
        for _ in range(num_micro_batches)
    ]
    
    metrics = actor.update_policy(micro_batches)
    
    print(f"\n📊 Routing Collapse Indicators:")
    print(f"  Expert utilization: {metrics['moe/expert_utilization_ratio']:.4f}")
    print(f"  Active experts: {metrics['moe/num_active_experts']:.0f} / {config.num_local_experts}")
    print(f"  Load imbalance ratio: {metrics['moe/load_imbalance_ratio']:.4f}")
    print(f"  Normalized entropy: {metrics['moe/expert_load_entropy_normalized']:.4f}")
    
    # In collapse scenario:
    # - Low utilization (only ~8 out of 128 experts used = 0.0625)
    # - Low entropy
    assert metrics["moe/expert_utilization_ratio"] < 0.2, "Utilization should be low"
    assert metrics["moe/num_active_experts"] <= 16, "Few experts should be active"
    
    print("\n⚠️  Routing collapse detected! These metrics indicate unhealthy training.")
    print("    Consider: adjusting load balance loss weight, lowering learning rate,")
    print("    or using auxiliary losses to encourage balanced routing.")
    
    print("\n✅ Routing collapse detection test PASSED")
    return True


def test_metrics_disabled_by_config():
    """Test that metrics can be explicitly disabled."""
    print("\n" + "=" * 70)
    print("Test: MoE Metrics Disabled by Config")
    print("=" * 70)
    
    # MoE model but metrics explicitly disabled
    config = MockMoEModelConfig()
    actor = MockFlowRLActor(model_config=config, collect_moe_metrics=False)
    
    assert actor.is_moe_model == True, "Should detect MoE model"
    assert actor.collect_moe_metrics == False, "Should be disabled by config"
    
    micro_batches = [
        generate_mock_moe_output(num_experts=128, num_moe_layers=28)
    ]
    
    metrics = actor.update_policy(micro_batches)
    
    moe_metric_keys = [k for k in metrics.keys() if k.startswith("moe/")]
    assert len(moe_metric_keys) == 0, "Should not collect metrics when disabled"
    
    print("   MoE model detected but metrics disabled by config")
    print("   No MoE metrics collected (as expected)")
    
    print("\n✅ Metrics disabled by config test PASSED")
    return True


def test_wandb_logging_simulation():
    """Simulate what would be logged to wandb."""
    print("\n" + "=" * 70)
    print("Test: WandB Logging Simulation")
    print("=" * 70)
    
    config = MockMoEModelConfig()
    actor = MockFlowRLActor(model_config=config)
    
    # Simulate multiple training steps
    num_steps = 3
    all_metrics = []
    
    for step in range(num_steps):
        micro_batches = [
            generate_mock_moe_output(
                batch_size=4,
                seq_len=512,
                num_experts=128,
                num_moe_layers=28,
                routing_pattern="balanced"
            )
            for _ in range(4)
        ]
        
        metrics = actor.update_policy(micro_batches)
        all_metrics.append(metrics)
        
        print(f"\n📝 Step {step + 1} - WandB log preview:")
        print("   {")
        for key in sorted(metrics.keys()):
            if key.startswith("moe/"):
                print(f'       "{key}": {metrics[key]:.6f},')
        print("   }")
    
    print("\n✅ WandB logging simulation test PASSED")
    return True


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all end-to-end tests."""
    print("=" * 70)
    print("FlowRLActor MoE Metrics End-to-End Simulation Test")
    print("(CPU Only - No Actual Model Loading)")
    print("=" * 70)
    
    tests = [
        ("MoE Model Metrics Collection", test_moe_model_metrics_collection),
        ("Dense Model No Metrics", test_dense_model_no_metrics),
        ("Routing Collapse Detection", test_routing_collapse_detection),
        ("Metrics Disabled by Config", test_metrics_disabled_by_config),
        ("WandB Logging Simulation", test_wandb_logging_simulation),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {name}")
        if not success:
            all_passed = False
    
    print("=" * 70)
    if all_passed:
        print("🎉 All end-to-end tests passed!")
        print("   MoE metrics collection is ready for FlowRL training.")
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
