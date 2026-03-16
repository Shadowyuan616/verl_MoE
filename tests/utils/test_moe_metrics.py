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
Unit tests for MoE (Mixture of Experts) metrics utilities.

These tests verify that the MoE metrics collection and computation work correctly
for both MoE models (with router_logits) and dense models (without router_logits).

Run with: pytest tests/utils/test_moe_metrics.py -v
"""

import math
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch

from verl.utils.moe_metrics import (
    MoEMetricsCollector,
    compute_expert_load_from_logits,
    compute_load_balance_loss,
    compute_moe_metrics_from_model_output,
    compute_moe_metrics_from_router_logits,
)


class TestComputeExpertLoad:
    """Tests for compute_expert_load_from_logits function."""
    
    def test_basic_expert_load(self):
        """Test basic expert load computation."""
        # Create router logits where expert 0 always has highest logits
        num_tokens = 10
        num_experts = 4
        top_k = 2
        
        router_logits = torch.zeros(num_tokens, num_experts)
        router_logits[:, 0] = 10.0  # Expert 0 has highest logits
        router_logits[:, 1] = 5.0   # Expert 1 has second highest
        
        expert_load = compute_expert_load_from_logits(router_logits, top_k=top_k)
        
        # All tokens should select expert 0 and 1
        assert expert_load[0].item() == num_tokens  # Expert 0: 10 tokens
        assert expert_load[1].item() == num_tokens  # Expert 1: 10 tokens
        assert expert_load[2].item() == 0           # Expert 2: 0 tokens
        assert expert_load[3].item() == 0           # Expert 3: 0 tokens
    
    def test_expert_load_with_batch_dimension(self):
        """Test expert load computation with 3D input (batch, seq_len, num_experts)."""
        batch_size = 2
        seq_len = 5
        num_experts = 4
        top_k = 2
        
        router_logits = torch.randn(batch_size, seq_len, num_experts)
        expert_load = compute_expert_load_from_logits(router_logits, top_k=top_k)
        
        # Total selections should equal batch_size * seq_len * top_k
        total_selections = expert_load.sum().item()
        expected_selections = batch_size * seq_len * top_k
        assert total_selections == expected_selections
    
    def test_expert_load_with_attention_mask(self):
        """Test that attention mask correctly filters padding tokens."""
        batch_size = 2
        seq_len = 5
        num_experts = 4
        top_k = 2
        
        router_logits = torch.randn(batch_size, seq_len, num_experts)
        
        # Mask: first batch has 3 valid tokens, second has all 5
        attention_mask = torch.zeros(batch_size, seq_len)
        attention_mask[0, :3] = 1
        attention_mask[1, :] = 1
        
        expert_load = compute_expert_load_from_logits(
            router_logits, attention_mask=attention_mask, top_k=top_k
        )
        
        # Total selections: (3 + 5) valid tokens * top_k = 8 * 2 = 16
        total_selections = expert_load.sum().item()
        expected_selections = 8 * top_k
        assert total_selections == expected_selections


class TestLoadBalanceLoss:
    """Tests for compute_load_balance_loss function."""
    
    def test_perfectly_balanced_load(self):
        """Test that perfectly balanced routing results in minimum loss.
        
        The load balance loss formula is: num_experts * sum(fraction_tokens * fraction_router_probs)
        For perfectly balanced routing, this equals ~1.0 (the theoretical minimum).
        """
        num_tokens = 100
        num_experts = 4
        top_k = 1
        
        # Create routing where each expert gets equal share
        router_logits = torch.zeros(num_tokens, num_experts)
        for i in range(num_tokens):
            router_logits[i, i % num_experts] = 10.0
        
        lb_loss = compute_load_balance_loss(router_logits, top_k=top_k)
        
        # For perfectly balanced routing, loss should be close to 1.0
        # (the minimum value for this loss function)
        assert lb_loss.item() <= 1.1, f"Expected loss ~1.0 for balanced, got {lb_loss.item()}"
    
    def test_completely_imbalanced_load(self):
        """Test that completely imbalanced routing (single expert) results in high loss.
        
        When all tokens go to a single expert, the loss approaches num_experts (maximum).
        """
        num_tokens = 100
        num_experts = 4
        top_k = 1
        
        # All tokens route to expert 0
        router_logits = torch.zeros(num_tokens, num_experts)
        router_logits[:, 0] = 10.0
        
        lb_loss = compute_load_balance_loss(router_logits, top_k=top_k)
        
        # For completely imbalanced routing, loss should approach num_experts
        assert lb_loss.item() > 3.0, f"Expected loss close to {num_experts}, got {lb_loss.item()}"
        assert lb_loss.item() <= num_experts, f"Loss should not exceed {num_experts}"
    
    def test_loss_comparison(self):
        """Test that balanced routing has lower loss than imbalanced."""
        num_tokens = 100
        num_experts = 8
        top_k = 2
        
        # Balanced routing
        router_logits_balanced = torch.zeros(num_tokens, num_experts)
        for i in range(num_tokens):
            router_logits_balanced[i, i % num_experts] = 10.0
        
        # Imbalanced routing - first 2 experts get all tokens
        router_logits_imbalanced = torch.zeros(num_tokens, num_experts)
        router_logits_imbalanced[:, 0] = 10.0
        router_logits_imbalanced[:, 1] = 5.0
        
        lb_loss_balanced = compute_load_balance_loss(router_logits_balanced, top_k=top_k)
        lb_loss_imbalanced = compute_load_balance_loss(router_logits_imbalanced, top_k=top_k)
        
        assert lb_loss_balanced < lb_loss_imbalanced, \
            f"Balanced loss ({lb_loss_balanced.item()}) should be < imbalanced ({lb_loss_imbalanced.item()})"


class TestComputeMoEMetrics:
    """Tests for compute_moe_metrics_from_router_logits function."""
    
    def test_metrics_computation(self):
        """Test that all expected metrics are computed."""
        num_tokens = 50
        num_experts = 8
        num_layers = 4
        top_k = 2
        
        router_logits_list = [
            torch.randn(num_tokens, num_experts) for _ in range(num_layers)
        ]
        
        metrics = compute_moe_metrics_from_router_logits(
            router_logits_list, top_k=top_k
        )
        
        # Check that all expected metrics are present
        expected_metrics = [
            "moe/expert_load_entropy",
            "moe/expert_utilization_ratio",
            "moe/load_balance_loss",
            "moe/router_confidence_mean",
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
    
    def test_entropy_range(self):
        """Test that entropy is within expected range."""
        num_tokens = 100
        num_experts = 8
        top_k = 2
        
        router_logits = [torch.randn(num_tokens, num_experts)]
        metrics = compute_moe_metrics_from_router_logits(router_logits, top_k=top_k)
        
        # Entropy should be positive
        assert metrics["moe/expert_load_entropy"] > 0
        
        # Normalized entropy should be between 0 and 1
        assert 0 <= metrics["moe/expert_load_entropy_normalized"] <= 1
    
    def test_utilization_ratio_range(self):
        """Test that utilization ratio is within [0, 1]."""
        num_tokens = 50
        num_experts = 8
        top_k = 2
        
        router_logits = [torch.randn(num_tokens, num_experts)]
        metrics = compute_moe_metrics_from_router_logits(router_logits, top_k=top_k)
        
        assert 0 <= metrics["moe/expert_utilization_ratio"] <= 1


class TestMoEMetricsCollector:
    """Tests for MoEMetricsCollector class."""
    
    def test_single_add(self):
        """Test adding a single batch of router logits."""
        num_tokens = 50
        num_experts = 8
        top_k = 2
        
        collector = MoEMetricsCollector(top_k=top_k)
        
        router_logits = torch.randn(num_tokens, num_experts)
        collector.add(router_logits)
        
        metrics = collector.compute_and_reset()
        
        assert "moe/expert_load_mean" in metrics
        assert "moe/load_balance_loss" in metrics
        assert metrics["moe/total_tokens_processed"] == num_tokens
    
    def test_multiple_adds(self):
        """Test accumulating metrics from multiple forward passes."""
        num_tokens = 50
        num_experts = 8
        num_batches = 3
        top_k = 2
        
        collector = MoEMetricsCollector(top_k=top_k)
        
        for _ in range(num_batches):
            router_logits = torch.randn(num_tokens, num_experts)
            collector.add(router_logits)
        
        metrics = collector.compute_and_reset()
        
        # Total tokens should be accumulated
        assert metrics["moe/total_tokens_processed"] == num_tokens * num_batches
    
    def test_add_with_list_of_tensors(self):
        """Test adding router logits as a list (multiple MoE layers)."""
        num_tokens = 50
        num_experts = 8
        num_layers = 4
        top_k = 2
        
        collector = MoEMetricsCollector(top_k=top_k)
        
        router_logits_list = [torch.randn(num_tokens, num_experts) for _ in range(num_layers)]
        collector.add(router_logits_list)
        
        metrics = collector.compute_and_reset()
        
        assert metrics["moe/num_layers_processed"] == num_layers
    
    def test_add_none_is_safe(self):
        """Test that adding None router_logits is handled gracefully."""
        collector = MoEMetricsCollector(top_k=2)
        
        # Should not raise
        collector.add(None)
        
        metrics = collector.compute_and_reset()
        
        # Should return empty dict
        assert metrics == {}
    
    def test_add_empty_list_is_safe(self):
        """Test that adding empty list is handled gracefully."""
        collector = MoEMetricsCollector(top_k=2)
        
        collector.add([])
        
        metrics = collector.compute_and_reset()
        assert metrics == {}
    
    def test_reset(self):
        """Test that reset clears all accumulated data."""
        num_tokens = 50
        num_experts = 8
        top_k = 2
        
        collector = MoEMetricsCollector(top_k=top_k)
        
        router_logits = torch.randn(num_tokens, num_experts)
        collector.add(router_logits)
        
        collector.reset()
        
        metrics = collector.compute_and_reset()
        assert metrics == {}
    
    def test_compute_and_reset_resets(self):
        """Test that compute_and_reset properly resets the collector."""
        num_tokens = 50
        num_experts = 8
        top_k = 2
        
        collector = MoEMetricsCollector(top_k=top_k)
        
        router_logits = torch.randn(num_tokens, num_experts)
        collector.add(router_logits)
        
        metrics1 = collector.compute_and_reset()
        assert metrics1 != {}
        
        # Second call should return empty without new data
        metrics2 = collector.compute_and_reset()
        assert metrics2 == {}


class TestDenseModelCompatibility:
    """Tests to ensure MoE metrics code handles dense models gracefully."""
    
    def test_compute_metrics_from_output_without_router_logits(self):
        """Test that compute_moe_metrics_from_model_output returns empty dict for dense models."""
        # Mock model output without router_logits (dense model)
        @dataclass
        class DenseModelOutput:
            logits: torch.Tensor
            hidden_states: Optional[tuple] = None
        
        output = DenseModelOutput(logits=torch.randn(2, 10, 1000))
        
        metrics = compute_moe_metrics_from_model_output(output, top_k=2)
        
        assert metrics == {}
    
    def test_compute_metrics_from_output_with_none_router_logits(self):
        """Test handling of model output with router_logits=None."""
        @dataclass
        class ModelOutput:
            logits: torch.Tensor
            router_logits: Optional[tuple] = None
        
        output = ModelOutput(logits=torch.randn(2, 10, 1000), router_logits=None)
        
        metrics = compute_moe_metrics_from_model_output(output, top_k=2)
        
        assert metrics == {}
    
    def test_collector_handles_dense_model_output(self):
        """Test that collector handles dense model outputs gracefully."""
        collector = MoEMetricsCollector(top_k=2)
        
        # Simulate multiple forward passes with None router_logits
        for _ in range(5):
            collector.add(None)
        
        metrics = collector.compute_and_reset()
        assert metrics == {}


class TestMoEModelSimulation:
    """
    Integration tests simulating MoE model behavior during training.
    
    These tests simulate what happens in FlowRLActor.update_policy when
    collecting MoE metrics across multiple micro-batches.
    """
    
    def test_training_loop_simulation(self):
        """Simulate a complete training update with MoE metrics collection."""
        # Configuration
        num_experts = 64
        top_k = 8
        num_micro_batches = 4
        batch_size = 4
        seq_len = 512
        num_moe_layers = 28  # Typical for large MoE models
        
        collector = MoEMetricsCollector(top_k=top_k, num_experts=num_experts)
        
        # Simulate forward passes for each micro-batch
        for micro_batch_idx in range(num_micro_batches):
            # Each MoE layer produces router_logits
            router_logits_list = []
            for layer_idx in range(num_moe_layers):
                # Simulate router logits with some structure
                # In real models, these would show actual routing patterns
                router_logits = torch.randn(batch_size * seq_len, num_experts)
                
                # Add some bias to simulate non-uniform routing
                expert_biases = torch.randn(num_experts) * 0.5
                router_logits = router_logits + expert_biases
                
                router_logits_list.append(router_logits)
            
            collector.add(router_logits_list)
        
        # Compute final metrics
        metrics = collector.compute_and_reset()
        
        # Validate metrics
        assert "moe/expert_load_entropy" in metrics
        assert "moe/expert_utilization_ratio" in metrics
        assert "moe/load_balance_loss" in metrics
        assert "moe/router_confidence_mean" in metrics
        
        # Check reasonable values
        assert metrics["moe/expert_utilization_ratio"] > 0.9  # Most experts should be used
        assert metrics["moe/expert_load_entropy"] > 0
        assert metrics["moe/total_tokens_processed"] == num_micro_batches * num_moe_layers * batch_size * seq_len
        
        print("\n=== Training Loop Simulation Results ===")
        for key, value in sorted(metrics.items()):
            print(f"  {key}: {value:.6f}")
    
    def test_routing_collapse_detection(self):
        """Test that metrics can detect routing collapse (all tokens to few experts)."""
        num_experts = 64
        top_k = 8
        num_tokens = 1000
        
        # Simulate routing collapse: all tokens go to first few experts
        router_logits = torch.zeros(num_tokens, num_experts)
        router_logits[:, :top_k] = 10.0  # Only first top_k experts get high logits
        
        collector = MoEMetricsCollector(top_k=top_k, num_experts=num_experts)
        collector.add(router_logits)
        metrics = collector.compute_and_reset()
        
        # In collapse scenario:
        # - Low entropy (concentrated distribution)
        # - Low utilization ratio (few experts used)
        # - High load imbalance
        
        assert metrics["moe/expert_utilization_ratio"] == top_k / num_experts  # Only top_k experts used
        assert metrics["moe/num_active_experts"] == top_k
        
        print("\n=== Routing Collapse Detection ===")
        print(f"  Expert utilization: {metrics['moe/expert_utilization_ratio']:.4f}")
        print(f"  Active experts: {metrics['moe/num_active_experts']:.0f} / {num_experts}")
        print(f"  Load imbalance ratio: {metrics['moe/load_imbalance_ratio']:.4f}")
    
    def test_healthy_routing_detection(self):
        """Test metrics for healthy (well-distributed) routing."""
        num_experts = 64
        top_k = 8
        num_tokens = 10000
        
        # Simulate healthy routing: uniform distribution across experts
        router_logits = torch.randn(num_tokens, num_experts)  # Random = roughly uniform
        
        collector = MoEMetricsCollector(top_k=top_k, num_experts=num_experts)
        collector.add(router_logits)
        metrics = collector.compute_and_reset()
        
        # In healthy scenario:
        # - High entropy (spread distribution)
        # - High utilization ratio (all experts used)
        # - Low load imbalance
        
        assert metrics["moe/expert_utilization_ratio"] == 1.0  # All experts should be used
        assert metrics["moe/expert_load_entropy_normalized"] > 0.9  # Close to maximum entropy
        
        print("\n=== Healthy Routing Detection ===")
        print(f"  Expert utilization: {metrics['moe/expert_utilization_ratio']:.4f}")
        print(f"  Normalized entropy: {metrics['moe/expert_load_entropy_normalized']:.4f}")
        print(f"  Load imbalance ratio: {metrics['moe/load_imbalance_ratio']:.4f}")


class TestFlowRLActorMoEDetection:
    """
    Tests simulating the MoE detection logic in FlowRLActor.__init__
    """
    
    def test_detect_qwen3_moe_model(self):
        """Simulate detection of Qwen3-30B-A3B (MoE model)."""
        # Mock config for Qwen3-30B-A3B
        @dataclass
        class Qwen3MoEConfig:
            model_type: str = "qwen3_moe"
            num_experts_per_tok: int = 8
            num_local_experts: int = 128
            hidden_size: int = 4096
        
        config = Qwen3MoEConfig()
        
        # Detection logic from FlowRLActor
        is_moe_model = False
        moe_top_k = 8
        
        if hasattr(config, "num_experts_per_tok") and config.num_experts_per_tok is not None:
            is_moe_model = True
            moe_top_k = config.num_experts_per_tok
        elif hasattr(config, "num_local_experts") and config.num_local_experts is not None:
            is_moe_model = True
        
        model_type = getattr(config, "model_type", "")
        if "moe" in model_type.lower() or "mixtral" in model_type.lower():
            is_moe_model = True
        
        assert is_moe_model == True
        assert moe_top_k == 8
        print(f"\n[Test] Qwen3-30B-A3B detected as MoE: {is_moe_model}, top_k={moe_top_k}")
    
    def test_detect_qwen3_dense_model(self):
        """Simulate detection of Qwen3-8B (dense model)."""
        # Mock config for Qwen3-8B (dense)
        @dataclass
        class Qwen3DenseConfig:
            model_type: str = "qwen3"
            hidden_size: int = 4096
            num_attention_heads: int = 32
        
        config = Qwen3DenseConfig()
        
        # Detection logic from FlowRLActor
        is_moe_model = False
        
        if hasattr(config, "num_experts_per_tok") and config.num_experts_per_tok is not None:
            is_moe_model = True
        elif hasattr(config, "num_local_experts") and config.num_local_experts is not None:
            is_moe_model = True
        elif hasattr(config, "n_routed_experts") and config.n_routed_experts is not None:
            is_moe_model = True
        
        model_type = getattr(config, "model_type", "")
        if "moe" in model_type.lower() or "mixtral" in model_type.lower():
            is_moe_model = True
        
        assert is_moe_model == False
        print(f"\n[Test] Qwen3-8B detected as MoE: {is_moe_model}")
    
    def test_detect_mixtral_model(self):
        """Simulate detection of Mixtral model."""
        @dataclass
        class MixtralConfig:
            model_type: str = "mixtral"
            num_local_experts: int = 8
            num_experts_per_tok: int = 2
        
        config = MixtralConfig()
        
        is_moe_model = False
        
        model_type = getattr(config, "model_type", "")
        if "moe" in model_type.lower() or "mixtral" in model_type.lower():
            is_moe_model = True
        
        assert is_moe_model == True
        print(f"\n[Test] Mixtral detected as MoE: {is_moe_model}")


if __name__ == "__main__":
    # Run specific test classes for quick verification
    print("=" * 60)
    print("Running MoE Metrics Unit Tests (CPU only)")
    print("=" * 60)
    
    # Test basic functionality
    print("\n>>> Testing basic expert load computation...")
    test_load = TestComputeExpertLoad()
    test_load.test_basic_expert_load()
    test_load.test_expert_load_with_batch_dimension()
    test_load.test_expert_load_with_attention_mask()
    print("✅ Expert load computation tests passed")
    
    # Test load balance loss
    print("\n>>> Testing load balance loss...")
    test_lb = TestLoadBalanceLoss()
    test_lb.test_perfectly_balanced_load()
    test_lb.test_completely_imbalanced_load()
    test_lb.test_loss_comparison()
    print("✅ Load balance loss tests passed")
    
    # Test metrics computation
    print("\n>>> Testing MoE metrics computation...")
    test_metrics = TestComputeMoEMetrics()
    test_metrics.test_metrics_computation()
    test_metrics.test_entropy_range()
    test_metrics.test_utilization_ratio_range()
    print("✅ MoE metrics computation tests passed")
    
    # Test collector
    print("\n>>> Testing MoEMetricsCollector...")
    test_collector = TestMoEMetricsCollector()
    test_collector.test_single_add()
    test_collector.test_multiple_adds()
    test_collector.test_add_with_list_of_tensors()
    test_collector.test_add_none_is_safe()
    test_collector.test_add_empty_list_is_safe()
    test_collector.test_reset()
    test_collector.test_compute_and_reset_resets()
    print("✅ MoEMetricsCollector tests passed")
    
    # Test dense model compatibility
    print("\n>>> Testing dense model compatibility...")
    test_dense = TestDenseModelCompatibility()
    test_dense.test_compute_metrics_from_output_without_router_logits()
    test_dense.test_compute_metrics_from_output_with_none_router_logits()
    test_dense.test_collector_handles_dense_model_output()
    print("✅ Dense model compatibility tests passed")
    
    # Test MoE model simulation
    print("\n>>> Testing MoE model simulation...")
    test_sim = TestMoEModelSimulation()
    test_sim.test_training_loop_simulation()
    test_sim.test_routing_collapse_detection()
    test_sim.test_healthy_routing_detection()
    print("✅ MoE model simulation tests passed")
    
    # Test MoE detection logic
    print("\n>>> Testing FlowRLActor MoE detection logic...")
    test_detect = TestFlowRLActorMoEDetection()
    test_detect.test_detect_qwen3_moe_model()
    test_detect.test_detect_qwen3_dense_model()
    test_detect.test_detect_mixtral_model()
    print("✅ MoE detection logic tests passed")
    
    print("\n" + "=" * 60)
    print("🎉 All MoE metrics tests passed successfully!")
    print("=" * 60)
