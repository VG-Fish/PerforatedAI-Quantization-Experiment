import unittest

import torch
from torch import nn

from dendritic_benchmark.capacity_control import (
    UnsupportedTopology,
    apply_capacity_dense_control,
    dense_backbone_state_from_pai,
    retained_topology_from_state_dict,
)


class _Toy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 2)

    def forward(self, x):
        return self.proj(x)


class CapacityControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.state = {
            "proj.dendrite_module.parent_module.weight": torch.zeros(2, 3),
            "proj.dendrite_module.parent_module.bias": torch.zeros(2),
            "proj.dendrite_module.layers.0.weight": torch.zeros(2, 3),
            "proj.dendrite_module.layers.0.bias": torch.zeros(2),
            "proj.dendrites_to_top.0": torch.zeros(1, 2),
        }

    def test_extracts_and_builds_exact_one_branch_linear_control(self) -> None:
        topology = retained_topology_from_state_dict(self.state)
        self.assertEqual(topology.branches[0].target_path, "proj")
        model = _Toy()
        base = sum(p.numel() for p in model.parameters())
        controlled = apply_capacity_dense_control(model, topology, seed=0)
        self.assertEqual(sum(p.numel() for p in controlled.parameters()), base + 10)
        self.assertEqual(tuple(controlled(torch.randn(4, 3)).shape), (4, 2))

    def test_parent_state_is_restored_to_dense_names(self) -> None:
        dense = dense_backbone_state_from_pai(self.state)
        self.assertEqual(set(dense), {"proj.weight", "proj.bias"})

    def test_rejects_multiple_retained_branches(self) -> None:
        bad = dict(self.state)
        bad["proj.dendrite_module.layers.1.weight"] = torch.zeros(2, 3)
        with self.assertRaises(UnsupportedTopology):
            retained_topology_from_state_dict(bad)

