# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

import torch
from executorch.exir.pass_base import ExportPass, PassResult
from executorch.exir.schema import TensorShapeDynamism
from executorch.exir.tensor import TensorSpec


class MarkDynamicUnboundPass(ExportPass):
    """
    Marks mutable buffer TensorSpecs matching given name patterns as
    DYNAMIC_UNBOUND. This causes the memory planner to skip them (no
    allocation_info in the flatbuffer), and the runtime will allocate their
    memory lazily via DynamicAllocator.

    Typical usage: mark KV cache buffers so they start unallocated and grow
    on demand, avoiding the full upfront memory cost of max_context_length.
    """

    def __init__(
        self,
        name_patterns: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.name_patterns = name_patterns or ["k_cache", "v_cache"]

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        modified = False
        for node in graph_module.graph.nodes:
            if node.op != "placeholder":
                continue
            spec = node.meta.get("spec")
            if not isinstance(spec, TensorSpec):
                continue
            if not spec.const:
                # Only process mutable buffers (const=False means mutable).
                name = node.name
                if any(pattern in name for pattern in self.name_patterns):
                    spec.shape_dynamism = TensorShapeDynamism.DYNAMIC_UNBOUND
                    modified = True
        return PassResult(graph_module, modified)
