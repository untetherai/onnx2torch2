# pylint: disable=missing-docstring
__all__ = [
    'OnnxIsNaN',
]

import torch
from torch import nn

from onnx2torch2.node_converters.registry import add_converter
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.utils.common import OnnxMapping
from onnx2torch2.utils.common import OnnxToTorchModule
from onnx2torch2.utils.common import OperationConverterResult


class OnnxIsNaN(nn.Module, OnnxToTorchModule):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return torch.isnan(input_tensor)


@add_converter(operation_type='IsNaN', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    torch_module = OnnxIsNaN()

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=(node.input_values[0],),
            outputs=node.output_values,
        ),
    )
