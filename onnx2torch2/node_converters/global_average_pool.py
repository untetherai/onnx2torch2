# pylint: disable=missing-docstring
__all__ = [
    'OnnxGlobalAveragePool',
    'OnnxGlobalAveragePoolWithKnownInputShape',
]

from typing import List

import torch
from torch import nn

from onnx2torch2.node_converters.registry import add_converter
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.utils.common import OperationConverterResult
from onnx2torch2.utils.common import get_shape_from_value_info
from onnx2torch2.utils.common import onnx_mapping_from_node
from onnx2torch2.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch2.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxGlobalAveragePool(nn.Module, OnnxToTorchModuleWithCustomExport):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        def _forward():
            x_dims = list(range(2, len(input_tensor.shape)))
            return torch.mean(input_tensor, dim=x_dims, keepdim=True)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(_forward, 'GlobalAveragePool', input_tensor, {})

        return _forward()


class OnnxGlobalAveragePoolWithKnownInputShape(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(self, input_shape: List[int]):
        super().__init__()
        self._x_dims = list(range(2, len(input_shape)))

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            return torch.mean(input_tensor, dim=self._x_dims, keepdim=True)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(_forward, 'GlobalAveragePool', input_tensor, {})

        return _forward()


@add_converter(operation_type='GlobalAveragePool', version=1)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    input_value_info = graph.value_info[node.input_values[0]]
    input_shape = get_shape_from_value_info(input_value_info)

    if input_shape is not None:
        torch_module = OnnxGlobalAveragePoolWithKnownInputShape(input_shape=input_shape)
    else:
        torch_module = OnnxGlobalAveragePool()

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(node=node),
    )
