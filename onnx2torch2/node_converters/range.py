# pylint: disable=missing-docstring
__all__ = [
    'OnnxRange',
]

from typing import Union

import torch
from torch import nn

from onnx2torch2.node_converters.registry import add_converter
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.utils.common import OperationConverterResult
from onnx2torch2.utils.common import onnx_mapping_from_node
from onnx2torch2.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch2.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


class OnnxRange(nn.Module, OnnxToTorchModuleWithCustomExport):
    def __init__(self):
        super().__init__()
        self.register_buffer('dummy_buffer', torch.Tensor(), persistent=False)

    @staticmethod
    def _get_scalar(value) -> Union[float, int]:
        if isinstance(value, torch.Tensor):
            return value.item()

        return value

    def _arange(
        self,
        start: Union[torch.Tensor, float, int],
        limit: Union[torch.Tensor, float, int],
        delta: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        return torch.arange(
            start=self._get_scalar(start),
            end=self._get_scalar(limit),
            step=self._get_scalar(delta),
            device=self.dummy_buffer.device,
        )

    def forward(
        self,
        start: Union[torch.Tensor, float, int],
        limit: Union[torch.Tensor, float, int],
        delta: Union[torch.Tensor, float, int],
    ) -> torch.Tensor:
        def _forward() -> torch.Tensor:
            return self._arange(start, limit, delta)

        if torch.onnx.is_in_onnx_export():
            return DefaultExportToOnnx.export(_forward, 'Range', start, limit, delta, {})

        return _forward()


@add_converter(operation_type='Range', version=11)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:
    del graph
    return OperationConverterResult(
        torch_module=OnnxRange(),
        onnx_mapping=onnx_mapping_from_node(node),
    )
