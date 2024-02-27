__all__ = [
    'OnnxSlice',
]

from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch import nn

from onnx2torch.node_converters.registry import add_converter
from onnx2torch.onnx_graph import OnnxGraph
from onnx2torch.onnx_node import OnnxNode
from onnx2torch.utils.common import OnnxToTorchModule
from onnx2torch.utils.common import OperationConverterResult
from onnx2torch.utils.common import onnx_mapping_from_node, get_const_value
from onnx2torch.utils.custom_export_to_onnx import DefaultExportToOnnx
from onnx2torch.utils.custom_export_to_onnx import OnnxToTorchModuleWithCustomExport


def _get_slices(
    starts: List,
    ends: List,
    axes: List,
    steps: List,
) -> Tuple[List, List, List]:
    if not axes:
        axes = list(range(len(starts)))

    if not steps:
        steps = [1] * len(starts)
    
    slices = {}
    flip_dims = []
    for start, end, axis, step in zip(starts, ends, axes, steps):
        if step < 0:
            flip_dims.append(axis)
            start, end, step = -start - 1, -end - 1, -step

        slices[axis] = slice(start, end, step)

    pos_axes_slices = list(slices.get(a, slice(None, None)) for a in range(max(axes) + 1))
    neg_axes_slices = list(slices.get(a, slice(None, None)) for a in range(min(axes), 0))

    if neg_axes_slices:
        neg_axes_slices = [Ellipsis] + neg_axes_slices

    return flip_dims, pos_axes_slices, neg_axes_slices


def _do_slice(x: torch.Tensor, flip_dims: List, pos_axes_slices: List, neg_axes_slices: List):
    if flip_dims:
        x = torch.flip(x, dims=flip_dims)

    if pos_axes_slices:
        x = x[pos_axes_slices]

    if neg_axes_slices:
        x = x[neg_axes_slices]

    return x


class OnnxSliceV9(nn.Module, OnnxToTorchModule):  # pylint: disable=missing-class-docstring
    def __init__(self, starts: np.ndarray, ends: np.ndarray, axes: Optional[np.ndarray] = None):
        super().__init__()
        self._flip_dims, self._pos_axes_slices, self._neg_axes_slices = _get_slices(starts, ends, axes, None)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:  # pylint: disable=missing-function-docstring
        return _do_slice(input_tensor, self.flip_dims, self.pos_axes_slices, self.neg_axes_slices)


class OnnxSlice(nn.Module, OnnxToTorchModuleWithCustomExport):  # pylint: disable=missing-class-docstring
    def __init__(self, starts: List, ends: List, axes: List, steps: List):
        super().__init__()
        self.starts = starts
        self.ends = ends
        self.axes = axes
        self.steps = steps
        
    def forward(  # pylint: disable=missing-function-docstring
        self,
        input_tensor: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        axes: Optional[torch.Tensor] = None,
        steps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def _forward():
            starts_list = starts.tolist() if starts != None and not self.starts else self.starts
            ends_list = ends.tolist() if ends != None and not self.ends else self.ends
            axes_list = axes.tolist() if axes != None and not self.axes else self.axes
            steps_list = steps.tolist() if steps != None and not self.steps else self.steps
            flip_dims, pos_axes_slices, neg_axes_slices = _get_slices(starts_list, ends_list, axes_list, steps_list)
            return _do_slice(input_tensor, flip_dims, pos_axes_slices, neg_axes_slices)

        if torch.onnx.is_in_onnx_export():
            args = [input_tensor, starts, ends]
            if axes is not None:
                args.append(axes)
            if steps is not None:
                args.append(steps)

            return DefaultExportToOnnx.export(_forward, 'Slice', *args, {})
        
        return _forward()


@add_converter(operation_type='Slice', version=9)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    node_attributes = node.attributes
    return OperationConverterResult(
        torch_module=OnnxSliceV9(
            starts=node_attributes['starts'],
            ends=node_attributes['ends'],
            axes=node_attributes.get('axes', None),
        ),
        onnx_mapping=onnx_mapping_from_node(node),
    )


@add_converter(operation_type='Slice', version=10)
@add_converter(operation_type='Slice', version=11)
@add_converter(operation_type='Slice', version=13)
def _(node: OnnxNode, graph: OnnxGraph) -> OperationConverterResult:  # pylint: disable=unused-argument
    constant_params = []
    params_names = node.input_values[1:]
    for name in params_names:
        try:
            constant_params.append(get_const_value(name, graph).tolist())
        except KeyError:
            constant_params.append([])
    constant_params = constant_params + (4 - len(constant_params)) * [[]]


    return OperationConverterResult(
        torch_module=OnnxSlice(*constant_params),
        onnx_mapping=onnx_mapping_from_node(node),
    )
