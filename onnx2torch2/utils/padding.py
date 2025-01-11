import math
from typing import Optional
from typing import Tuple
from typing import Union

from torch import nn
from onnx2torch2.onnx_graph import OnnxGraph
from onnx2torch2.onnx_node import OnnxNode
from onnx2torch2.node_converters.pad import OnnxPadStatic


def is_symmetric_onnx_padding(padding: Tuple[int, ...]) -> bool:  # pylint: disable=missing-function-docstring
    half_len = len(padding) // 2
    return padding[:half_len] == padding[half_len:]


def onnx_auto_pad_to_torch_padding(  # pylint: disable=missing-function-docstring
    auto_pad: str,
    onnx_padding: Tuple[int, ...],
    node: OnnxNode | None = None,
    graph: OnnxGraph | None = None,
) -> Tuple[Union[int, Tuple[int, ...]], Optional[nn.Module]]:
    if auto_pad == 'NOTSET':
        if onnx_padding is None:
            return 0, None

        if is_symmetric_onnx_padding(onnx_padding):
            half_len = len(onnx_padding) // 2
            return onnx_padding[:half_len], None

        return 0, OnnxPadStatic.create_from_onnx_params(onnx_pads=onnx_padding)

    if auto_pad == 'VALID':
        return 0, None

    if auto_pad in ('SAME_UPPER', 'SAME_LOWER') and node and graph and node.operation_type == 'Conv':
        input_name = node.input_values[0]
        if input_name in graph.value_info:
            va = graph.value_info[input_name]
            input_shape = [tmp_dim.dim_value for tmp_dim in va.type.tensor_type.shape.dim][2:]
            weights = graph.initializers[node.input_values[1]].to_torch()
            spatial_rank = len(weights.shape) - 2
            node_attributes = node.attributes
            strides = node_attributes.get('strides', [1] * spatial_rank)
            dilations = node_attributes.get('dilations', [1] * spatial_rank)
            kernel_size = node_attributes.get('kernel_shape', weights.shape[2:])
            # infer output shape when auto_pad enabled
            output_shape = [math.ceil(inp / st) for inp, st in zip(input_shape, strides)]
        else:
            raise ValueError(f"input_name:{input_name} not in graph")
        # infer paddings when auto_pad enabled
        paddings = [
            ((out - 1) * s - (inp - k - d + 1))
            for inp, out, k, s, d in zip(input_shape, output_shape, kernel_size, strides, dilations, strict=True)
        ]
        pad_res = [p % 2 for p in paddings]
        if sum(pad_res) == 0:  # all are even paddings
            return [p // 2 for p in paddings], None
        else:
            raise NotImplementedError(f'"{auto_pad}" auto_pad is not implemented with odd padding')

    raise ValueError(f'Got unexpected auto_pad value "{auto_pad}"')
