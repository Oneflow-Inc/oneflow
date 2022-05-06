## Ops Version : Alpha


|op name   | Doc Test | Compatiable/Completeness Test | Exception |
| ------------------------- | ------------- | ----------------------------- | --------- |
| oneflow.Tensor | [oneflow.tensor](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L20)   | [tensor_scatter_nd_update](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_tensor_scatter_nd_update.py#L91)   |  |
| oneflow.BoolTensor |  |  |  |
| oneflow.ByteTensor |  |  |  |
| oneflow.CharTensor |  |  |  |
| oneflow.DoubleTensor |  |  |  |
| oneflow.FloatTensor |  |  |  |
| oneflow.HalfTensor |  |  |  |
| oneflow.IntTensor |  |  |  |
| oneflow.LongTensor |  |  |  |
| oneflow.Size | [oneflow.Tensor.size](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1127)   |  |  |
| oneflow.abs | [oneflow.Tensor.abs](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L471)   | [abs_with_ndim_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_abs.py#L34)   |  |
| oneflow.acos | [oneflow.Tensor.acos](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L478)   | [acos_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L348)   |  |
| oneflow.acosh | [oneflow.Tensor.acosh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L492)   | [acosh_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L368)   |  |
| oneflow.adaptive_avg_pool1d |  |  |  |
| oneflow.adaptive_avg_pool2d |  |  |  |
| oneflow.adaptive_avg_pool3d |  |  |  |
| oneflow.add | [oneflow.Tensor.add](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L985)   | [add_with_alpha](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_add.py#L198)   |  |
| oneflow.addmm | [oneflow.Tensor.addmm](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L992)   | [addmm](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_addmm.py#L60)   |  |
| oneflow.any |  | [any_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_logical_reduce.py#L47)   |  |
| oneflow.arange | [oneflow.arange](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/arange.py#L20)   | [arange](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_arange.py#L58)   |  |
| oneflow.arccos | [oneflow.Tensor.arccos](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L485)   | [arccos_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L338)   |  |
| oneflow.arccosh | [oneflow.Tensor.arccosh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L499)   | [arccosh_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L358)   |  |
| oneflow.arcsin | [oneflow.Tensor.arcsin](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1013)   |  |  |
| oneflow.arcsinh | [oneflow.Tensor.arcsinh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1020)   |  |  |
| oneflow.arctan | [oneflow.Tensor.arctanh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L506)   | [arctan_tensor_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_2.py#L440)   |  |
| oneflow.arctanh | [oneflow.Tensor.arctanh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L506)   | [arctanh_tensor_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_2.py#L462)   |  |
| oneflow.argmax | [oneflow.argmax](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L139)   | [argmax](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_argmax.py#L83)   |  |
| oneflow.argmin | [oneflow.argmin](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L169)   | [argmin](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_argmin.py#L34)   |  |
| oneflow.argsort | [oneflow.Tensor.argsort](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L527)   | [argsort](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_argsort.py#L36)   |  |
| oneflow.argwhere | [oneflow.Tensor.argwhere](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L534)   | [argwhere](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_1.py#L625)   |  |
| oneflow.as_strided |  |  |  |
| oneflow.as_tensor |  |  |  |
| oneflow.asin | [oneflow.Tensor.asin](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1006)   |  |  |
| oneflow.asinh | [oneflow.asinh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L298)   |  |  |
| oneflow.atan | [oneflow.Tensor.atan2](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L122)   | [atanh_tensor_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_2.py#L412)   |  |
| oneflow.atan2 | [oneflow.Tensor.atan2](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L122)   |  |  |
| oneflow.atanh | [oneflow.Tensor.atanh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L541)   | [atanh_tensor_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_2.py#L412)   |  |
| oneflow.autograd |  | [autograd_interface](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_autograd.py#L81)   |  |
| oneflow.batch_gather |  |  |  |
| oneflow.bernoulli | [oneflow.bernoulli](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/random.py#L20)   | [bernoulli](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_bernoulli.py#L49)   |  |
| oneflow.bfloat16 |  |  |  |
| oneflow.bmm | [oneflow.Tensor.bmm](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L695)   | [bmm](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_bmm.py#L93)   |  |
| oneflow.bool |  | [bool_add](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_add.py#L212)   |  |
| oneflow.boxing |  |  |  |
| oneflow.broadcast_like |  |  |  |
| oneflow.cast | [oneflow.broadcast_like](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/broadcast_like.py#L20)   | [cast](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_flatten.py#L63)   |  |
| oneflow.cat | [oneflow.cat](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L333)   | [scatter_1n4d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_comm_ops.py#L84)   |  |
| oneflow.ceil | [oneflow.Tensor.ceil](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1440)   | [ceil_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_ceil.py#L29)   |  |
| oneflow.char |  |  |  |
| oneflow.chunk | [oneflow.Tensor.chunk](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L702)   |  |  |
| oneflow.clamp | [oneflow.Tensor.clamp](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1266)   | [clamp](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_clamp.py#L96)   |  |
| oneflow.clamp_ |  |  |  |
| oneflow.clip | [oneflow.Tensor.clip](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1280)   | [clip_grad](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_clip_grad.py#L152)   |  |
| oneflow.clip_ |  |  |  |
| oneflow.concat |  | [concat](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_concat.py#L124)   |  |
| oneflow.constant_initializer |  |  |  |
| oneflow.convert_oneflow_dtype_to_numpy_dtype |  |  |  |
| oneflow.cos | [oneflow.Tensor.acos](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L478)   | [cos](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L88)   |  |
| oneflow.cosh | [oneflow.Tensor.acosh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L492)   | [arccosh_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L358)   |  |
| oneflow.cumprod | [oneflow.cumprod](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L1576)   | [cumprod](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_cum_ops.py#L37)   |  |
| oneflow.cumsum | [oneflow.cumsum](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L1543)   | [cumsum](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_cumsum.py#L36)   |  |
| oneflow.device | [oneflow.Tensor.device](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L84)   |  |  |
| oneflow.diag | [oneflow.diagonal](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L20)   | [diag](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_diag.py#L35)   |  |
| oneflow.diagonal | [oneflow.diagonal](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L20)   | [diagonal](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_diagonal.py#L43)   |  |
| oneflow.distributed_partial_fc_sample |  |  |  |
| oneflow.div | [oneflow.Tensor.div_](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L893)   | [div](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_1.py#L478)   |  |
| oneflow.div_ |  |  |  |
| oneflow.dot | [oneflow.dot](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L1262)   | [dot](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_dot.py#L26)   |  |
| oneflow.double | [oneflow.Tensor.double](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1673)   | [double](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_tensor_ops.py#L128)   |  |
| oneflow.dtype |  |  |  |
| oneflow.dtypes |  |  |  |
| oneflow.einsum | [oneflow.einsum](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/einsum.py#L20)   | [einsum_bilinear_transformation](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_einsum_bilinear_transformation.py#L42)   |  |
| oneflow.empty |  | [empty_consistent](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_empty.py#L54)   |  |
| oneflow.eq | [oneflow.Tensor.requires_grad](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L621)   | [eq_with_0_size_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_eq.py#L32)   |  |
| oneflow.equal |  |  |  |
| oneflow.erf | [oneflow.Tensor.erf](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L763)   | [erf](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_erf.py#L35)   |  |
| oneflow.erfc | [oneflow.Tensor.erfc](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L772)   | [erfc](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_erfc.py#L35)   |  |
| oneflow.erfinv | [oneflow.Tensor.erfinv](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L781)   | [erfinv_tensor_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_2.py#L702)   |  |
| oneflow.erfinv_ |  |  |  |
| oneflow.exp | [oneflow.Tensor.expand](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L129)   | [expand_broadcast](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_expand_op.py#L208)   |  |
| oneflow.expand | [oneflow.Tensor.expand](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L129)   | [expand_broadcast](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_expand_op.py#L208)   |  |
| oneflow.expm1 | [oneflow.Tensor.expm1](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1447)   | [expm1](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_expm1.py#L46)   |  |
| oneflow.eye | [oneflow.eye](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L1382)   | [eye](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_eye.py#L50)   |  |
| oneflow.flatten | [oneflow.Tensor.flatten](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L154)   | [flatten](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_flatten.py#L38)   |  |
| oneflow.flip | [oneflow.Tensor.flip](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L168)   | [flip](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_flip.py#L40)   |  |
| oneflow.float | [oneflow.Tensor.float](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1652)   | [float](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_tensor_ops.py#L114)   |  |
| oneflow.float16 |  |  |  |
| oneflow.float32 |  |  |  |
| oneflow.float64 |  |  |  |
| oneflow.floor | [oneflow.Tensor.floor](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L161)   | [floor](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_floor.py#L49)   |  |
| oneflow.floor_ |  |  |  |
| oneflow.floor_divide |  |  |  |
| oneflow.fmod | [oneflow.Tensor.fmod](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1370)   | [fmod_with_0_size_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_1.py#L832)   |  |
| oneflow.from_numpy |  |  |  |
| oneflow.full |  | [full_with_random_data_int](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_constant.py#L115)   |  |
| oneflow.gather | [oneflow.gather](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L367)   | [gather_1n4d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_comm_ops.py#L106)   |  |
| oneflow.gather_nd |  |  |  |
| oneflow.ge | [oneflow.gelu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L74)   | [image_normalize](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_image_normalize.py#L75)   |  |
| oneflow.gelu | [oneflow.gelu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L74)   | [gelu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L147)   |  |
| oneflow.glorot_normal_initializer |  |  |  |
| oneflow.glorot_uniform_initializer |  |  |  |
| oneflow.grad_enable |  |  |  |
| oneflow.greater | [oneflow.greater](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/comparison.py#L21)   | [greater_equal](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_greater_equal.py#L38)   |  |
| oneflow.greater_equal |  |  |  |
| oneflow.gt | [oneflow.Tensor.gt](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L857)   |  |  |
| oneflow.half |  |  |  |
| oneflow.hsplit | [oneflow.hsplit](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L1459)   |  |  |
| oneflow.in_top_k |  |  |  |
| oneflow.index_select |  |  |  |
| oneflow.int | [oneflow.Tensor.int](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1610)   | [interpolate](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_interpolate.py#L658)   |  |
| oneflow.int32 |  |  |  |
| oneflow.int64 |  |  |  |
| oneflow.int8 |  |  |  |
| oneflow.is_floating_point |  |  |  |
| oneflow.is_grad_enabled |  |  |  |
| oneflow.is_nonzero |  |  |  |
| oneflow.is_tensor |  |  |  |
| oneflow.kaiming_initializer |  |  |  |
| oneflow.le | [oneflow.tile](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tile.py#L20)   | [upsample2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_upsample.py#L380)   |  |
| oneflow.linalg_flow |  |  |  |
| oneflow.linalg_matrix_norm |  |  |  |
| oneflow.linalg_norm |  |  |  |
| oneflow.linalg_vector_norm |  |  |  |
| oneflow.linspace |  | [linspace_int_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_linspace.py#L32)   |  |
| oneflow.log | [oneflow.Tensor.logical_not](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L355)   | [logical_slice_assign](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_slice.py#L171)   |  |
| oneflow.log1p | [oneflow.Tensor.log1p](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L864)   | [log1p_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_log1p.py#L31)   |  |
| oneflow.log2 | [oneflow.log2](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L928)   |  |  |
| oneflow.log_softmax |  |  |  |
| oneflow.logical_and |  |  |  |
| oneflow.logical_not |  |  |  |
| oneflow.logical_or |  |  |  |
| oneflow.logical_slice |  |  |  |
| oneflow.logical_slice_assign |  |  |  |
| oneflow.logical_xor |  |  |  |
| oneflow.long | [oneflow.Tensor.long](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1631)   | [long](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_tensor_ops.py#L86)   |  |
| oneflow.lt | [oneflow.Tensor.lt](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L802)   | [multistep_lr](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_lr_scheduler.py#L160)   |  |
| oneflow.manual_seed |  |  |  |
| oneflow.masked_fill |  |  |  |
| oneflow.masked_select |  |  |  |
| oneflow.matmul | [oneflow.Tensor.matmul](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L443)   | [matmul](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_matmul.py#L42)   |  |
| oneflow.max | [oneflow.argmax](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L139)   | [maxpool](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_maxpool.py#L219)   |  |
| oneflow.maximum | [oneflow.maximum](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L977)   | [maximum_minimum_with_same_input](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_maximum_minimum.py#L93)   |  |
| oneflow.mean | [oneflow.Tensor.mean](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1504)   | [mean](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_mean.py#L33)   |  |
| oneflow.meshgrid | [oneflow.meshgrid](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/meshgrid.py#L20)   | [meshgrid](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_meshgrid.py#L68)   |  |
| oneflow.min | [oneflow.argmin](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L169)   | [min_max_observer](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_min_max_observer.py#L136)   |  |
| oneflow.minimum | [oneflow.minimum](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L955)   |  |  |
| oneflow.mish | [oneflow.mish](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L254)   | [mish_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L182)   |  |
| oneflow.movedim | [oneflow.movedim](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L1320)   | [movedim](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_movedim.py#L37)   |  |
| oneflow.mul | [oneflow.Tensor.matmul](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L443)   | [mul_with_scalar](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_mul.py#L47)   |  |
| oneflow.narrow | [oneflow.Tensor.narrow](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L450)   | [narrow](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_narrow.py#L34)   |  |
| oneflow.ne | [oneflow.decode_onerec](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/dataset.py#L20)   | [ones_like](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_ones_like.py#L53)   |  |
| oneflow.neg | [oneflow.Tensor.negative](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L907)   | [negative_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_negative.py#L42)   |  |
| oneflow.negative | [oneflow.Tensor.negative](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L907)   | [negative_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_negative.py#L42)   |  |
| oneflow.new_ones |  |  |  |
| oneflow.nms | [oneflow.Tensor.nms](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1461)   | [nms](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_nms.py#L91)   |  |
| oneflow.no_grad |  |  |  |
| oneflow.nonzero | [oneflow.nonzero](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/nonzero.py#L20)   | [nonzero](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_nozero.py#L31)   |  |
| oneflow.not_equal |  |  |  |
| oneflow.numel | [oneflow.Tensor.numel](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L193)   |  |  |
| oneflow.one_embedding |  |  |  |
| oneflow.ones | [oneflow.ones_like](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/constant.py#L20)   | [ones_like](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_ones_like.py#L53)   |  |
| oneflow.ones_initializer |  |  |  |
| oneflow.ones_like |  |  |  |
| oneflow.pad |  | [ConstantPad2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_zeropad2d.py#L96)   |  |
| oneflow.permute | [oneflow.Tensor.permute](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L464)   | [permute4d_tensor_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_contiguous.py#L69)   |  |
| oneflow.placement | [oneflow.Tensor.placement](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L94)   |  |  |
| oneflow.pow | [oneflow.Tensor.pow](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L950)   | [pow_float_scalar_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L163)   |  |
| oneflow.prod | [oneflow.Tensor.prod](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1513)   | [cumprod](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_cum_ops.py#L37)   |  |
| oneflow.randint |  | [randint_consistent](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_randint.py#L56)   |  |
| oneflow.randn |  | [randn](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_randn.py#L86)   |  |
| oneflow.random_normal_initializer |  |  |  |
| oneflow.random_uniform_initializer |  |  |  |
| oneflow.randperm |  | [randperm](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_randperm.py#L86)   |  |
| oneflow.reciprocal | [oneflow.Tensor.reciprocal](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L978)   |  |  |
| oneflow.relu | [oneflow.relu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L50)   | [prelu_4dim_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_prelu.py#L32)   |  |
| oneflow.repeat | [oneflow.Tensor.repeat](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1334)   |  |  |
| oneflow.reshape | [oneflow.Tensor.reshape](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1522)   | [reshape](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_reshape.py#L59)   | [reshape_exception_only_one_dim_infered](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/exceptions/test_reshape.py#L25)   |
| oneflow.roi_align |  |  |  |
| oneflow.roll | [oneflow.Tensor.roll](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L964)   |  |  |
| oneflow.round | [oneflow.Tensor.round](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L971)   | [round_tensor_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_2.py#L724)   |  |
| oneflow.rsqrt | [oneflow.Tensor.rsqrt](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1064)   | [rsqrt_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L136)   |  |
| oneflow.save |  | [save_state_dict](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_module.py#L179)   |  |
| oneflow.sbp | [oneflow.Tensor.sbp](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L101)   | [sbp_symbol](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_sbp_symbol.py#L23)   |  |
| oneflow.scatter |  | [scatter_1n4d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_comm_ops.py#L84)   |  |
| oneflow.scatter_add |  |  |  |
| oneflow.select | [oneflow.select](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L1291)   |  |  |
| oneflow.selu | [oneflow.selu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L396)   | [selu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L192)   |  |
| oneflow.set_num_threads |  |  |  |
| oneflow.set_printoptions |  |  |  |
| oneflow.set_rng_state |  |  |  |
| oneflow.sigmoid | [oneflow.sigmoid](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L325)   | [sigmoid_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L152)   |  |
| oneflow.sign | [oneflow.Tensor.sign](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1106)   | [sign](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_sign.py#L45)   |  |
| oneflow.silu | [oneflow.silu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L224)   | [silu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L187)   |  |
| oneflow.sin | [oneflow.Tensor.asin](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1006)   | [cosine_decay_lr](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_lr_scheduler.py#L82)   |  |
| oneflow.sin_ |  |  |  |
| oneflow.sinh | [oneflow.Tensor.arcsinh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1020)   |  |  |
| oneflow.slice |  | [slice](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_slice.py#L133)   |  |
| oneflow.softmax | [oneflow.Tensor.softmax](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1141)   | [softmax_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L395)   |  |
| oneflow.softplus | [oneflow.softplus](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L133)   | [softplus](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L502)   |  |
| oneflow.softshrink |  | [softshrink_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L207)   |  |
| oneflow.softsign | [oneflow.Tensor.softsign](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1155)   | [softsign_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L685)   |  |
| oneflow.sort | [oneflow.sort](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/sort.py#L20)   | [argsort](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_argsort.py#L36)   |  |
| oneflow.split | [oneflow.Tensor.split](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L709)   |  |  |
| oneflow.sqrt | [oneflow.Tensor.sqrt](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L363)   | [sqrt_sum_with_cpu_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_sqrt_square_sum.py#L48)   |  |
| oneflow.square | [oneflow.Tensor.square](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L370)   | [square_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L146)   |  |
| oneflow.squeeze | [oneflow.squeeze](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L303)   | [squeeze_1d_input](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_squeeze.py#L51)   |  |
| oneflow.stack | [oneflow.stack](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L272)   | [stack_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_stack.py#L28)   |  |
| oneflow.stateful_op |  |  |  |
| oneflow.std | [oneflow.Tensor.std](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L377)   | [std_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_std.py#L26)   |  |
| oneflow.sub | [oneflow.Tensor.sub_](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L900)   | [sub](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_sub.py#L96)   |  |
| oneflow.sum | [oneflow.einsum](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/einsum.py#L20)   | [einsum_bilinear_transformation](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_einsum_bilinear_transformation.py#L42)   |  |
| oneflow.support |  |  |  |
| oneflow.swapaxes | [oneflow.swapaxes](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/swapaxes.py#L20)   | [swapaxes_flow_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_swapaxes.py#L32)   |  |
| oneflow.t | [oneflow.nn.functional.layer_norm](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/normalization.py#L20)   | [cast](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_flatten.py#L63)   |  |
| oneflow.tan | [oneflow.tanh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L150)   | [ConstantPad2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_zeropad2d.py#L96)   |  |
| oneflow.tanh | [oneflow.tanh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L150)   | [tanh_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L132)   |  |
| oneflow.tensor_buffer |  |  |  |
| oneflow.tensor_buffer_to_list_of_tensors |  |  |  |
| oneflow.tensor_buffer_to_tensor |  |  |  |
| oneflow.tensor_scatter_nd_update |  |  |  |
| oneflow.tensor_split |  |  |  |
| oneflow.tensor_to_tensor_buffer |  |  |  |
| oneflow.tile | [oneflow.tile](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tile.py#L20)   |  |  |
| oneflow.to_global |  |  |  |
| oneflow.to_local |  |  |  |
| oneflow.topk | [oneflow.topk](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/topk.py#L20)   |  |  |
| oneflow.transpose | [oneflow.transpose](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L245)   | [transpose](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_transpose.py#L86)   |  |
| oneflow.tril | [oneflow.tril](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L84)   | [tril_without_diag](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_tril.py#L26)   |  |
| oneflow.triu | [oneflow.triu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L114)   | [triu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_triu.py#L47)   |  |
| oneflow.truncated_normal_initializer |  |  |  |
| oneflow.uint8 |  |  |  |
| oneflow.unsqueeze | [oneflow.Tensor.unsqueeze](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L457)   | [unsqueeze_with_0_size_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_unsqueeze.py#L88)   |  |
| oneflow.var | [oneflow.Tensor.var](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L384)   |  |  |
| oneflow.variance_scaling_initializer |  |  |  |
| oneflow.version |  |  |  |
| oneflow.view | [oneflow.Tensor.view](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1529)   | [view](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_view.py#L78)   |  |
| oneflow.vsplit | [oneflow.vsplit](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/math_ops.py#L1502)   |  |  |
| oneflow.where | [oneflow.Tensor.argwhere](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L534)   | [argwhere](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_1.py#L625)   |  |
| oneflow.xavier_normal_initializer |  |  |  |
| oneflow.xavier_uniform_initializer |  |  |  |
| oneflow.zero_ |  |  |  |
| oneflow.zeros | [oneflow.zeros_like](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/constant.py#L43)   | [zeros_](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_part_1.py#L908)   |  |
| oneflow.zeros_initializer |  |  |  |
| oneflow.zeros_like |  |  |  |
| oneflow.optim.Adagrad |  | [adagrad](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_adagrad.py#L197)   |  |
| oneflow.optim.Adam |  | [adam](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_adam.py#L241)   |  |
| oneflow.optim.AdamW |  | [adamw](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_adamw.py#L244)   |  |
| oneflow.optim.LAMB |  | [lambda_lr](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_lr_scheduler.py#L199)   |  |
| oneflow.optim.RMSprop |  | [rmsprop](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_rmsprop.py#L228)   |  |
| oneflow.optim.SGD |  | [sgd](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_sgd.py#L194)   |  |
| oneflow.optim.lr_scheduler.ChainedScheduler |  |  |  |
| oneflow.optim.lr_scheduler.ConstantLR |  |  |  |
| oneflow.optim.lr_scheduler.CosineAnnealingLR |  |  |  |
| oneflow.optim.lr_scheduler.CosineAnnealingWarmRestarts |  |  |  |
| oneflow.optim.lr_scheduler.CosineDecayLR |  |  |  |
| oneflow.optim.lr_scheduler.ExponentialLR |  |  |  |
| oneflow.optim.lr_scheduler.LambdaLR |  |  |  |
| oneflow.optim.lr_scheduler.LinearLR |  |  |  |
| oneflow.optim.lr_scheduler.MultiStepLR |  |  |  |
| oneflow.optim.lr_scheduler.PolynomialLR |  |  |  |
| oneflow.optim.lr_scheduler.ReduceLROnPlateau |  |  |  |
| oneflow.optim.lr_scheduler.SequentialLR |  |  |  |
| oneflow.optim.lr_scheduler.StepLR |  |  |  |
| oneflow.optim.lr_scheduler.WarmUpLR |  |  |  |
| oneflow.nn.AdaptiveAvgPool1d |  |  |  |
| oneflow.nn.AdaptiveAvgPool2d |  |  |  |
| oneflow.nn.AdaptiveAvgPool3d |  |  |  |
| oneflow.nn.AllReduce |  |  |  |
| oneflow.nn.AvgPool1d |  | [avgpool1d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_avgpool.py#L28)   |  |
| oneflow.nn.AvgPool2d |  | [avgpool2d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_avgpool.py#L44)   |  |
| oneflow.nn.AvgPool3d |  | [avgpool3d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_avgpool.py#L61)   |  |
| oneflow.nn.BCELoss |  |  |  |
| oneflow.nn.BCEWithLogitsLoss |  |  |  |
| oneflow.nn.BatchNorm1d |  | [batchnorm1d_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_batchnorm.py#L32)   |  |
| oneflow.nn.BatchNorm2d |  | [batchnorm2d_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_batchnorm.py#L48)   |  |
| oneflow.nn.BatchNorm3d |  | [batchnorm3d_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_batchnorm.py#L64)   |  |
| oneflow.nn.CELU |  | [celu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L142)   |  |
| oneflow.nn.COCOReader |  |  |  |
| oneflow.nn.CTCLoss |  |  |  |
| oneflow.nn.CoinFlip |  |  |  |
| oneflow.nn.CombinedMarginLoss |  |  |  |
| oneflow.nn.ConstantPad1d |  | [constantpad1d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_constantpad.py#L32)   |  |
| oneflow.nn.ConstantPad2d |  | [ConstantPad2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_zeropad2d.py#L96)   |  |
| oneflow.nn.ConstantPad3d |  | [constantpad3d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_constantpad.py#L64)   |  |
| oneflow.nn.Conv1d |  | [conv1d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_conv1d.py#L422)   |  |
| oneflow.nn.Conv2d |  | [deconv2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_deconv2d.py#L68)   |  |
| oneflow.nn.Conv3d |  | [conv3d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_conv3d.py#L26)   |  |
| oneflow.nn.ConvTranspose1d |  |  |  |
| oneflow.nn.ConvTranspose2d |  |  |  |
| oneflow.nn.ConvTranspose3d |  |  |  |
| oneflow.nn.CropMirrorNormalize |  |  |  |
| oneflow.nn.CrossEntropyLoss |  |  |  |
| oneflow.nn.DistributedPariticalFCSample |  |  |  |
| oneflow.nn.Dropout |  | [dropout_numpy_case](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_dropout.py#L239)   |  |
| oneflow.nn.ELU | [oneflow.relu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L50)   | [prelu_4dim_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_prelu.py#L32)   |  |
| oneflow.nn.Embedding |  | [embedding](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_sparse.py#L152)   |  |
| oneflow.nn.FakeQuantization |  |  |  |
| oneflow.nn.Flatten | [oneflow.Tensor.flatten](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L154)   | [flatten](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_flatten.py#L38)   |  |
| oneflow.nn.Fold | [oneflow.Tensor.unfold](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L398)   | [fold](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_fold.py#L45)   |  |
| oneflow.nn.FusedBatchNorm1d |  |  |  |
| oneflow.nn.FusedBatchNorm2d |  |  |  |
| oneflow.nn.FusedBatchNorm3d |  |  |  |
| oneflow.nn.FusedMLP |  |  |  |
| oneflow.nn.GELU | [oneflow.gelu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L74)   | [gelu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L147)   |  |
| oneflow.nn.GLU |  | [glu_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_glu.py#L37)   |  |
| oneflow.nn.GPTIndexedBinDataReader |  |  |  |
| oneflow.nn.GRU |  |  |  |
| oneflow.nn.GroupNorm |  | [groupnorm](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_groupnorm.py#L332)   |  |
| oneflow.nn.Hardsigmoid |  | [hardsigmoid_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L157)   |  |
| oneflow.nn.Hardswish |  | [hardswish_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L167)   |  |
| oneflow.nn.Hardtanh |  | [hardtanh_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L172)   |  |
| oneflow.nn.Identity |  | [identity_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_linear.py#L217)   |  |
| oneflow.nn.InstanceNorm1d |  |  |  |
| oneflow.nn.InstanceNorm2d |  |  |  |
| oneflow.nn.InstanceNorm3d |  |  |  |
| oneflow.nn.KLDivLoss |  |  |  |
| oneflow.nn.L1Loss |  |  |  |
| oneflow.nn.LSTM |  |  |  |
| oneflow.nn.LayerNorm |  |  |  |
| oneflow.nn.LeakyReLU |  | [leakyrelu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L177)   |  |
| oneflow.nn.Linear |  | [linear_warmup_exp_lr](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_lr_scheduler.py#L376)   |  |
| oneflow.nn.LogSigmoid |  | [logsigmoid_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L162)   |  |
| oneflow.nn.LogSoftmax |  | [logsoftmax_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L414)   |  |
| oneflow.nn.MSELoss |  |  |  |
| oneflow.nn.MarginRankingLoss |  |  |  |
| oneflow.nn.MaxPool1d |  | [maxpool1d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_maxpool.py#L155)   |  |
| oneflow.nn.MaxPool2d |  | [maxpool2d_channel_last](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_maxpool.py#L135)   |  |
| oneflow.nn.MaxPool3d |  | [maxpool3d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_maxpool.py#L199)   |  |
| oneflow.nn.MinMaxObserver |  |  |  |
| oneflow.nn.Mish | [oneflow.mish](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L254)   | [mish_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L182)   |  |
| oneflow.nn.Module | [oneflow.nn.Module.to_consistent](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/module.py#L20)   | [module_to_global](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_module_to_consistent.py#L30)   |  |
| oneflow.nn.ModuleDict |  | [moduledict](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_module.py#L303)   |  |
| oneflow.nn.ModuleList |  |  |  |
| oneflow.nn.MovingAverageMinMaxObserver |  |  |  |
| oneflow.nn.NLLLoss |  |  |  |
| oneflow.nn.PReLU |  | [prelu_4dim_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_prelu.py#L32)   |  |
| oneflow.nn.Parameter |  | [parameter](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_module.py#L98)   |  |
| oneflow.nn.ParameterDict |  |  |  |
| oneflow.nn.ParameterList |  |  |  |
| oneflow.nn.PixelShuffle |  |  |  |
| oneflow.nn.Quantization |  |  |  |
| oneflow.nn.RNN |  |  |  |
| oneflow.nn.ReLU | [oneflow.relu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L50)   | [prelu_4dim_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_prelu.py#L32)   |  |
| oneflow.nn.ReLU6 |  | [relu6_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L127)   |  |
| oneflow.nn.ReflectionPad2d |  |  |  |
| oneflow.nn.ReplicationPad2d |  | [ReplicationPad2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_replicationpad2d.py#L104)   |  |
| oneflow.nn.SELU | [oneflow.selu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L396)   | [selu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L192)   |  |
| oneflow.nn.Sequential |  |  |  |
| oneflow.nn.SiLU | [oneflow.silu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L224)   | [silu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L187)   |  |
| oneflow.nn.Sigmoid | [oneflow.sigmoid](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L325)   | [sigmoid_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L152)   |  |
| oneflow.nn.SmoothL1Loss |  |  |  |
| oneflow.nn.Softmax | [oneflow.Tensor.softmax](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1141)   | [softmax_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L395)   |  |
| oneflow.nn.Softplus | [oneflow.softplus](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L133)   | [softplus](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L502)   |  |
| oneflow.nn.Softshrink |  | [softshrink_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L207)   |  |
| oneflow.nn.Softsign | [oneflow.Tensor.softsign](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1155)   | [softsign_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L685)   |  |
| oneflow.nn.Tanh | [oneflow.tanh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L150)   | [tanh_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L132)   |  |
| oneflow.nn.TripletMarginLoss |  |  |  |
| oneflow.nn.Unfold | [oneflow.Tensor.unfold](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L398)   | [unfold_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_unfold.py#L42)   |  |
| oneflow.nn.UpsamplingBilinear2d |  |  |  |
| oneflow.nn.UpsamplingNearest2d |  |  |  |
| oneflow.nn.ZeroPad2d |  |  |  |
| oneflow.nn.functional.adaptive_avg_pool1d |  |  |  |
| oneflow.nn.functional.adaptive_avg_pool2d |  |  |  |
| oneflow.nn.functional.adaptive_avg_pool3d |  |  |  |
| oneflow.nn.functional.affine_grid |  |  |  |
| oneflow.nn.functional.avg_pool1d |  |  |  |
| oneflow.nn.functional.avg_pool2d |  |  |  |
| oneflow.nn.functional.avg_pool3d |  |  |  |
| oneflow.nn.functional.celu |  | [celu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L142)   |  |
| oneflow.nn.functional.conv1d |  | [conv1d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_conv1d.py#L422)   |  |
| oneflow.nn.functional.conv2d |  | [deconv2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_deconv2d.py#L68)   |  |
| oneflow.nn.functional.conv3d |  | [conv3d_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_conv3d.py#L26)   |  |
| oneflow.nn.functional.cross_entropy |  |  |  |
| oneflow.nn.functional.ctc_greedy_decoder |  |  |  |
| oneflow.nn.functional.dropout |  | [dropout_numpy_case](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_dropout.py#L239)   |  |
| oneflow.nn.functional.elu | [oneflow.relu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L50)   | [prelu_4dim_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_prelu.py#L32)   |  |
| oneflow.nn.functional.embedding |  | [embedding](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_sparse.py#L152)   |  |
| oneflow.nn.functional.functional_maxpool |  |  |  |
| oneflow.nn.functional.gelu | [oneflow.gelu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L74)   | [gelu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L147)   |  |
| oneflow.nn.functional.glu |  | [glu_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_glu.py#L37)   |  |
| oneflow.nn.functional.grid_sample |  |  |  |
| oneflow.nn.functional.hardsigmoid |  | [hardsigmoid_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L157)   |  |
| oneflow.nn.functional.hardswish |  | [hardswish_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L167)   |  |
| oneflow.nn.functional.hardtanh |  | [hardtanh_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L172)   |  |
| oneflow.nn.functional.interpolate |  | [interpolate](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_interpolate.py#L658)   |  |
| oneflow.nn.functional.layer_norm |  |  |  |
| oneflow.nn.functional.leaky_relu |  |  |  |
| oneflow.nn.functional.linear |  | [linear_warmup_exp_lr](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_lr_scheduler.py#L376)   |  |
| oneflow.nn.functional.log_softmax |  |  |  |
| oneflow.nn.functional.logsigmoid |  | [logsigmoid_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L162)   |  |
| oneflow.nn.functional.max_pool1d |  |  |  |
| oneflow.nn.functional.max_pool2d |  |  |  |
| oneflow.nn.functional.max_pool3d |  |  |  |
| oneflow.nn.functional.mish | [oneflow.mish](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L254)   | [mish_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L182)   |  |
| oneflow.nn.functional.normalize |  | [normalize_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_normalize.py#L36)   |  |
| oneflow.nn.functional.one_hot |  |  |  |
| oneflow.nn.functional.pad |  | [ConstantPad2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_zeropad2d.py#L96)   |  |
| oneflow.nn.functional.prelu |  | [prelu_4dim_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_prelu.py#L32)   |  |
| oneflow.nn.functional.relu | [oneflow.relu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L50)   | [prelu_4dim_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_prelu.py#L32)   |  |
| oneflow.nn.functional.relu6 |  | [relu6_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L127)   |  |
| oneflow.nn.functional.selu | [oneflow.selu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L396)   | [selu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L192)   |  |
| oneflow.nn.functional.sigmoid | [oneflow.sigmoid](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L325)   | [sigmoid_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L152)   |  |
| oneflow.nn.functional.silu | [oneflow.silu](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L224)   | [silu_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L187)   |  |
| oneflow.nn.functional.smooth_l1_loss |  |  |  |
| oneflow.nn.functional.softmax | [oneflow.Tensor.softmax](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1141)   | [softmax_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L395)   |  |
| oneflow.nn.functional.softplus | [oneflow.softplus](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L133)   | [softplus](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L502)   |  |
| oneflow.nn.functional.softshrink |  | [softshrink_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L207)   |  |
| oneflow.nn.functional.softsign | [oneflow.Tensor.softsign](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1155)   | [softsign_module_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_activation.py#L685)   |  |
| oneflow.nn.functional.sparse_softmax_cross_entropy |  |  |  |
| oneflow.nn.functional.tanh | [oneflow.tanh](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/activation.py#L150)   | [tanh_module](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_consistent_activation.py#L132)   |  |
| oneflow.nn.functional.triplet_margin_loss |  |  |  |
| oneflow.nn.functional.upsample |  | [upsample2d](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_upsample.py#L380)   |  |
| oneflow.nn.init.CalcGain |  |  |  |
| oneflow.nn.init.calculate_gain |  |  |  |
| oneflow.nn.init.constant_ |  |  |  |
| oneflow.nn.init.flow | [oneflow.decode_onerec](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/dataset.py#L20)   | [flow_erf_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_erf.py#L33)   |  |
| oneflow.nn.init.kaiming_normal_ |  |  |  |
| oneflow.nn.init.kaiming_uniform_ |  |  |  |
| oneflow.nn.init.normal_ |  |  |  |
| oneflow.nn.init.ones_ |  |  |  |
| oneflow.nn.init.os | [oneflow.transpose](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/framework/docstr/array_ops.py#L245)   | [cos](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_math_ops.py#L88)   |  |
| oneflow.nn.init.trunc_normal_ |  |  |  |
| oneflow.nn.init.uniform_ |  |  |  |
| oneflow.nn.init.xavier_normal_ |  |  |  |
| oneflow.nn.init.xavier_uniform_ |  |  |  |
| oneflow.nn.init.zeros_ |  |  |  |
| oneflow.nn.init.adagrad |  | [adagrad](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_adagrad.py#L197)   |  |
| oneflow.nn.init.adam |  | [adam](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_adam.py#L241)   |  |
| oneflow.nn.init.adamw |  | [adamw](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_adamw.py#L244)   |  |
| oneflow.nn.init.chained_scheduler |  |  |  |
| oneflow.nn.init.constant_lr |  |  |  |
| oneflow.nn.init.cosine_annealing_lr |  |  |  |
| oneflow.nn.init.cosine_annealing_warm_restarts |  |  |  |
| oneflow.nn.init.cosine_decay_lr |  |  |  |
| oneflow.nn.init.exponential_lr |  |  |  |
| oneflow.nn.init.lamb |  | [lambda_lr](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_lr_scheduler.py#L199)   |  |
| oneflow.nn.init.lambda_lr |  |  |  |
| oneflow.nn.init.linear_lr |  |  |  |
| oneflow.nn.init.lr_scheduler |  |  |  |
| oneflow.nn.init.multistep_lr |  |  |  |
| oneflow.nn.init.polynomial_lr |  |  |  |
| oneflow.nn.init.reduce_lr_on_plateau |  |  |  |
| oneflow.nn.init.rmsprop |  | [rmsprop](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_rmsprop.py#L228)   |  |
| oneflow.nn.init.sequential_lr |  |  |  |
| oneflow.nn.init.sgd |  | [sgd](https://github.com/Oneflow-Inc/oneflow/blob/8e2da64b33b59cc907195de423dc7fa632c1fee6/python/oneflow/test/../../../python/oneflow/test/modules/test_optim_sgd.py#L194)   |  |
| oneflow.nn.init.step_lr |  |  |  |
| oneflow.nn.init.warmup_lr |  |  |  |
## Test Data Summary
- OneFlow Total API Number: ====================>448
- Doc Test Ratio: ====================>35.71% = 160 / 448
- Compatiable/Completeness Test Ratio: ====================>48.21% = 216 / 448
- Exception Test Ratio: ====================>0.22% = 1 / 448
