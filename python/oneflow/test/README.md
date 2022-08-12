## Ops Version : Alpha


| Op Name | Doc Test | Compatiable/Completeness Test | Exception |
| ------------------------- | ------------- | ----------------------------- | --------- |
| oneflow.nn.init.calculate_gain |  |  |  |
| oneflow.nn.init.uniform_ | [oneflow.Tensor.uniform_](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1455)   |  |  |
| oneflow.nn.init.normal_ | [oneflow.Tensor.normal_](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L1154)   | [eager_boxing_normal_1d_exhaustive_testing](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_eager_boxing_exhaustive.py#L113)   | [normal_data_type_error](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/exceptions/test_nn_functor.py#L278)   |
| oneflow.nn.init.constant_ |  | [constant_global](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_global_constant.py#L99)   |  |
| oneflow.nn.init.ones_ |  | [ones_like_float](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_ones_like.py#L27)   |  |
| oneflow.nn.init.zeros_ |  | [zeros_like_float](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_global_zeros_like.py#L27)   |  |
| oneflow.nn.init.xavier_uniform_ |  |  |  |
| oneflow.nn.init.xavier_normal_ |  |  |  |
| oneflow.nn.init.kaiming_uniform_ |  |  |  |
| oneflow.nn.init.kaiming_normal_ |  |  |  |
| oneflow.nn.init.trunc_normal_ |  |  |  |
| oneflow.nn.init.orthogonal_ |  |  |  |
| oneflow.utils.data |  | [flow_erfc_with_random_data](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_erfc.py#L33)   | [global_branch_error_global_data_mean](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/exceptions/test_global_branch_error_with_global_mean.py#L32)   |
| DataLoader |  | [dataloader_indexing_with_1_dim_tensor](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/tensor/test_tensor_indexing.py#L473)   |  |
| Dataset |  |  |  |
| IterableDataset |  |  |  |
| TensorDataset |  |  |  |
| ConcatDataset |  |  |  |
| Subset |  |  |  |
| oneflow.utils.data.random_split |  |  |  |
| oneflow.utils.data.Sampler |  |  |  |
| oneflow.utils.data.SequentialSampler |  |  |  |
| oneflow.utils.data.RandomSampler |  |  |  |
| oneflow.utils.data.SubsetRandomSampler |  |  |  |
| oneflow.utils.data.BatchSampler |  |  |  |
| oneflow.utils.data.distributed.DistributedSampler |  |  |  |
| oneflow.placement | [oneflow.Tensor.placement](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L95)   | [eager_boxing_with_same_placement_p_to_s1](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_eager_boxing.py#L3093)   | [meshgrid_tensors_placement_runtime_error](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/exceptions/test_array_functor.py#L302)   |
| oneflow.env.all_device_placement |  |  |  |
| oneflow.sbp.sbp | [oneflow.Tensor.sbp](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/framework/docstr/tensor.py#L102)   | [eager_global_cast_with_same_placement_and_sbp](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_eager_boxing.py#L3205)   | [get_sbp_with_invalid_axis](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/exceptions/test_local_global_convert_error.py#L24)   |
| Optimizer |  |  |  |
| oneflow.one_embedding.make_table_options |  |  |  |
| oneflow.one_embedding.make_table |  |  |  |
| oneflow.one_embedding.MultiTableEmbedding |  |  |  |
| oneflow.one_embedding.MultiTableMultiColumnEmbedding |  |  |  |
| oneflow.one_embedding.Ftrl |  | [ftrl](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_one_embedding_ftrl.py#L191)   |  |
| Function |  | [unsqueeze_tensor_function](https://github.com/Oneflow-Inc/oneflow/blob/5086f3257530ab37906c814cad5ea8f92481c505/python/oneflow/test/../../../python/oneflow/test/modules/test_unsqueeze.py#L37)   |  |
## Test Data Summary
- OneFlow Total API Number: 36
- Doc Test Ratio: 11.11% (4 / 36)
- Compatiable/Completeness Test Ratio: 27.78% (10 / 36)
- Exception Test Ratio: 11.11% (4 / 36)
