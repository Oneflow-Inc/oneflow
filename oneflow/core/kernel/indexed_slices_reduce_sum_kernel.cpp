/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/indexed_slices_reduce_sum_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class IndexedSlicesReduceSumKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IndexedSlicesReduceSumKernel);
  IndexedSlicesReduceSumKernel() = default;
  ~IndexedSlicesReduceSumKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T, typename K>
void IndexedSlicesReduceSumKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x_indices = BnInOp2Blob("x_indices");
  const Blob* x_values = BnInOp2Blob("x_values");
  Blob* y_indices = BnInOp2Blob("y_indices");
  Blob* y_values = BnInOp2Blob("y_values");
  Blob* num_unique = BnInOp2Blob("num_unique");
  Blob* workspace = BnInOp2Blob("workspace");
  void* workspace_ptr = nullptr;
  int64_t workspace_size_in_bytes = 0;
  if (workspace != nullptr) {
    workspace_ptr = workspace->mut_dptr();
    workspace_size_in_bytes = workspace->ByteSizeOfBlobBody();
  }
  const int64_t n = x_indices->shape().elem_cnt();
  const int64_t m = x_values->shape().elem_cnt() / n;
  IndexedSlicesReduceSumKernelUtil<device_type, K, T, int64_t>::ReduceSum(
      ctx.device_ctx, n, m, x_indices->dptr<K>(), x_values->dptr<T>(),
      num_unique->mut_dptr<int64_t>(), y_indices->mut_dptr<K>(), y_values->mut_dptr<T>(),
      workspace_ptr, workspace_size_in_bytes);
}

#define MAKE_INDEXED_SLICES_REDUCE_SUM_KERNEL_ENTRY(device_type_v, data_type_pair,             \
                                                    indices_type_pair)                         \
  NEW_REGISTER_KERNEL(                                                                         \
      OperatorConf::kIndexedSlicesReduceSumConf,                                               \
      IndexedSlicesReduceSumKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),            \
                                   OF_PP_PAIR_FIRST(indices_type_pair)>)                       \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                            \
        return ((kernel_conf.op_attribute().op_conf().device_tag() == ToString(device_type_v)) \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())            \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                       \
                    == kernel_conf.indexed_slices_reduce_sum_conf().indices_data_type()));     \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_INDEXED_SLICES_REDUCE_SUM_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_INDEXED_SLICES_REDUCE_SUM_KERNEL_ENTRY

}  // namespace oneflow
