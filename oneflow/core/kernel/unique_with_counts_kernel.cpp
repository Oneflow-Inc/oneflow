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
#include "oneflow/core/kernel/unique_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class UniqueWithCountsKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniqueWithCountsKernel);
  UniqueWithCountsKernel() = default;
  ~UniqueWithCountsKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T, typename K>
void UniqueWithCountsKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x = BnInOp2Blob("x");
  Blob* y = BnInOp2Blob("y");
  Blob* idx = BnInOp2Blob("idx");
  Blob* count = BnInOp2Blob("count");
  Blob* workspace = BnInOp2Blob("workspace");
  Blob* num_unique = BnInOp2Blob("num_unique");
  void* workspace_ptr = nullptr;
  int64_t workspace_size_in_bytes = 0;
  if (workspace != nullptr) {
    workspace_ptr = workspace->mut_dptr();
    workspace_size_in_bytes = workspace->ByteSizeOfBlobBody();
  }
  UniqueKernelUtil<device_type, T, K>::UniqueWithCounts(
      ctx.device_ctx, x->shape().elem_cnt(), x->dptr<T>(), num_unique->mut_dptr<K>(),
      y->mut_dptr<T>(), idx->mut_dptr<K>(), count->mut_dptr<K>(), workspace_ptr,
      workspace_size_in_bytes);
}

#define MAKE_UNIQUE_WITH_COUNTS_KERNEL_ENTRY(device_type_v, data_type_pair, indices_type_pair) \
  NEW_REGISTER_KERNEL(OperatorConf::kUniqueWithCountsConf,                                     \
                      UniqueWithCountsKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),  \
                                             OF_PP_PAIR_FIRST(indices_type_pair)>)             \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                            \
        return ((kernel_conf.op_attribute().op_conf().device_tag() == ToString(device_type_v)) \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())            \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                       \
                    == kernel_conf.unique_with_counts_conf().indices_data_type()));            \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_UNIQUE_WITH_COUNTS_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_UNIQUE_WITH_COUNTS_KERNEL_ENTRY

}  // namespace oneflow
