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
#include "oneflow/core/kernel/indexed_slices_naive_model_update_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename K>
class IndexedSlicesNaiveMdUpdateKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IndexedSlicesNaiveMdUpdateKernel);
  IndexedSlicesNaiveMdUpdateKernel() = default;
  ~IndexedSlicesNaiveMdUpdateKernel() override = default;

 private:
  const PbMessage& GetCustomizedOpConf() const override;
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<DeviceType device_type, typename T, typename K>
const PbMessage& IndexedSlicesNaiveMdUpdateKernel<device_type, T, K>::GetCustomizedOpConf() const {
  return this->op_conf().indexed_slices_naive_model_update_conf();
}

template<DeviceType device_type, typename T, typename K>
void IndexedSlicesNaiveMdUpdateKernel<device_type, T, K>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("model_diff_indices");
  const Blob* values = BnInOp2Blob("model_diff_values");
  const Blob* learning_rate = BnInOp2Blob("learning_rate");
  Blob* model = BnInOp2Blob("model");
  const int64_t offset = this->kernel_conf().indexed_slices_naive_model_update_conf().lower_bound();
  CHECK_EQ(this->kernel_conf().indexed_slices_naive_model_update_conf().upper_bound() - offset,
           model->shape().At(0));
  IndexedSlicesNaiveMdUpdateKernelUtil<device_type, T, K>::Update(
      ctx.device_ctx, indices->dptr<K>(), values->dptr<T>(), learning_rate->dptr<float>(),
      indices->shape().elem_cnt(), model->shape().At(0), model->shape().Count(1), offset,
      model->mut_dptr<T>());
}

#define MAKE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_ENTRY(device_type_v, data_type_pair,         \
                                                            indices_type_pair)                     \
  NEW_REGISTER_KERNEL(                                                                             \
      OperatorConf::kIndexedSlicesNaiveModelUpdateConf,                                            \
      IndexedSlicesNaiveMdUpdateKernel<device_type_v, OF_PP_PAIR_FIRST(data_type_pair),            \
                                       OF_PP_PAIR_FIRST(indices_type_pair)>)                       \
      .SetIsMatchedPred([](const KernelConf& kernel_conf) -> bool {                                \
        return ((kernel_conf.op_attribute().op_conf().device_tag() == ToString(device_type_v))     \
                && ((OF_PP_PAIR_SECOND(data_type_pair)) == kernel_conf.data_type())                \
                && (OF_PP_PAIR_SECOND(indices_type_pair)                                           \
                    == kernel_conf.indexed_slices_naive_model_update_conf().indices_data_type())); \
      });

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_ENTRY,
                                 DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)
#undef MAKE_INDEXED_SLICES_NAIVE_MODEL_UPDATE_KERNEL_ENTRY

}  // namespace oneflow
