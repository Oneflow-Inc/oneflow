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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class CollectiveBoxingPackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingPackKernel);
  CollectiveBoxingPackKernel() = default;
  ~CollectiveBoxingPackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void CollectiveBoxingPackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const CollectiveBoxingPackOpConf& pack_conf = this->op_conf().collective_boxing_pack_conf();
  const int64_t num_ranks = pack_conf.num_ranks();
  const Shape logical_shape(pack_conf.logical_shape());
  const bool need_transpose = !((pack_conf.dst_sbp_parallel().has_split_parallel()
                                 && pack_conf.dst_sbp_parallel().split_parallel().axis() == 0)
                                || pack_conf.dst_sbp_parallel().has_broadcast_parallel()
                                || pack_conf.dst_sbp_parallel().has_partial_sum_parallel());
  if (need_transpose) {
    const int64_t dst_split_axis = pack_conf.dst_sbp_parallel().split_parallel().axis();
    DimVector transpose_in_dim_vec = logical_shape.dim_vec();
    if (pack_conf.src_sbp_parallel().has_split_parallel()) {
      const int64_t src_split_axis = pack_conf.src_sbp_parallel().split_parallel().axis();
      transpose_in_dim_vec[src_split_axis] = transpose_in_dim_vec.at(src_split_axis) / num_ranks;
    }
    CHECK_EQ(transpose_in_dim_vec.at(dst_split_axis) % num_ranks, 0);
    transpose_in_dim_vec[dst_split_axis] = transpose_in_dim_vec.at(dst_split_axis) / num_ranks;
    transpose_in_dim_vec.insert(transpose_in_dim_vec.begin() + dst_split_axis, num_ranks);
    const Shape transpose_in_shape(transpose_in_dim_vec);
    DimVector transpose_out_dim_vec;
    std::vector<int32_t> perm;
    perm.push_back(dst_split_axis);
    transpose_out_dim_vec.push_back(transpose_in_shape.At(dst_split_axis));
    FOR_RANGE(int64_t, i, 0, transpose_in_shape.NumAxes()) {
      if (i != dst_split_axis) {
        perm.push_back(i);
        transpose_out_dim_vec.push_back(transpose_in_shape.At(i));
      }
    }
    const Shape transpose_out_shape(transpose_out_dim_vec);
    NewKernelUtil<device_type>::Transpose(
        ctx.device_ctx, transpose_in_shape.NumAxes(), transpose_in_shape, transpose_out_shape, perm,
        transpose_in_shape.elem_cnt(), in->dptr<T>(), out->mut_dptr<T>());
  } else {
    out->CopyDataContentFrom(ctx.device_ctx, in);
  }
}

#define REGISTER_COLLECTIVE_BOXING_PACK_KERNEL(device_type_v, dtype_pair)                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(                                                    \
      OperatorConf::kCollectiveBoxingPackConf, device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
      CollectiveBoxingPackKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_COLLECTIVE_BOXING_PACK_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

#if defined(WITH_CUDA)
REGISTER_COLLECTIVE_BOXING_PACK_KERNEL(DeviceType::kGPU, (float16, DataType::kFloat16))
#endif

}  // namespace oneflow
