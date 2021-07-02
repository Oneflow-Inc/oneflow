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
class CollectiveBoxingUnpackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingUnpackKernel);
  CollectiveBoxingUnpackKernel() = default;
  ~CollectiveBoxingUnpackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void CollectiveBoxingUnpackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const CollectiveBoxingUnpackOpConf& unpack_conf = this->op_conf().collective_boxing_unpack_conf();
  const int64_t num_ranks = unpack_conf.num_ranks();
  const Shape logical_shape(unpack_conf.logical_shape());
  const bool need_transpose = !((unpack_conf.src_sbp_parallel().has_split_parallel()
                                 && unpack_conf.src_sbp_parallel().split_parallel().axis() == 0)
                                || unpack_conf.src_sbp_parallel().has_broadcast_parallel()
                                || unpack_conf.src_sbp_parallel().has_partial_sum_parallel());
  if (need_transpose) {
    const int64_t src_split_axis = unpack_conf.src_sbp_parallel().split_parallel().axis();
    DimVector transpose_in_dim_vec = logical_shape.dim_vec();
    CHECK_EQ(transpose_in_dim_vec.at(src_split_axis) % num_ranks, 0);
    transpose_in_dim_vec[src_split_axis] = transpose_in_dim_vec.at(src_split_axis) / num_ranks;
    if (unpack_conf.dst_sbp_parallel().has_split_parallel()) {
      const int64_t dst_split_axis = unpack_conf.dst_sbp_parallel().split_parallel().axis();
      CHECK_EQ(transpose_in_dim_vec.at(dst_split_axis) % num_ranks, 0);
      transpose_in_dim_vec[dst_split_axis] = transpose_in_dim_vec.at(dst_split_axis) / num_ranks;
    }
    transpose_in_dim_vec.insert(transpose_in_dim_vec.begin(), num_ranks);
    const Shape transpose_in_shape(transpose_in_dim_vec);
    DimVector transpose_out_dim_vec;
    std::vector<int32_t> perm;
    FOR_RANGE(int64_t, i, 1, transpose_in_shape.NumAxes()) {
      perm.push_back(i);
      transpose_out_dim_vec.push_back(transpose_in_shape.At(i));
    }
    perm.insert(perm.begin() + src_split_axis, 0);
    transpose_out_dim_vec.insert(transpose_out_dim_vec.begin() + src_split_axis,
                                 transpose_in_shape.At(0));
    const Shape transpose_out_shape(transpose_out_dim_vec);
    NewKernelUtil<device_type>::Transpose(
        ctx.device_ctx, transpose_in_shape.NumAxes(), transpose_in_shape, transpose_out_shape, perm,
        transpose_in_shape.elem_cnt(), in->dptr<T>(), out->mut_dptr<T>());
  } else {
    out->CopyDataContentFrom(ctx.device_ctx, in);
  }
}

#define REGISTER_COLLECTIVE_BOXING_UNPACK_KERNEL(device_type_v, dtype_pair)                   \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(                                                      \
      OperatorConf::kCollectiveBoxingUnpackConf, device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
      CollectiveBoxingUnpackKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_COLLECTIVE_BOXING_UNPACK_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

#if defined(WITH_CUDA)
REGISTER_COLLECTIVE_BOXING_UNPACK_KERNEL(DeviceType::kGPU, (float16, DataType::kFloat16))
#endif

}  // namespace oneflow
