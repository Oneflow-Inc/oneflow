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
class BoxingS2SAll2AllPackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingS2SAll2AllPackKernel);
  BoxingS2SAll2AllPackKernel() = default;
  ~BoxingS2SAll2AllPackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void BoxingS2SAll2AllPackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const BoxingS2SAll2AllPackOpConf& pack_conf = this->op_conf().boxing_s2s_all2all_pack_conf();
  const int64_t dst_split_axis = pack_conf.dst_split_axis();
  const int64_t num_ranks = pack_conf.num_ranks();
  const bool need_transpose = (dst_split_axis != 0);
  if (need_transpose) {
    DimVector transpose_in_dim_vec;
    const ShapeView& in_shape = in->shape();
    FOR_RANGE(int64_t, i, 0, in_shape.NumAxes()) {
      if (i == dst_split_axis) {
        transpose_in_dim_vec.push_back(num_ranks);
        CHECK_EQ(in_shape.At(i) % num_ranks, 0);
        transpose_in_dim_vec.push_back(in_shape.At(i) / num_ranks);
      } else {
        transpose_in_dim_vec.push_back(in_shape.At(i));
      }
    }
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
    CHECK_EQ(dst_split_axis, 0);
    out->CopyDataContentFrom(ctx.device_ctx, in);
  }
}

#define REGISTER_BOXING_S2S_ALL2ALL_PACK_KERNEL(device_type_v, dtype_pair)                  \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(                                                    \
      OperatorConf::kBoxingS2SAll2AllPackConf, device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
      BoxingS2SAll2AllPackKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_BOXING_S2S_ALL2ALL_PACK_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

#if defined(WITH_CUDA)
REGISTER_BOXING_S2S_ALL2ALL_PACK_KERNEL(DeviceType::kGPU, (float16, DataType::kFloat16))
#endif

}  // namespace oneflow
