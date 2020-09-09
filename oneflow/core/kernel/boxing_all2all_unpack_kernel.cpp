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
class BoxingAll2AllUnpackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingAll2AllUnpackKernel);
  BoxingAll2AllUnpackKernel() = default;
  ~BoxingAll2AllUnpackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
void BoxingAll2AllUnpackKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const BoxingAll2AllUnpackOpConf& boxing_all2all_unpack_conf =
      this->op_conf().boxing_all2all_unpack_conf();
  const int64_t src_split_axis = boxing_all2all_unpack_conf.src_split_axis();
  const int64_t dst_split_axis = boxing_all2all_unpack_conf.dst_split_axis();
  const int64_t parallel_num = boxing_all2all_unpack_conf.parallel_num();
  const Shape logical_shape(boxing_all2all_unpack_conf.logical_shape());
  if (boxing_all2all_unpack_conf.need_transpose()) {
    DimVector dim_vec = logical_shape.dim_vec();
    dim_vec[src_split_axis] = dim_vec.at(src_split_axis) / parallel_num;
    dim_vec[dst_split_axis] = dim_vec.at(dst_split_axis) / parallel_num;
    dim_vec.insert(dim_vec.begin(), parallel_num);
    const Shape transpose_in_shape = Shape(dim_vec);

    DimVector out_dim_vec;
    std::vector<int32_t> perm;
    FOR_RANGE(int64_t, i, 1, transpose_in_shape.NumAxes()) {
      perm.push_back(i);
      out_dim_vec.push_back(transpose_in_shape.At(i));
    }
    perm.insert(perm.begin() + src_split_axis, 0);
    out_dim_vec.insert(out_dim_vec.begin() + src_split_axis, transpose_in_shape.At(0));
    const Shape transpose_out_shape = Shape(out_dim_vec);
    NewKernelUtil<device_type>::Transpose(
        ctx.device_ctx, transpose_in_shape.NumAxes(), transpose_in_shape, transpose_out_shape, perm,
        transpose_in_shape.elem_cnt(), in->dptr<T>(), out->mut_dptr<T>());
  } else {
    CHECK_EQ(src_split_axis, 0);
    out->CopyDataContentFrom(ctx.device_ctx, in);
  }
}

#ifdef WITH_CUDA
#define REGISTER_BOXING_ALL2ALL_UNPACK_KERNEL(dtype)                                              \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kBoxingAll2AllUnpackConf, DeviceType::kGPU, \
                                        dtype, BoxingAll2AllUnpackKernel<DeviceType::kGPU, dtype>)
REGISTER_BOXING_ALL2ALL_UNPACK_KERNEL(float16);
REGISTER_BOXING_ALL2ALL_UNPACK_KERNEL(float);
REGISTER_BOXING_ALL2ALL_UNPACK_KERNEL(double);
REGISTER_BOXING_ALL2ALL_UNPACK_KERNEL(int8_t);
REGISTER_BOXING_ALL2ALL_UNPACK_KERNEL(int32_t);
REGISTER_BOXING_ALL2ALL_UNPACK_KERNEL(int64_t);
#endif

}  // namespace oneflow
