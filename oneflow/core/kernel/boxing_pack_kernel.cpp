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

template<DeviceType device_type>
class BoxingPackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingPackKernel);
  BoxingPackKernel() = default;
  ~BoxingPackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
void BoxingPackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const BoxingPackOpConf& boxing_pack_conf = this->op_conf().boxing_pack_conf();
  if (boxing_pack_conf.need_transpose()) {
    const ShapeView& in_shape = in->shape();
    const int64_t parallel_num = boxing_pack_conf.parallel_num();
    const int64_t src_split_axis = boxing_pack_conf.src_split_axis();
    const int64_t dst_split_axis = boxing_pack_conf.dst_split_axis();
    DimVector dim_vec;
    FOR_RANGE(int64_t, i, 0, in_shape.NumAxes()) {
      if (i == dst_split_axis) {
        dim_vec.push_back(parallel_num);
        dim_vec.push_back(in_shape.At(i) / parallel_num);
      } else {
        dim_vec.push_back(in_shape.At(i));
      }
    }
    Shape transpose_in_shape = Shape(dim_vec);

    // transpose axis dst_split_axis to 0
    std::vector<int32_t> perm;
    DimVector out_dim_vec;
    perm.push_back(dst_split_axis);
    out_dim_vec.push_back(transpose_in_shape.At(dst_split_axis));
    FOR_RANGE(int64_t, i, 0, transpose_in_shape.NumAxes()) {
      if (i != dst_split_axis) {
        perm.push_back(i);
        out_dim_vec.push_back(transpose_in_shape.At(i));
      }
    }
    Shape transpose_out_shape = Shape(out_dim_vec);
    NewKernelUtil<device_type>::Transpose(
        ctx.device_ctx, transpose_in_shape.NumAxes(), transpose_in_shape, transpose_out_shape, perm,
        transpose_in_shape.elem_cnt(), in->dptr<float>(), out->mut_dptr<float>());
  } else {
    out->CopyDataContentFrom(ctx.device_ctx, in);
  }
}

#ifdef WITH_CUDA
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kBoxingPackConf, DeviceType::kGPU,
                            BoxingPackKernel<DeviceType::kGPU>);
#endif

}  // namespace oneflow
