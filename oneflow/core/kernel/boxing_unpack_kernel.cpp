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
class BoxingUnpackKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingUnpackKernel);
  BoxingUnpackKernel() = default;
  ~BoxingUnpackKernel() override = default;

 private:
  bool IsStateless() const override { return false; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type>
void BoxingUnpackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob("out");
    const BoxingUnpackOpConf& boxing_unpack_conf = this->op_conf().boxing_unpack_conf();
    if(boxing_unpack_conf.need_transpose()) {
        const int64_t parallel_num = boxing_unpack_conf.parallel_num();
        const int64_t src_split_axis = boxing_unpack_conf.src_split_axis();
        const int64_t dst_split_axis = boxing_unpack_conf.dst_split_axis();
        const Shape src_shape(boxing_unpack_conf.src_shape());
        const Shape dst_shape(boxing_unpack_conf.dst_shape());
        DimVector dim_vec;
        dim_vec.push_back(parallel_num); //boxing is split 0
        dim_vec.push_back(src_shape.At(0) / parallel_num);
        FOR_RANGE(int64_t, i, 1, src_shape.NumAxes()) {
            dim_vec.push_back(src_shape.At(i));
        }
        Shape transpose_in_shape = Shape(dim_vec);
        std::vector<int32_t> perm;
        DimVector out_dim_vec;
        FOR_RANGE(int64_t, i, 1, transpose_in_shape.NumAxes()) {
            perm.push_back(i);
            out_dim_vec.push_back(transpose_in_shape.At(i));
        }
        //transpose axis 0 to src_split_axis
        perm.insert(perm.begin() + src_split_axis, 0);
        out_dim_vec.insert(out_dim_vec.begin() + src_split_axis, transpose_in_shape.At(0));

        Shape transpose_out_shape = Shape(out_dim_vec);
        NewKernelUtil<device_type>::Transpose(ctx.device_ctx, transpose_in_shape.NumAxes(), transpose_in_shape,
                                              transpose_out_shape, perm, transpose_in_shape.elem_cnt(),
                                              in->dptr<float>(), out->mut_dptr<float>());
    } else {
      out->CopyDataContentFrom(ctx.device_ctx, in);
    }
}

#ifdef WITH_CUDA
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kBoxingUnpackConf, DeviceType::kGPU,
                            BoxingUnpackKernel<DeviceType::kGPU>);
#endif

}  // namespace oneflow
