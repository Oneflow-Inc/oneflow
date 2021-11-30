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
#include "oneflow/core/ep/include/primitive/fill.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ShapeElemCntKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ShapeElemCntKernel);
  ShapeElemCntKernel() = default;
  ~ShapeElemCntKernel() override = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
  int32_t GetShapePartialElemCnt(const ShapeView& shape) const;
};

template<DeviceType device_type, typename T>
void ShapeElemCntKernel<device_type, T>::ForwardDataContent(KernelContext* ctx) const {
  const T elem_cnt = GetShapePartialElemCnt(ctx->BnInOp2Blob("x")->shape());
  std::unique_ptr<ep::primitive::Fill> fill =
      ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->stream()->device_type(),
                                                              ctx->BnInOp2Blob("y")->data_type());
  CHECK(fill);
  fill->Launch(ctx->stream(), ctx->BnInOp2Blob("y")->mut_dptr(), elem_cnt, 1);
}

template<DeviceType device_type, typename T>
int32_t ShapeElemCntKernel<device_type, T>::GetShapePartialElemCnt(const ShapeView& shape) const {
  int32_t ret = 1;
  for (int32_t axis : this->kernel_conf().shape_elem_cnt_conf().axis()) { ret *= shape.At(axis); }
  return ret;
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kShapeElemCntConf, ShapeElemCntKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
