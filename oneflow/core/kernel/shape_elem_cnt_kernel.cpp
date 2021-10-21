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
#include "oneflow/core/kernel/shape_elem_cnt_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ShapeElemCntKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const T elem_cnt = GetShapePartialElemCnt(BnInOp2Blob("x")->shape());
  KernelUtil<device_type, T>::Set(ctx.device_ctx, elem_cnt, BnInOp2Blob("y")->mut_dptr<T>());
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
