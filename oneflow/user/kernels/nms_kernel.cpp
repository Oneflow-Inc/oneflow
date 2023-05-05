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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace {

template<typename T>
__inline__ T IoU(T const* const a, T const* const b) {
  T interS = std::max(std::min(a[2], b[2]) - std::max(a[0], b[0]), static_cast<T>(0.f))
             * std::max(std::min(a[3], b[3]) - std::max(a[1], b[1]), static_cast<T>(0.f));
  T Sa = (a[2] - a[0]) * (a[3] - a[1]);
  T Sb = (b[2] - b[0]) * (b[3] - b[1]);
  return interS / (Sa + Sb - interS);
}

}  // namespace

template<typename T>
class NmsCpuKernel final : public user_op::OpKernel {
 public:
  NmsCpuKernel() = default;
  ~NmsCpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* boxes_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* keep_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* boxes = boxes_blob->dptr<T>();
    int8_t* keep = keep_blob->mut_dptr<int8_t>();

    const int num_boxes = boxes_blob->shape_view().At(0);
    int num_keep = ctx->Attr<int>("keep_n");
    if (num_keep <= 0 || num_keep > num_boxes) { num_keep = num_boxes; }
    const float iou_threshold = ctx->Attr<float>("iou_threshold");
    for (int i = 0; i < num_boxes; i++) { keep[i] = -1; }
    for (int i = 0; i < num_boxes; i++) {
      if (keep[i] == 0) continue;
      keep[i] = 1;
      for (int j = i + 1; j < num_boxes; j++) {
        if (IoU(boxes + i * 4, boxes + j * 4) > iou_threshold) { keep[j] = 0; }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_NMS_CPU_KERNEL(dtype)                                            \
  REGISTER_USER_KERNEL("nms").SetCreateFn<NmsCpuKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                              \
      && (user_op::HobDataType("out", 0) == DataType::kInt8)                      \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_NMS_CPU_KERNEL(float)
REGISTER_NMS_CPU_KERNEL(double)

}  // namespace oneflow
