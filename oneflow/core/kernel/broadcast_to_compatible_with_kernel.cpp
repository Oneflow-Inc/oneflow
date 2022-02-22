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
#include "oneflow/core/ndarray/ndarray_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BroadcastToCompatibleWithKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BroadcastToCompatibleWithKernel);
  BroadcastToCompatibleWithKernel() = default;
  ~BroadcastToCompatibleWithKernel() = default;

 private:
  void ForwardDataContent(KernelContext* ctx) const override;
};

template<DeviceType device_type, typename T>
void BroadcastToCompatibleWithKernel<device_type, T>::ForwardDataContent(KernelContext* ctx) const {
  const Blob* x = ctx->BnInOp2Blob("x");
  Blob* y = ctx->BnInOp2Blob("y");
  const auto& broadcast_axes =
      this->kernel_conf().broadcast_to_compatible_with_conf().broadcast_axes();
  int64_t num_axes = y->shape().NumAxes();
  Shape x_extend_shape = CreateLeftExtendedShape(x->shape(), num_axes);
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (std::find(broadcast_axes.begin(), broadcast_axes.end(), i) == broadcast_axes.end()) {
      CHECK_EQ(x_extend_shape.At(i), y->shape().At(i));
    } else {
      CHECK_EQ(x_extend_shape.At(i), 1);
    }
  }
  NdarrayUtil<device_type, T>::BroadcastTo(ctx->stream(), XpuVarNdarray<T>(y, num_axes),
                                           XpuVarNdarray<const T>(x, num_axes));
}

#define REGISTTER_BROADCAST_TO_COMPATIBLE_WITH_KERNEL(device_type_v, dtype_pair)                 \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(                                                         \
      OperatorConf::kBroadcastToCompatibleWithConf, device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
      BroadcastToCompatibleWithKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair)>)

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTTER_BROADCAST_TO_COMPATIBLE_WITH_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ BOOL_DATA_TYPE_SEQ)

#if defined(WITH_CUDA)
REGISTTER_BROADCAST_TO_COMPATIBLE_WITH_KERNEL(DeviceType::kCUDA, (float16, DataType::kFloat16))
#endif

}  // namespace oneflow
