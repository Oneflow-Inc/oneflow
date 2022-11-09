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
#include <cmath>
namespace oneflow {

template<typename T>
class CpuErfinvKernel final : public user_op::OpKernel {
 public:
  CpuErfinvKernel() = default;
  ~CpuErfinvKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    const T* x_ptr = x->dptr<T>();
    T* y_ptr = y->mut_dptr<T>();
    constexpr float central_range = 0.7;
    const T temp = static_cast<T>(2.0) / static_cast<T>(std::sqrt(M_PI));
    T a[4] = {T(0.886226899), T(-1.645349621), T(0.914624893), T(-0.140543331)};
    T b[4] = {T(-2.118377725), T(1.442710462), T(-0.329097515), T(0.012229801)};
    T c[4] = {T(-1.970840454), T(-1.624906493), T(3.429567803), T(1.641345311)};
    T d[2] = {T(3.543889200), T(1.637067800)};
    FOR_RANGE(int32_t, i, 0, elem_cnt) {
      T z, num, dem;
      T x = x_ptr[i];  // Promise the correctness of inplace version.
      T x_abs = std::abs(x);
      if (x_abs > 1.0) {
        y_ptr[i] = std::numeric_limits<T>::quiet_NaN();
        continue;
      }
      if (x_abs == 1.0) {
        y_ptr[i] = std::copysign(std::numeric_limits<T>::infinity(), x);
        continue;
      }
      if (x_abs <= static_cast<T>(central_range)) {
        z = x * x;
        num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
        dem = ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + static_cast<T>(1.0));
        y_ptr[i] = x * num / dem;
      } else {
        z = std::sqrt(-std::log((static_cast<T>(1.0) - x_abs) / static_cast<T>(2.0)));
        num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
        dem = (d[1] * z + d[0]) * z + static_cast<T>(1.0);
        y_ptr[i] = std::copysign(num, x) / dem;
      }
      y_ptr[i] = y_ptr[i] - (std::erf(y_ptr[i]) - x) / (temp * std::exp(-y_ptr[i] * y_ptr[i]));
      y_ptr[i] = y_ptr[i] - (std::erf(y_ptr[i]) - x) / (temp * std::exp(-y_ptr[i] * y_ptr[i]));
    }
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_ERFINV_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("erfinv")                                                       \
      .SetCreateFn<CpuErfinvKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value)) \
      .SetInplaceProposalFn(                                                           \
          [](const user_op::InferContext&,                                             \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {   \
            OF_RETURN_IF_ERROR(AddInplaceArgPairFn("y", 0, "x", 0, true));             \
            return Maybe<void>::Ok();                                                  \
          });

REGISTER_CPU_ERFINV_KERNEL(float)
REGISTER_CPU_ERFINV_KERNEL(double)

}  // namespace oneflow
