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
#include "oneflow/core/ndarray/binary_func.h"
namespace oneflow {

namespace {
template<typename T, template<typename> class BinaryFunc>
void CumForward(const T* in_ptr, T* out_ptr, int64_t up_space, int64_t space, int64_t down_space,
                int64_t elem_cnt) {
  std::copy_n(in_ptr, elem_cnt, out_ptr);
  auto* tmp_out_ptr_base = out_ptr;
  auto step = space * down_space;
  for (auto i = 0; i < up_space; i++) {
    for (auto j = 1; j < space; j++) {
      auto* tmp_out_ptr = tmp_out_ptr_base + j * down_space;
      auto* last_tmp_out_ptr = tmp_out_ptr - down_space;
      for (auto k = 0; k < down_space; k++) {
        tmp_out_ptr[k] = BinaryFunc<T>::Invoke(tmp_out_ptr[k], last_tmp_out_ptr[k]);
      }
    }
    tmp_out_ptr_base += step;
  }
}
}  // namespace

template<typename T, template<typename> class BinaryFunc>
class CpuCumKernel : public user_op::OpKernel {
 public:
  CpuCumKernel() = default;
  ~CpuCumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const auto* in = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto elem_cnt = in->shape_view().elem_cnt();
    // judge whether tensor has 0 size dimension first
    if (!elem_cnt) { return; }

    auto* out = ctx->Tensor4ArgNameAndIndex("y", 0);
    auto dim = ctx->Attr<int64_t>("dim");
    const auto* in_ptr = in->dptr<T>();
    auto* out_ptr = out->mut_dptr<T>();

    // data partition: up_space|space|down_space
    auto up_space = elem_cnt / in->shape_view().Count(dim);
    auto space = in->shape_view().At(dim);
    auto down_space = in->shape_view().Count(dim + 1);

    CumForward<T, BinaryFunc>(in_ptr, out_ptr, up_space, space, down_space, elem_cnt);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define CUMOP_SEQ                                \
  OF_PP_MAKE_TUPLE_SEQ("cumprod", BinaryFuncMul) \
  OF_PP_MAKE_TUPLE_SEQ("cumsum", BinaryFuncAdd)

#define REGISTER_CUMOP_KERNEL(dtype, op_name, op_functor)                                       \
  REGISTER_USER_KERNEL(op_name).SetCreateFn<CpuCumKernel<dtype, op_functor>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                            \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_CUMOP_KERNEL_WITH_DTYPE(op_name, op_functor) \
  REGISTER_CUMOP_KERNEL(int32_t, op_name, op_functor)         \
  REGISTER_CUMOP_KERNEL(int64_t, op_name, op_functor)         \
  REGISTER_CUMOP_KERNEL(float, op_name, op_functor)           \
  REGISTER_CUMOP_KERNEL(double, op_name, op_functor)

OF_PP_FOR_EACH_TUPLE(REGISTER_CUMOP_KERNEL_WITH_DTYPE, CUMOP_SEQ);

#undef REGISTER_CUMOP_KERNEL
#undef REGISTER_CUMOP_KERNEL_WITH_DTYPE
#undef CUMOP_SEQ
}  // namespace oneflow
