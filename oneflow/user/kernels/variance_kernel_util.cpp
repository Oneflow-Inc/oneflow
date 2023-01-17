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
#include "oneflow/user/kernels/variance_kernel_util.h"

namespace oneflow {

namespace user_op {

template<typename T, typename ComputeType>
struct VarFunctor<DeviceType::kCPU, T, ComputeType> final {
  void operator()(ep::Stream* stream, const T* in_ptr, T* out_ptr, ComputeType* tmp_buffer_ptr,
                  const VarParam var_param) {
    // if var_param.parallel_num is 0, do nothing, return 0-size tensor
    if (IsNanOut(var_param)) {
      for (size_t i = 0; i < var_param.parallel_num; i++) {
        out_ptr[i] = std::numeric_limits<T>::quiet_NaN();
      }
    } else {
      for (size_t i = 0; i < var_param.parallel_num; i++) {
        const size_t input_offset = LinearIndex2Offset(
            i, var_param.dim_size_in_caxis, var_param.stride_in_caxis, var_param.caxis_size);
        ComputeVarUsingWelford<T, ComputeType>(&in_ptr[input_offset], &out_ptr[i], var_param);
      }
    }
  }
};

template struct VarFunctor<DeviceType::kCPU, float, double>;
template struct VarFunctor<DeviceType::kCPU, double, double>;
template struct VarFunctor<DeviceType::kCPU, float16, double>;
template struct VarFunctor<DeviceType::kCPU, bfloat16, double>;
}  // namespace user_op
}  // namespace oneflow
