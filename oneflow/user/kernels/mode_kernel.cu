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
#include <cub/cub.cuh>
#include <device_launch_parameters.h>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/user/kernels/radix_sort.cuh"

namespace oneflow {


template<typename T>
class CudaModeKernel final : public user_op::OpKernel {
 public:
  CudaModeKernel() = default;
  ~CudaModeKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
   
  //  大概思路
  //  1. 排序
  //  2. 使用 cub::DeviceRunLengthEncode::NonTrivialRuns 获得每一个元素的数量
  //  https://nvlabs.github.io/cub/structcub_1_1_device_run_length_encode.html#aa2318dc7a69f28a8c47d417aaf53db3a
  //  3. 获取最大数量的元素的值和索引,并根据排序时的索引得到原始位置的索引

  // 难点:
  // 1. InferTmpSize 是否每个开辟了辅助内存的函数都需要将其内存size加上这里
  // 2. 核函数的启动和参数的设置（多维数组需要设置迭代器？？ 不太清楚  还需要看一下其他算子的核函数的代码）
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};


#define REGISTER_CUDA_MODE_KERNEL(dtype)                                                   \
  REGISTER_USER_KERNEL("mode")                                                             \
      .SetCreateFn<CudaModeKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        const Shape& in_shape = ctx->InputShape("input", 0);                               \
        const int64_t instance_size = in_shape.dim_vec().back();                           \
        const int64_t instance_num = in_shape.elem_cnt() / instance_size;                  \
        return 0;                                                                          \
      });

REGISTER_CUDA_MODE_KERNEL(float)
REGISTER_CUDA_MODE_KERNEL(double)
REGISTER_CUDA_MODE_KERNEL(int8_t)
REGISTER_CUDA_MODE_KERNEL(uint8_t)
REGISTER_CUDA_MODE_KERNEL(int32_t)
REGISTER_CUDA_MODE_KERNEL(int64_t)

}  // namespace oneflow
