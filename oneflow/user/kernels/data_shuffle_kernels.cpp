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

namespace oneflow {

namespace {}  // namespace

template<typename T>
class IdShuffleKernel final : public user_op::OpKernel {
 public:
  IdShuffleKernel() = default;
  ~IdShuffleKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "IdShuffleKernel";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_ID_SHUFFLE_KERNEL(dtype)                         \
  REGISTER_USER_KERNEL("id_shuffle")                                   \
      .SetCreateFn<IdShuffleKernel<dtype>>()                           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("ids", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_ID_SHUFFLE_KERNEL(int32_t)
REGISTER_CUDA_ID_SHUFFLE_KERNEL(int64_t)

template<typename T>
class EmbeddingShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingShuffleKernel() = default;
  ~EmbeddingShuffleKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "EmbeddingShuffleKernel";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL(dtype)                  \
  REGISTER_USER_KERNEL("embedding_shuffle")                            \
      .SetCreateFn<EmbeddingShuffleKernel<dtype>>()                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("embeddings", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_EMBEDDING_SHUFFLE_KERNEL(float)

template<typename T>
class EmbeddingGradientShuffleKernel final : public user_op::OpKernel {
 public:
  EmbeddingGradientShuffleKernel() = default;
  ~EmbeddingGradientShuffleKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    LOG(ERROR) << "EmbeddingGradientShuffleKernel";
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL(dtype) \
  REGISTER_USER_KERNEL("embedding_gradient_shuffle")           \
      .SetCreateFn<EmbeddingGradientShuffleKernel<dtype>>()    \
      .SetIsMatchedHob(                                        \
          (user_op::HobDeviceType() == DeviceType::kCUDA)      \
          && (user_op::HobDataType("embedding_diff", 0) == GetDataType<dtype>::value));

REGISTER_CUDA_EMBEDDING_GRADIENT_SHUFFLE_KERNEL(float)

}  // namespace oneflow
