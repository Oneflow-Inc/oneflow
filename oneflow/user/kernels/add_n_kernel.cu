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

namespace {

template<typename T, int32_t N>
struct Param {
  const T* in[N];
  T* out;
};

template<typename T, int32_t N>
__global__ void gpu_add(const int64_t n, Param<T, N> para) {
  if (para.out == para.in[0]) {
    CUDA_1D_KERNEL_LOOP(i, n) {
      T tmp = 0;
#pragma unroll
      for (int j = 1; j < N; ++j) { tmp += para.in[j][i]; }
      if (tmp != 0) { para.out[i] += tmp; }
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, n) {
      T tmp = para.in[0][i];
#pragma unroll
      for (int j = 1; j < N; ++j) { tmp += para.in[j][i]; }
      para.out[i] = tmp;
    }
  }
}

template<typename T, int32_t N>
struct GpuAddCaller {
  static void call(user_op::KernelComputeContext* ctx) {
    CHECK_EQ(N, ctx->inputs().size());

    Param<T, N> para;
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = out->shape().elem_cnt();
    para.out = out->mut_dptr<T>();
    for (int32_t i = 0; i < N; ++i) {
      para.in[i] = ctx->Tensor4ArgNameAndIndex("in", i)->dptr<T>();
    }

    gpu_add<T, N>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, para);
  }
};

using CallFn = std::function<void(user_op::KernelComputeContext*)>;
using AddNKernelRegistry = std::map<int32_t, CallFn>;

#define ADD_NUM_PARAM_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(2) \
  OF_PP_MAKE_TUPLE_SEQ(3) \
  OF_PP_MAKE_TUPLE_SEQ(4) \
  OF_PP_MAKE_TUPLE_SEQ(5) \
  OF_PP_MAKE_TUPLE_SEQ(6) \
  OF_PP_MAKE_TUPLE_SEQ(7) \
  OF_PP_MAKE_TUPLE_SEQ(8)

template<typename T>
const AddNKernelRegistry& SingletonRegistry() {
  static AddNKernelRegistry s_registry = {
#define REG_ENTRY(n) {n, &GpuAddCaller<T, n>::call},
      OF_PP_FOR_EACH_TUPLE(REG_ENTRY, ADD_NUM_PARAM_SEQ)
#undef REG_ENTRY
  };
  return s_registry;
}

template<typename T>
const CallFn* LookUpInRegistry(int32_t in_num) {
  auto it = SingletonRegistry<T>().find(in_num);
  if (it == SingletonRegistry<T>().end()) { return nullptr; }
  return &(it->second);
}

}  // namespace

template<typename T>
class GpuAddNKernel : public user_op::OpKernel {
 public:
  GpuAddNKernel() = default;
  ~GpuAddNKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    int32_t in_num = ctx->inputs().size();

    const auto* caller = LookUpInRegistry<T>(in_num);
    CHECK(caller != nullptr) << "GpuAddNKernel: Cannot find registered funtion for in_num: "
                             << in_num << " of data_type: " << DataType_Name(GetDataType<T>::value);
    (*caller)(ctx);
  }
};

#define REGISTER_GPU_ADDN_KERNEL(cpp_type, dtype)                                               \
  REGISTER_USER_KERNEL("add_n")                                                                 \
      .SetCreateFn<GpuAddNKernel<cpp_type>>()                                                   \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")                                       \
                       & (user_op::HobDataType("in", 0) == dtype))                              \
      .SetInplaceProposalFn([](const user_op::InferContext&,                                    \
                               user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> { \
        OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));                       \
        return Maybe<void>::Ok();                                                               \
      });

OF_PP_FOR_EACH_TUPLE(REGISTER_GPU_ADDN_KERNEL, ARITHMETIC_DATA_TYPE_SEQ);

namespace {

template<int32_t N>
__global__ void gpu_half_add(const int64_t n, Param<half, N> para) {
  if (para.out == para.in[0]) {
    CUDA_1D_KERNEL_LOOP(i, n) {
      half tmp = 0;
#pragma unroll
      for (int j = 1; j < N; ++j) { tmp = __hadd(tmp, para.in[j][i]); }
      para.out[i] = __hadd(para.out[i], tmp);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(i, n) {
      half tmp = para.in[0][i];
#pragma unroll
      for (int j = 1; j < N; ++j) { tmp = __hadd(tmp, para.in[j][i]); }
      para.out[i] = tmp;
    }
  }
}

template<int32_t N>
struct GpuAddCaller<float16, N> {
  static void call(user_op::KernelComputeContext* ctx) {
    CHECK_EQ(N, ctx->inputs().size());

    Param<half, N> para;
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = out->shape().elem_cnt();
    para.out = reinterpret_cast<half*>(out->mut_dptr<float16>());
    for (int32_t i = 0; i < N; ++i) {
      para.in[i] =
          reinterpret_cast<const half*>(ctx->Tensor4ArgNameAndIndex("in", i)->dptr<float16>());
    }

    gpu_half_add<N>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->device_ctx()->cuda_stream()>>>(
            n, para);
  }
};

}  // namespace

class GpuAddNHalfKernel : public user_op::OpKernel {
 public:
  GpuAddNHalfKernel() = default;
  ~GpuAddNHalfKernel() = default;

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    int32_t in_num = ctx->inputs().size();

    const auto* caller = LookUpInRegistry<float16>(in_num);
    CHECK(caller != nullptr) << "GpuAddNHalfKernel: Cannot find registered funtion for in_num: "
                             << in_num << " of data_type: " << DataType_Name(DataType::kFloat16);
    (*caller)(ctx);
  }
};

REGISTER_USER_KERNEL("add_n")
    .SetCreateFn<GpuAddNHalfKernel>()
    .SetIsMatchedHob((user_op::HobDeviceTag() == "gpu")
                     & (user_op::HobDataType("in", 0) == DataType::kFloat16))
    .SetInplaceProposalFn([](const user_op::InferContext&,
                             user_op::AddInplaceArgPair AddInplaceArgPairFn) -> Maybe<void> {
      OF_RETURN_IF_ERROR(AddInplaceArgPairFn("out", 0, "in", 0, true));
      return Maybe<void>::Ok();
    });

}  // namespace oneflow
