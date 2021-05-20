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
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {
namespace user_op {

class SummaMatmulABKernelCommState final : public user_op::OpKernelState {
 public:
  SummaMatmulABKernelCommState(user_op::KernelInitContext* ctx)
      : is_init_(false),
        parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()) {
    OF_CUDA_CHECK(cudaStreamCreate(&nccl_stream_));
  }
  ~SummaMatmulABKernelCommState() { cudaStreamDestroy(nccl_stream_); };

  ncclComm_t a_comm() {
    if (!is_init_) { Init(); }
    return a_comm_;
  }

  ncclComm_t b_comm() {
    if (!is_init_) { Init(); }
    return b_comm_;
  }

  int64_t num_ranks() {
    if (!is_init_) { Init(); }
    return num_ranks_;
  }

  cudaStream_t nccl_stream() {
    if (!is_init_) { Init(); }
    return nccl_stream_;
  }

 private:
  void Init() {
    CHECK(!is_init_);
    std::set<std::pair<int64_t, int64_t>> a_device_set;
    std::set<std::pair<int64_t, int64_t>> b_device_set;
    const Shape& hierarchy = *parallel_desc_.hierarchy();
    CHECK_EQ(hierarchy.NumAxes(), 2);
    CHECK_EQ(hierarchy.At(0), hierarchy.At(1));
    const int64_t q = hierarchy.At(0);
    int64_t row_rank = this_parallel_id_ / q;
    int64_t col_rank = this_parallel_id_ % q;
    for (int64_t i = 0; i < q; ++i) {
      const int64_t parallel_id = row_rank * q + i;
      const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      a_device_set.emplace(std::make_pair(machine_id, device_id));
    }
    for (int64_t i = 0; i < q; ++i) {
      const int64_t parallel_id = i * q + col_rank;
      const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      b_device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get());
    a_comm_ = comm_mgr->GetCommForDevice(a_device_set);
    b_comm_ = comm_mgr->GetCommForDevice(b_device_set);
    num_ranks_ = q;
    is_init_ = true;
  }

  bool is_init_;
  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  int64_t num_ranks_;
  ncclComm_t a_comm_;
  ncclComm_t b_comm_;
  cudaStream_t nccl_stream_;
};

template<typename T>
class SummaMatmulABKernel final : public user_op::OpKernel {
 public:
  SummaMatmulABKernel() = default;
  ~SummaMatmulABKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<SummaMatmulABKernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<SummaMatmulABKernelCommState*>(state);
    CHECK(kernel_state != nullptr);
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t num_axes = a->shape().NumAxes();
    const int m = out->shape().At(num_axes - 2);
    const int n = out->shape().At(num_axes - 1);
    const int k = a->shape().At(num_axes - 1);
    char* a_buffer_0 = tmp_buffer->mut_dptr<char>();
    char* a_buffer_1 = tmp_buffer->mut_dptr<char>() + a->shape().elem_cnt() * sizeof(T);
    char* b_buffer_0 = tmp_buffer->mut_dptr<char>() + 2 * a->shape().elem_cnt() * sizeof(T);
    char* b_buffer_1 = tmp_buffer->mut_dptr<char>() + 2 * a->shape().elem_cnt() * sizeof(T)
                       + b->shape().elem_cnt() * sizeof(T);
    std::vector<void*> a_buffer;
    a_buffer.push_back(reinterpret_cast<void*>(a_buffer_0));
    a_buffer.push_back(reinterpret_cast<void*>(a_buffer_1));
    std::vector<void*> b_buffer;
    b_buffer.push_back(reinterpret_cast<void*>(b_buffer_0));
    b_buffer.push_back(reinterpret_cast<void*>(b_buffer_1));

    const double alpha = ctx->Attr<double>("alpha");
    double beta = 1.0;
    int q = kernel_state->num_ranks();
    const int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    const int64_t row_rank = parallel_id / q;
    const int64_t col_rank = parallel_id % q;
    cudaEvent_t start_comm_event;
    cudaEventCreate(&start_comm_event);

    cudaEvent_t data_release_event[q];
    cudaEvent_t buffer_free_event[q];
    FOR_RANGE(int64_t, i, 0, q) {
      cudaEventCreate(&data_release_event[i]);
      cudaEventCreate(&buffer_free_event[i]);
    }
    cudaEventRecord(start_comm_event, ctx->device_ctx()->cuda_stream());
    cudaStreamWaitEvent(kernel_state->nccl_stream(), start_comm_event, 0);
    OF_NCCL_CHECK(ncclBroadcast(a->dptr(), a_buffer.at(0), a->shape().elem_cnt(),
                                GetNcclDataType(a->data_type()), 0, kernel_state->a_comm(),
                                kernel_state->nccl_stream()));
    OF_NCCL_CHECK(ncclBroadcast(b->dptr(), b_buffer.at(0), b->shape().elem_cnt(),
                                GetNcclDataType(b->data_type()), 0, kernel_state->b_comm(),
                                kernel_state->nccl_stream()));
    cudaEventRecord(data_release_event[0], kernel_state->nccl_stream());
    cudaEventRecord(buffer_free_event[0], ctx->device_ctx()->cuda_stream());
    for (int64_t i = 1; i < q; ++i) {
      cudaStreamWaitEvent(kernel_state->nccl_stream(), buffer_free_event[i - 1], 0);
      OF_NCCL_CHECK(ncclBroadcast(a->dptr(), a_buffer.at(i % 2), a->shape().elem_cnt(),
                                  GetNcclDataType(a->data_type()), i, kernel_state->a_comm(),
                                  kernel_state->nccl_stream()));
      OF_NCCL_CHECK(ncclBroadcast(b->dptr(), b_buffer.at(i % 2), b->shape().elem_cnt(),
                                  GetNcclDataType(b->data_type()), i, kernel_state->b_comm(),
                                  kernel_state->nccl_stream()));
      cudaEventRecord(data_release_event[i], kernel_state->nccl_stream());
      cudaStreamWaitEvent(ctx->device_ctx()->cuda_stream(), data_release_event[i - 1], 0);

      const T* a_ptr = reinterpret_cast<T*>(a_buffer.at((i - 1) % 2));
      const T* b_ptr = reinterpret_cast<T*>(b_buffer.at((i - 1) % 2));
      NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx->device_ctx(), CblasNoTrans, CblasNoTrans, m, n,
                                              k, alpha, a_ptr, b_ptr, beta, out->mut_dptr<T>());
      cudaEventRecord(buffer_free_event[i], ctx->device_ctx()->cuda_stream());
    }
    cudaStreamWaitEvent(ctx->device_ctx()->cuda_stream(), data_release_event[q - 1], 0);
    NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx->device_ctx(), CblasNoTrans, CblasNoTrans, m, n, k,
                                            alpha, reinterpret_cast<T*>(a_buffer.at(1)),
                                            reinterpret_cast<T*>(b_buffer.at(1)), beta,
                                            out->mut_dptr<T>());
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SUMMA_MATMUL_AB_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("summa_matmul_ab")                                                     \
      .SetCreateFn<SummaMatmulABKernel<dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                          \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value))         \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                     \
        const TensorDesc* a_desc = ctx->TensorDesc4ArgNameAndIndex("a", 0);                   \
        const TensorDesc* b_desc = ctx->TensorDesc4ArgNameAndIndex("b", 0);                   \
        return 2 * (a_desc->shape().elem_cnt() + b_desc->shape().elem_cnt()) * sizeof(dtype); \
      });

#ifdef WITH_CUDA
REGISTER_SUMMA_MATMUL_AB_KERNEL(float16);
REGISTER_SUMMA_MATMUL_AB_KERNEL(float);
REGISTER_SUMMA_MATMUL_AB_KERNEL(double);
#endif
}  // namespace user_op
}  // namespace oneflow
