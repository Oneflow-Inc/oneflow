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

class SummaMatmulKernelCommState final : public user_op::OpKernelState {
 public:
  SummaMatmulKernelCommState(user_op::KernelInitContext* ctx)
      : parallel_desc_(ctx->parallel_desc()), this_parallel_id_(ctx->parallel_ctx().parallel_id()) {
    OF_CUDA_CHECK(cudaStreamCreate(&row_nccl_stream_));
    OF_CUDA_CHECK(cudaStreamCreate(&col_nccl_stream_));
    OF_CUDA_CHECK(cudaEventCreate(&start_comm_event_));
    CHECK_EQ(parallel_desc_.hierarchy()->NumAxes(), 2);
    CHECK_EQ(parallel_desc_.hierarchy()->At(0), parallel_desc_.hierarchy()->At(1));
    summa_dim_ = parallel_desc_.hierarchy()->At(0);
    row_data_release_events_.resize(summa_dim_);
    col_data_release_events_.resize(summa_dim_);
    buffer_free_events_.resize(summa_dim_);
    FOR_RANGE(int64_t, i, 0, summa_dim_) {
      OF_CUDA_CHECK(cudaEventCreate(&row_data_release_events_.at(i)));
      OF_CUDA_CHECK(cudaEventCreate(&col_data_release_events_.at(i)));
      OF_CUDA_CHECK(cudaEventCreate(&buffer_free_events_.at(i)));
    }
  }
  ~SummaMatmulKernelCommState() {
    OF_CUDA_CHECK(cudaStreamDestroy(row_nccl_stream_));
    OF_CUDA_CHECK(cudaStreamDestroy(col_nccl_stream_));
    OF_CUDA_CHECK(cudaEventDestroy(start_comm_event_));
    CHECK_EQ(buffer_free_events_.size(), row_data_release_events_.size());
    FOR_RANGE(int64_t, i, 0, buffer_free_events_.size()) {
      OF_CUDA_CHECK(cudaEventDestroy(buffer_free_events_.at(i)));
      OF_CUDA_CHECK(cudaEventDestroy(row_data_release_events_.at(i)));
      OF_CUDA_CHECK(cudaEventDestroy(col_data_release_events_.at(i)));
    }
  };

  ncclComm_t row_comm() { return GetOrCreate().row_comm; }

  ncclComm_t col_comm() { return GetOrCreate().col_comm; }

  int64_t summa_dim() { return summa_dim_; }

  cudaStream_t row_nccl_stream() { return row_nccl_stream_; }

  cudaStream_t col_nccl_stream() { return col_nccl_stream_; }

  cudaEvent_t start_comm_event() { return start_comm_event_; }

  const std::vector<cudaEvent_t>& row_data_release_events() { return row_data_release_events_; }

  const std::vector<cudaEvent_t>& col_data_release_events() { return col_data_release_events_; }

  const std::vector<cudaEvent_t>& buffer_free_events() { return buffer_free_events_; }

  struct Comm {
    Comm(ncclComm_t row_comm, ncclComm_t col_comm) : row_comm(row_comm), col_comm(col_comm) {}
    ncclComm_t row_comm;
    ncclComm_t col_comm;
  };

  std::unique_ptr<Comm> comm_;

  const Comm& GetOrCreate() {
    if (!comm_) { Init(); }
    return *comm_;
  }

 private:
  void Init() {
    std::set<std::pair<int64_t, int64_t>> a_device_set;
    std::set<std::pair<int64_t, int64_t>> b_device_set;
    int64_t row_rank = this_parallel_id_ / summa_dim_;
    int64_t col_rank = this_parallel_id_ % summa_dim_;
    for (int64_t i = 0; i < summa_dim_; ++i) {
      const int64_t parallel_id = row_rank * summa_dim_ + i;
      const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      a_device_set.emplace(std::make_pair(machine_id, device_id));
    }
    for (int64_t i = 0; i < summa_dim_; ++i) {
      const int64_t parallel_id = i * summa_dim_ + col_rank;
      const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      b_device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get());
    ncclComm_t row_comm = comm_mgr->GetCommForDevice(a_device_set);
    ncclComm_t col_comm = comm_mgr->GetCommForDevice(b_device_set);
    comm_.reset(new Comm(row_comm, col_comm));
  }

  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  int64_t summa_dim_;
  cudaStream_t row_nccl_stream_;
  cudaStream_t col_nccl_stream_;
  cudaEvent_t start_comm_event_;
  std::vector<cudaEvent_t> row_data_release_events_;
  std::vector<cudaEvent_t> col_data_release_events_;
  std::vector<cudaEvent_t> buffer_free_events_;
};

template<typename T>
class SummaMatmulABKernel final : public user_op::OpKernel {
 public:
  SummaMatmulABKernel() = default;
  ~SummaMatmulABKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<SummaMatmulKernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<SummaMatmulKernelCommState*>(state);
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
    int summa_dim = kernel_state->summa_dim();
    OF_CUDA_CHECK(
        cudaEventRecord(kernel_state->start_comm_event(), ctx->device_ctx()->cuda_stream()));
    OF_CUDA_CHECK(
        cudaStreamWaitEvent(kernel_state->row_nccl_stream(), kernel_state->start_comm_event(), 0));
    OF_NCCL_CHECK(ncclBroadcast(a->dptr(), a_buffer.at(0), a->shape().elem_cnt(),
                                GetNcclDataType(a->data_type()), 0, kernel_state->row_comm(),
                                kernel_state->row_nccl_stream()));
    OF_CUDA_CHECK(cudaEventRecord(kernel_state->row_data_release_events().at(0),
                                  kernel_state->row_nccl_stream()));
    OF_CUDA_CHECK(
        cudaStreamWaitEvent(kernel_state->col_nccl_stream(), kernel_state->start_comm_event(), 0));
    OF_NCCL_CHECK(ncclBroadcast(b->dptr(), b_buffer.at(0), b->shape().elem_cnt(),
                                GetNcclDataType(b->data_type()), 0, kernel_state->col_comm(),
                                kernel_state->col_nccl_stream()));
    OF_CUDA_CHECK(cudaEventRecord(kernel_state->col_data_release_events().at(0),
                                  kernel_state->col_nccl_stream()));
    OF_CUDA_CHECK(cudaEventRecord(kernel_state->buffer_free_events().at(0),
                                  ctx->device_ctx()->cuda_stream()));
    for (int64_t i = 1; i < summa_dim; ++i) {
      OF_CUDA_CHECK(cudaStreamWaitEvent(kernel_state->row_nccl_stream(),
                                        kernel_state->buffer_free_events().at(i - 1), 0));
      OF_NCCL_CHECK(ncclBroadcast(a->dptr(), a_buffer.at(i % 2), a->shape().elem_cnt(),
                                  GetNcclDataType(a->data_type()), i, kernel_state->row_comm(),
                                  kernel_state->row_nccl_stream()));
      OF_CUDA_CHECK(cudaEventRecord(kernel_state->row_data_release_events().at(i),
                                    kernel_state->row_nccl_stream()));
      OF_CUDA_CHECK(cudaStreamWaitEvent(kernel_state->col_nccl_stream(),
                                        kernel_state->buffer_free_events().at(i - 1), 0));
      OF_NCCL_CHECK(ncclBroadcast(b->dptr(), b_buffer.at(i % 2), b->shape().elem_cnt(),
                                  GetNcclDataType(b->data_type()), i, kernel_state->col_comm(),
                                  kernel_state->col_nccl_stream()));
      OF_CUDA_CHECK(cudaEventRecord(kernel_state->col_data_release_events().at(i),
                                    kernel_state->col_nccl_stream()));
      OF_CUDA_CHECK(cudaStreamWaitEvent(ctx->device_ctx()->cuda_stream(),
                                        kernel_state->row_data_release_events().at(i - 1), 0));
      OF_CUDA_CHECK(cudaStreamWaitEvent(ctx->device_ctx()->cuda_stream(),
                                        kernel_state->col_data_release_events().at(i - 1), 0));
      const T* a_ptr = reinterpret_cast<T*>(a_buffer.at((i - 1) % 2));
      const T* b_ptr = reinterpret_cast<T*>(b_buffer.at((i - 1) % 2));
      NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx->device_ctx(), CblasNoTrans, CblasNoTrans, m, n,
                                              k, alpha, a_ptr, b_ptr, beta, out->mut_dptr<T>());
      OF_CUDA_CHECK(cudaEventRecord(kernel_state->buffer_free_events().at(i),
                                    ctx->device_ctx()->cuda_stream()));
    }
    OF_CUDA_CHECK(cudaStreamWaitEvent(ctx->device_ctx()->cuda_stream(),
                                      kernel_state->row_data_release_events().at(summa_dim - 1),
                                      0));
    OF_CUDA_CHECK(cudaStreamWaitEvent(ctx->device_ctx()->cuda_stream(),
                                      kernel_state->col_data_release_events().at(summa_dim - 1),
                                      0));
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
