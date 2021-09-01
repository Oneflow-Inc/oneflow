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
#include "oneflow/core/control/ctrl_client.h"

namespace oneflow {

namespace {
int64_t Gcd(int64_t a, int64_t b) { return b == 0 ? a : Gcd(b, a % b); }
}  // namespace

namespace user_op {

class SummaMatmulKernelCommState final : public user_op::OpKernelState {
 public:
  SummaMatmulKernelCommState(user_op::KernelInitContext* ctx)
      : parallel_desc_(ctx->parallel_desc()),
        this_parallel_id_(ctx->parallel_ctx().parallel_id()),
        op_name_(ctx->op_name()) {
    CHECK_EQ(parallel_desc_.hierarchy()->NumAxes(), 2);
    summa_dim0_ = parallel_desc_.hierarchy()->At(0);
    summa_dim1_ = parallel_desc_.hierarchy()->At(1);
    lcm_summa_dim_ = summa_dim0_ * summa_dim1_ / Gcd(summa_dim0_, summa_dim1_);
  }

  ncclComm_t row_comm() { return GetOrCreate().row_comm; }

  ncclComm_t col_comm() { return GetOrCreate().col_comm; }

  int64_t summa_dim0() { return summa_dim0_; }
  int64_t summa_dim1() { return summa_dim1_; }
  int64_t lcm_summa_dim() { return lcm_summa_dim_; }
  int64_t parallel_id() { return this_parallel_id_; }

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
    std::set<std::pair<int64_t, int64_t>> row_device_set;
    std::set<std::pair<int64_t, int64_t>> col_device_set;
    int64_t row_rank = this_parallel_id_ / summa_dim1_;
    int64_t col_rank = this_parallel_id_ % summa_dim1_;
    for (int64_t i = 0; i < summa_dim1_; ++i) {
      const int64_t parallel_id = row_rank * summa_dim1_ + i;
      const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      row_device_set.emplace(std::make_pair(machine_id, device_id));
    }
    for (int64_t i = 0; i < summa_dim0_; ++i) {
      const int64_t parallel_id = i * summa_dim1_ + col_rank;
      const int64_t machine_id = CHECK_JUST(parallel_desc_.MachineId4ParallelId(parallel_id));
      const int64_t device_id = CHECK_JUST(parallel_desc_.DeviceId4ParallelId(parallel_id));
      col_device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Global<EagerNcclCommMgr>::Get());
    ncclComm_t row_comm = comm_mgr->GetCommForDevice(row_device_set);
    ncclComm_t col_comm = comm_mgr->GetCommForDevice(col_device_set);
    comm_.reset(new Comm(row_comm, col_comm));
    Global<CtrlClient>::Get()->Barrier(op_name_, parallel_desc_.parallel_num());
  }

  ParallelDesc parallel_desc_;
  int64_t this_parallel_id_;
  int64_t summa_dim0_;
  int64_t summa_dim1_;
  int64_t lcm_summa_dim_;
  std::string op_name_;
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
    CHECK(kernel_state->row_comm());
    CHECK(kernel_state->col_comm());
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t num_axes = a->shape().NumAxes();
    const int n = out->shape().At(num_axes - 1);
    const int m = out->shape().elem_cnt() / n;
    const int a_k = a->shape().At(num_axes - 1);
    const int b_k = b->shape().At(0);
    CHECK_EQ(a->shape().elem_cnt() / a_k, m);
    CHECK_EQ(b->shape().elem_cnt() / b_k, n);
    const double alpha = ctx->Attr<double>("alpha");
    double beta = 1.0;
    Memset<DeviceType::kGPU>(ctx->device_ctx(), out->mut_dptr(), 0,
                             out->shape().elem_cnt() * sizeof(T));

    const int64_t lcm_summa_dim = kernel_state->lcm_summa_dim();
    const int64_t num_a = lcm_summa_dim / kernel_state->summa_dim1();
    const int64_t summa_k = a_k / num_a;
    const int64_t a_send_elem_cnt = m * summa_k;
    const int64_t num_b = lcm_summa_dim / kernel_state->summa_dim0();
    CHECK_EQ(b_k / num_b, summa_k);
    const int64_t b_send_elem_cnt = summa_k * n;

    const size_t a_recv_buffer_size = GetCudaAlignedSize(a_send_elem_cnt * sizeof(T));
    const size_t a_send_buffer_size = num_a > 1 ? a_recv_buffer_size : 0;
    void* a_send_buffer = reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>());
    void* a_recv_buffer =
        reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + a_send_buffer_size);
    void* b_recv_buffer = reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + a_send_buffer_size
                                                  + a_recv_buffer_size);
    CHECK_EQ(tmp_buffer->shape().elem_cnt(),
             a_send_buffer_size + a_recv_buffer_size + b_send_elem_cnt * sizeof(T));

    const int64_t cur_a_rank = kernel_state->parallel_id() % kernel_state->summa_dim1();
    const void* a_ptr = a->dptr();
    for (int64_t i = 0; i < lcm_summa_dim; ++i) {
      const int64_t a_data_id = i % num_a;
      const int64_t a_rank_id = i / num_a;
      if (num_a > 1 && cur_a_rank == a_rank_id) {
        NewKernelUtil<DeviceType::kGPU>::CopyColsRegion(
            ctx->device_ctx(), m, summa_k, a->dptr<T>(), a_data_id * summa_k, a_k,
            reinterpret_cast<T*>(a_send_buffer), 0, summa_k);
        a_ptr = a_send_buffer;
      }
      OF_NCCL_CHECK(ncclBroadcast(a_ptr, a_recv_buffer, a_send_elem_cnt,
                                  GetNcclDataType(a->data_type()), a_rank_id,
                                  kernel_state->row_comm(), ctx->device_ctx()->cuda_stream()));

      const int64_t b_data_id = i % num_b;
      const int64_t b_rank_id = i / num_b;
      const void* b_ptr = reinterpret_cast<const void*>(b->dptr<T>() + b_data_id * b_send_elem_cnt);
      OF_NCCL_CHECK(ncclBroadcast(b_ptr, b_recv_buffer, b_send_elem_cnt,
                                  GetNcclDataType(b->data_type()), b_rank_id,
                                  kernel_state->col_comm(), ctx->device_ctx()->cuda_stream()));

      NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx->device_ctx(), CblasNoTrans, CblasNoTrans, m, n,
                                              summa_k, alpha, reinterpret_cast<T*>(a_recv_buffer),
                                              reinterpret_cast<T*>(b_recv_buffer), beta,
                                              out->mut_dptr<T>());
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SUMMA_MATMUL_AB_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("summa_matmul")                                                      \
      .SetCreateFn<SummaMatmulABKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                        \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value)        \
                       & (user_op::HobAttr<bool>("transpose_a") == false)                   \
                       & (user_op::HobAttr<bool>("transpose_b") == false))                  \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const TensorDesc* a_desc = ctx->TensorDesc4ArgNameAndIndex("a", 0);                 \
        const TensorDesc* b_desc = ctx->TensorDesc4ArgNameAndIndex("b", 0);                 \
        const int64_t a_k = a_desc->shape().At(a_desc->shape().NumAxes() - 1);              \
        const int64_t b_k = b_desc->shape().At(0);                                          \
        const int64_t summa_k = Gcd(a_k, b_k);                                              \
        const int64_t num_a = a_k / summa_k;                                                \
        const int64_t num_b = b_k / summa_k;                                                \
        const int64_t num_a_buffer = (num_a > 1) ? 2 : 1;                                   \
        return num_a_buffer                                                                 \
                   * GetCudaAlignedSize(a_desc->shape().elem_cnt() / num_a * sizeof(dtype)) \
               + GetCudaAlignedSize(b_desc->shape().elem_cnt() / num_b * sizeof(dtype));    \
      });

#ifdef WITH_CUDA
REGISTER_SUMMA_MATMUL_AB_KERNEL(float16);
REGISTER_SUMMA_MATMUL_AB_KERNEL(float);
REGISTER_SUMMA_MATMUL_AB_KERNEL(double);
#endif

#define REGISTER_SUMMA_BROADCAST_MATMUL_AB_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("summa_broadcast_matmul")                                            \
      .SetCreateFn<SummaMatmulABKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                        \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value))       \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const TensorDesc* a_desc = ctx->TensorDesc4ArgNameAndIndex("a", 0);                 \
        const TensorDesc* b_desc = ctx->TensorDesc4ArgNameAndIndex("b", 0);                 \
        const int64_t a_k = a_desc->shape().At(a_desc->shape().NumAxes() - 1);              \
        const int64_t b_k = b_desc->shape().At(0);                                          \
        const int64_t summa_k = Gcd(a_k, b_k);                                              \
        const int64_t num_a = a_k / summa_k;                                                \
        const int64_t num_b = b_k / summa_k;                                                \
        const int64_t num_a_buffer = (num_a > 1) ? 2 : 1;                                   \
        return num_a_buffer                                                                 \
                   * GetCudaAlignedSize(a_desc->shape().elem_cnt() / num_a * sizeof(dtype)) \
               + GetCudaAlignedSize(b_desc->shape().elem_cnt() / num_b * sizeof(dtype));    \
      });

#ifdef WITH_CUDA
REGISTER_SUMMA_BROADCAST_MATMUL_AB_KERNEL(float16);
REGISTER_SUMMA_BROADCAST_MATMUL_AB_KERNEL(float);
REGISTER_SUMMA_BROADCAST_MATMUL_AB_KERNEL(double);
#endif

template<typename T>
class SummaMatmulABTKernel final : public user_op::OpKernel {
 public:
  SummaMatmulABTKernel() = default;
  ~SummaMatmulABTKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<SummaMatmulKernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<SummaMatmulKernelCommState*>(state);
    CHECK(kernel_state != nullptr);
    CHECK(kernel_state->row_comm());
    CHECK(kernel_state->col_comm());
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t num_axes = a->shape().NumAxes();
    const int c_n = out->shape().At(num_axes - 1);
    const int b_n = b->shape().At(0);
    const int m = out->shape().elem_cnt() / c_n;
    const int k = a->shape().At(num_axes - 1);
    CHECK_EQ(a->shape().elem_cnt() / k, m);
    CHECK_EQ(b->shape().elem_cnt() / b_n, k);

    const int64_t lcm_summa_dim = kernel_state->lcm_summa_dim();
    const int64_t num_c = lcm_summa_dim / kernel_state->summa_dim1();
    const int64_t summa_n = c_n / num_c;
    const int64_t c_send_elem_cnt = m * summa_n;
    const int64_t num_b = lcm_summa_dim / kernel_state->summa_dim0();
    CHECK_EQ(b_n / num_b, summa_n);
    const int64_t b_send_elem_cnt = summa_n * k;
    const size_t c_send_buffer_size = GetCudaAlignedSize(c_send_elem_cnt * sizeof(T));
    const size_t c_recv_buffer_size = num_c > 1 ? c_send_buffer_size : 0;
    const size_t b_buffer_size = GetCudaAlignedSize(b_send_elem_cnt * sizeof(T));
    CHECK_EQ(tmp_buffer->shape().elem_cnt(),
             c_send_buffer_size + c_recv_buffer_size + b_buffer_size);
    void* b_recv_buffer = reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>());
    void* out_send_buffer = reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + b_buffer_size);
    void* out_recv_buffer =
        reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + b_buffer_size + c_send_buffer_size);
    if (num_c == 1) { out_recv_buffer = out->mut_dptr(); }
    const double alpha = ctx->Attr<double>("alpha");
    double beta = 0.0;

    for (int64_t i = 0; i < lcm_summa_dim; ++i) {
      const int64_t b_data_id = i % num_b;
      const int64_t b_rank_id = i / num_b;
      const void* b_ptr = reinterpret_cast<const void*>(b->dptr<T>() + b_data_id * b_send_elem_cnt);
      OF_NCCL_CHECK(ncclBroadcast(b_ptr, b_recv_buffer, b_send_elem_cnt,
                                  GetNcclDataType(b->data_type()), b_rank_id,
                                  kernel_state->col_comm(), ctx->device_ctx()->cuda_stream()));

      NewKernelUtil<DeviceType::kGPU>::OFGemm(
          ctx->device_ctx(), CblasNoTrans, CblasTrans, m, summa_n, k, alpha, a->dptr<T>(),
          reinterpret_cast<T*>(b_recv_buffer), beta, reinterpret_cast<T*>(out_send_buffer));
      const int64_t c_data_id = i % num_c;
      const int64_t c_rank_id = i / num_c;

      OF_NCCL_CHECK(ncclReduce(out_send_buffer, out_recv_buffer, c_send_elem_cnt,
                               GetNcclDataType(out->data_type()), ncclRedOp_t::ncclSum, c_rank_id,
                               kernel_state->row_comm(), ctx->device_ctx()->cuda_stream()));
      if (num_c > 1) {
        NewKernelUtil<DeviceType::kGPU>::CopyColsRegion(
            ctx->device_ctx(), m, summa_n, reinterpret_cast<T*>(out_recv_buffer), 0, summa_n,
            out->mut_dptr<T>(), c_data_id * summa_n, c_n);
      }
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SUMMA_MATMUL_ABT_KERNEL(dtype)                                                  \
  REGISTER_USER_KERNEL("summa_matmul")                                                           \
      .SetCreateFn<SummaMatmulABTKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                             \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value)             \
                       & (user_op::HobAttr<bool>("transpose_a") == false)                        \
                       & (user_op::HobAttr<bool>("transpose_b") == true))                        \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const TensorDesc* b_desc = ctx->TensorDesc4ArgNameAndIndex("b", 0);                      \
        const TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);                  \
        const int c_n = out_desc->shape().At(out_desc->shape().NumAxes() - 1);                   \
        const int b_n = b_desc->shape().At(0);                                                   \
        const int summa_n = Gcd(c_n, b_n);                                                       \
        const int64_t num_b = b_n / summa_n;                                                     \
        const int64_t num_c = c_n / summa_n;                                                     \
        const int64_t num_c_buffer = num_c > 1 ? 2 : 1;                                          \
        return GetCudaAlignedSize(b_desc->shape().elem_cnt() / num_b * sizeof(dtype))            \
               + num_c_buffer                                                                    \
                     * GetCudaAlignedSize(out_desc->shape().elem_cnt() / num_c * sizeof(dtype)); \
      });

#ifdef WITH_CUDA
REGISTER_SUMMA_MATMUL_ABT_KERNEL(float16);
REGISTER_SUMMA_MATMUL_ABT_KERNEL(float);
REGISTER_SUMMA_MATMUL_ABT_KERNEL(double);
#endif

#define REGISTER_SUMMA_BROADCAST_MATMUL_ABT_KERNEL(dtype)                                        \
  REGISTER_USER_KERNEL("summa_broadcast_matmul_grad_a")                                          \
      .SetCreateFn<SummaMatmulABTKernel<dtype>>()                                                \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                             \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value))            \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const TensorDesc* b_desc = ctx->TensorDesc4ArgNameAndIndex("b", 0);                      \
        const TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);                  \
        const int c_n = out_desc->shape().At(out_desc->shape().NumAxes() - 1);                   \
        const int b_n = b_desc->shape().At(0);                                                   \
        const int summa_n = Gcd(c_n, b_n);                                                       \
        const int64_t num_b = b_n / summa_n;                                                     \
        const int64_t num_c = c_n / summa_n;                                                     \
        const int64_t num_c_buffer = num_c > 1 ? 2 : 1;                                          \
        return GetCudaAlignedSize(b_desc->shape().elem_cnt() / num_b * sizeof(dtype))            \
               + num_c_buffer                                                                    \
                     * GetCudaAlignedSize(out_desc->shape().elem_cnt() / num_c * sizeof(dtype)); \
      });

#ifdef WITH_CUDA
REGISTER_SUMMA_BROADCAST_MATMUL_ABT_KERNEL(float16);
REGISTER_SUMMA_BROADCAST_MATMUL_ABT_KERNEL(float);
REGISTER_SUMMA_BROADCAST_MATMUL_ABT_KERNEL(double);
#endif

template<typename T>
class SummaMatmulATBKernel final : public user_op::OpKernel {
 public:
  SummaMatmulATBKernel() = default;
  ~SummaMatmulATBKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<SummaMatmulKernelCommState>(ctx);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* kernel_state = dynamic_cast<SummaMatmulKernelCommState*>(state);
    CHECK(kernel_state != nullptr);
    CHECK(kernel_state->row_comm());
    CHECK(kernel_state->col_comm());
    const user_op::Tensor* a = ctx->Tensor4ArgNameAndIndex("a", 0);
    const user_op::Tensor* b = ctx->Tensor4ArgNameAndIndex("b", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    const int32_t num_axes = a->shape().NumAxes();
    const int n = b->shape().At(num_axes - 1);
    const int c_m = out->shape().elem_cnt() / n;
    const int k = b->shape().elem_cnt() / n;
    const int a_m = a->shape().At(num_axes - 1);
    CHECK_EQ(a->shape().elem_cnt() / a_m, k);

    const int64_t lcm_summa_dim = kernel_state->lcm_summa_dim();
    const int64_t num_a = lcm_summa_dim / kernel_state->summa_dim1();
    const int64_t summa_m = a_m / num_a;
    const int64_t a_send_elem_cnt = k * summa_m;
    const int64_t num_c = lcm_summa_dim / kernel_state->summa_dim0();
    CHECK_EQ(c_m / num_c, summa_m);
    const int64_t c_send_elem_cnt = summa_m * n;
    const size_t a_recv_buffer_size = GetCudaAlignedSize(a_send_elem_cnt * sizeof(T));
    const size_t a_send_buffer_size = num_a > 1 ? a_recv_buffer_size : 0;
    const size_t c_buffer_size = GetCudaAlignedSize(c_send_elem_cnt * sizeof(T));
    CHECK_EQ(tmp_buffer->shape().elem_cnt(),
             a_send_buffer_size + a_recv_buffer_size + c_buffer_size);
    void* a_send_buffer = reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>());
    void* a_recv_buffer =
        reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + a_send_buffer_size);
    void* out_send_buffer = reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>()
                                                    + a_send_buffer_size + a_recv_buffer_size);

    const double alpha = ctx->Attr<double>("alpha");
    double beta = 0.0;

    const int64_t cur_a_rank = kernel_state->parallel_id() % kernel_state->summa_dim1();
    const void* a_ptr = a->dptr();
    for (int64_t i = 0; i < lcm_summa_dim; ++i) {
      const int64_t a_data_id = i % num_a;
      const int64_t a_rank_id = i / num_a;
      if (num_a > 1 && cur_a_rank == a_rank_id) {
        NewKernelUtil<DeviceType::kGPU>::CopyColsRegion(
            ctx->device_ctx(), k, summa_m, a->dptr<T>(), a_data_id * summa_m, a_m,
            reinterpret_cast<T*>(a_send_buffer), 0, summa_m);
        a_ptr = a_send_buffer;
      }
      OF_NCCL_CHECK(ncclBroadcast(a_ptr, a_recv_buffer, a_send_elem_cnt,
                                  GetNcclDataType(a->data_type()), a_rank_id,
                                  kernel_state->row_comm(), ctx->device_ctx()->cuda_stream()));

      NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx->device_ctx(), CblasTrans, CblasNoTrans, summa_m,
                                              n, k, alpha, reinterpret_cast<T*>(a_recv_buffer),
                                              b->dptr<T>(), beta,
                                              reinterpret_cast<T*>(out_send_buffer));
      const int64_t c_data_id = i % num_c;
      const int64_t c_rank_id = i / num_c;
      void* out_ptr = reinterpret_cast<void*>(out->mut_dptr<T>() + c_data_id * c_send_elem_cnt);
      OF_NCCL_CHECK(ncclReduce(reinterpret_cast<T*>(out_send_buffer), out_ptr, c_send_elem_cnt,
                               GetNcclDataType(out->data_type()), ncclRedOp_t::ncclSum, c_rank_id,
                               kernel_state->col_comm(), ctx->device_ctx()->cuda_stream()));
    }
  };
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SUMMA_MATMUL_ATB_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("summa_matmul")                                                      \
      .SetCreateFn<SummaMatmulATBKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                        \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value)        \
                       & (user_op::HobAttr<bool>("transpose_a") == true)                    \
                       & (user_op::HobAttr<bool>("transpose_b") == false))                  \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const TensorDesc* a_desc = ctx->TensorDesc4ArgNameAndIndex("a", 0);                 \
        const TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);             \
        const int n = out_desc->shape().At(out_desc->shape().NumAxes() - 1);                \
        const int c_m = out_desc->shape().elem_cnt() / n;                                   \
        const int a_m = a_desc->shape().At(a_desc->shape().NumAxes() - 1);                  \
        const int summa_m = Gcd(c_m, a_m);                                                  \
        const int num_a = a_m / summa_m;                                                    \
        const int num_c = c_m / summa_m;                                                    \
        const int64_t num_a_buffer = (num_a > 1) ? 2 : 1;                                   \
        return num_a_buffer                                                                 \
                   * GetCudaAlignedSize(a_desc->shape().elem_cnt() / num_a * sizeof(dtype)) \
               + GetCudaAlignedSize(out_desc->shape().elem_cnt() / num_c * sizeof(dtype));  \
      });

#ifdef WITH_CUDA
REGISTER_SUMMA_MATMUL_ATB_KERNEL(float16);
REGISTER_SUMMA_MATMUL_ATB_KERNEL(float);
REGISTER_SUMMA_MATMUL_ATB_KERNEL(double);
#endif

#define REGISTER_BROADCAST_SUMMA_MATMUL_ATB_KERNEL(dtype)                                   \
  REGISTER_USER_KERNEL("summa_broadcast_matmul_grad_b")                                     \
      .SetCreateFn<SummaMatmulATBKernel<dtype>>()                                           \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)                        \
                       & (user_op::HobDataType("a", 0) == GetDataType<dtype>::value))       \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                   \
        const TensorDesc* a_desc = ctx->TensorDesc4ArgNameAndIndex("a", 0);                 \
        const TensorDesc* out_desc = ctx->TensorDesc4ArgNameAndIndex("out", 0);             \
        const int n = out_desc->shape().At(out_desc->shape().NumAxes() - 1);                \
        const int c_m = out_desc->shape().elem_cnt() / n;                                   \
        const int a_m = a_desc->shape().At(a_desc->shape().NumAxes() - 1);                  \
        const int summa_m = Gcd(c_m, a_m);                                                  \
        const int num_a = a_m / summa_m;                                                    \
        const int num_c = c_m / summa_m;                                                    \
        const int64_t num_a_buffer = (num_a > 1) ? 2 : 1;                                   \
        return num_a_buffer                                                                 \
                   * GetCudaAlignedSize(a_desc->shape().elem_cnt() / num_a * sizeof(dtype)) \
               + GetCudaAlignedSize(out_desc->shape().elem_cnt() / num_c * sizeof(dtype));  \
      });

#ifdef WITH_CUDA
REGISTER_BROADCAST_SUMMA_MATMUL_ATB_KERNEL(float16);
REGISTER_BROADCAST_SUMMA_MATMUL_ATB_KERNEL(float);
REGISTER_BROADCAST_SUMMA_MATMUL_ATB_KERNEL(double);
#endif

}  // namespace user_op
}  // namespace oneflow
