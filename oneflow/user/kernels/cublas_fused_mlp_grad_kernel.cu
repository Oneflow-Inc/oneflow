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
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/user/kernels/cublas_fused_mlp_util.cuh"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/device/nccl_util.h"
#include "oneflow/core/job/eager_nccl_comm_manager.h"
// CUBLAS_AUX_EPILOGUE only support in cuda11.4 or higher version, in cuda11.4 it need static link.
#if CUDA_VERSION >= 11060

namespace oneflow {

namespace {

struct Comm {
  Comm(ncclComm_t comm) : comm(comm) {}
  ncclComm_t comm;
};

class MatmulGradKernelState final : public user_op::OpKernelState {
 public:
  MatmulGradKernelState(user_op::KernelInitContext* ctx)
      : if_need_comm_(false), stream_name_(EagerNcclCommMgr::kDefaultStreamName) {
    OF_CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    OF_CUDA_CHECK(cudaStreamCreate(&allreduce_stream_));
    OF_CUBLAS_CHECK(cublasLtCreate(&cublas_lt_handle_));
    workspace_size_ =
        ParseIntegerFromEnv("ONEFLOW_EP_CUDA_CUBLAS_WORKSPACE_SIZE_MB", kDefaultWorkspaceSizeMb)
        * 1024 * 1024;
    OF_CUDA_CHECK(cudaMalloc(&workspace_, workspace_size_));
    if (ctx->parallel_ctx().parallel_num() > 1) {
      parallel_conf_ = ctx->parallel_desc().parallel_conf();
    }
  }
  ~MatmulGradKernelState() {
    OF_CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
    OF_CUBLAS_CHECK(cublasLtDestroy(cublas_lt_handle_));
    OF_CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
    OF_CUDA_CHECK(cudaStreamSynchronize(allreduce_stream_));
    OF_CUDA_CHECK(cudaStreamDestroy(allreduce_stream_));
    OF_CUDA_CHECK(cudaFree(workspace_));
  }
  cudaStream_t grad_cuda_stream() const { return cuda_stream_; }
  cudaStream_t allreduce_stream() const { return allreduce_stream_; }
  cublasLtHandle_t cublas_lt_handle() const { return cublas_lt_handle_; }
  size_t cublas_workspace_size() const { return workspace_size_; }
  void* cublas_workspace() const { return workspace_; }

  bool IfCommCreate() const {
    if (!comm_) { return false; }
    return true;
  }

  bool IfNeedComm() const { return if_need_comm_; }

  ncclComm_t comm() { return GetOrCreate().comm; }

  const Comm& GetOrCreate() {
    if (!comm_) { InitCommMgr(); }
    return *comm_;
  }

  void InitNeedComm(user_op::KernelInitContext* ctx) {
    if_need_comm_ = false;
    if (ctx->parallel_ctx().parallel_num() > 1) {
      const int64_t d_weights_size = ctx->output_size("d_weights");
      if (ctx->SbpParallel4ArgNameAndIndex("d_weights", 0).has_broadcast_parallel()) {
        for (int i = 0; i < d_weights_size; i++) {
          CHECK(ctx->SbpParallel4ArgNameAndIndex("d_weights", i).has_broadcast_parallel())
              << "All d_weight's SBP should be Broadcast. ";
          CHECK(ctx->SbpParallel4ArgNameAndIndex("d_biases", i).has_broadcast_parallel())
              << "All d_bias's SBP should be Broadcast. ";
        }
        if (ctx->SbpParallel4ArgNameAndIndex("dy", 0).has_split_parallel()) {
          if_need_comm_ = true;
        }
      }
    }
  }

  void InitCommMgr() {
    std::set<std::pair<int64_t, int64_t>> device_set;
    const ParallelDesc parallel_desc(parallel_conf_);
    for (int64_t parallel_id = 0; parallel_id < parallel_desc.parallel_num(); ++parallel_id) {
      int64_t machine_id = CHECK_JUST(parallel_desc.MachineId4ParallelId(parallel_id));
      int64_t device_id = CHECK_JUST(parallel_desc.DeviceId4ParallelId(parallel_id));
      device_set.emplace(std::make_pair(machine_id, device_id));
    }
    EagerNcclCommMgr* comm_mgr = CHECK_NOTNULL(Singleton<EagerNcclCommMgr>::Get());
    ncclComm_t comm;
    comm = comm_mgr->GetCommForDeviceAndStreamName(device_set, stream_name_);
    comm_.reset(new Comm(comm));
  }

 private:
  cudaStream_t cuda_stream_{};
  cudaStream_t allreduce_stream_{};
  cublasLtHandle_t cublas_lt_handle_{};
  void* workspace_{};
  size_t workspace_size_;
  std::string stream_name_;
  std::unique_ptr<Comm> comm_;
  bool if_need_comm_;
  ParallelConf parallel_conf_;
};

template<typename T>
class CublasFusedMLPGradKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  CublasFusedMLPGradKernel() {
    OF_CUDA_CHECK(cudaEventCreate(&main_stream_event_));
    OF_CUDA_CHECK(cudaEventCreate(&async_weight_grad_event_));
    OF_CUDA_CHECK(cudaEventCreate(&dweight_event_));
    OF_CUDA_CHECK(cudaEventCreate(&allreduce_event_));
  };
  ~CublasFusedMLPGradKernel() override {
    OF_CUDA_CHECK(cudaEventDestroy(main_stream_event_));
    OF_CUDA_CHECK(cudaEventDestroy(async_weight_grad_event_));
    OF_CUDA_CHECK(cudaEventDestroy(dweight_event_));
    OF_CUDA_CHECK(cudaEventDestroy(allreduce_event_));
  };

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    std::shared_ptr<MatmulGradKernelState> kernel_state =
        std::make_shared<MatmulGradKernelState>(ctx);
    kernel_state->InitNeedComm(ctx);
    return kernel_state;
  }

 private:
  cudaEvent_t main_stream_event_;
  cudaEvent_t async_weight_grad_event_;
  cudaEvent_t dweight_event_;
  cudaEvent_t allreduce_event_;

  bool IsReadyForCapture(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
                         const user_op::OpKernelCache* cache) const override {
    auto* kernel_state = dynamic_cast<MatmulGradKernelState*>(state);
    if (kernel_state->IfNeedComm()) {
      return kernel_state->IfCommCreate();
    } else {
      return true;
    }
  }

  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    int64_t tmp_buf_elem_cnt = tmp_buffer->shape_view().elem_cnt();
    const int64_t weight_num = ctx->input_size("weights");
    user_op::Tensor* d_x = ctx->Tensor4ArgNameAndIndex("d_x", 0);
    const std::vector<float> alpha_list = ctx->Attr<std::vector<float>>("alpha_list");

    auto* kernel_state = dynamic_cast<MatmulGradKernelState*>(state);
    const auto* matmul_grad_cache =
        CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));

    ncclComm_t comm{};
    bool if_need_comm = kernel_state->IfNeedComm();

    if (if_need_comm) { comm = kernel_state->comm(); }

    void* dy_tmp_buf = tmp_buffer->mut_dptr();
    size_t tmp_buf_offset = 0;
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();

    const DataType data_type = dy->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha_one = 1.0;
    auto sp_alpha_one = GetCublasScalarParameter(alpha_one, cublas_compute_dtype);
    double alpha = 1.0;
    auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    double beta = 0.0;
    auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

    // currently only support 2D matmul.
    DimVector weight_shape(2);
    DimVector hidden_shape(2);
    DimVector dy_shape(2);
    dy->shape_view().ToDimVector(&dy_shape);
    const void* dgrad_buf = dy->dptr();

    const int64_t batch_size = dy->shape_view().At(0);
    const void* ones = nullptr;
    ep::CudaDevice* cuda_device = dynamic_cast<ep::CudaDevice*>(ctx->stream()->device());
    CHECK_NOTNULL(cuda_device);
    ones = cuda_device->GetConstOnes(dy->data_type(), batch_size);
    if (ones == nullptr) {
      std::unique_ptr<ep::primitive::Fill> fill =
          ep::primitive::NewPrimitive<ep::primitive::FillFactory>(ctx->stream()->device_type(),
                                                                  data_type);
      CHECK(fill);
      fill->Launch(ctx->stream(), tmp_buffer->mut_dptr(), 1.0, batch_size);
      ones = tmp_buffer->mut_dptr();
      tmp_buf_offset += GetCudaAlignedSize(batch_size * sizeof(T));
      dy_tmp_buf = reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + tmp_buf_offset);
    }

    for (int idx = weight_num - 1; idx >= 0; idx--) {
      const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weights", idx);
      weight->shape_view().ToDimVector(&weight_shape);
      InferMatmulCublasMNK(dy_shape, weight_shape,
                           /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                           /*transpose_b=*/ep::primitive::BlasTransposeType::N, &cublas_m,
                           &cublas_n, &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);
      if (idx != 0) {
        alpha = alpha_list.at(idx - 1);
        sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
        const user_op::Tensor* aux = ctx->Tensor4ArgNameAndIndex("cublas_aux", idx - 1);
        user_op::Tensor* d_bias = ctx->Tensor4ArgNameAndIndex("d_biases", idx - 1);
        epilogue = CUBLASLT_EPILOGUE_DRELU_BGRAD;
        SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/true,
                      /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                      /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue,
                      d_bias->mut_dptr(), aux->dptr(), cublas_m, cublas_n, cublas_k, cublas_lda,
                      cublas_ldb, cublas_ldc);
        /*
        a = dy, b = weight
        cublas_a=weight, cublas_b=dy
        */
        OF_CUDA_CHECK(cudaEventRecord(main_stream_event_, cuda_stream->cuda_stream()));
        OF_CUBLAS_CHECK(cublasLtMatmul(
            cuda_stream->cublas_lt_handle(), matmul_grad_cache->operation_desc, &sp_alpha,
            weight->dptr(), matmul_grad_cache->cublas_a_desc, dgrad_buf,
            matmul_grad_cache->cublas_b_desc, &sp_beta, dy_tmp_buf,
            matmul_grad_cache->cublas_c_desc, dy_tmp_buf, matmul_grad_cache->cublas_c_desc, nullptr,
            cuda_stream->cublas_workspace(), cuda_stream->cublas_workspace_size(),
            cuda_stream->cuda_stream()));
      } else {
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/false,
                      /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                      /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue, nullptr,
                      nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);
        /*
        a = dy, b = weight
        cublas_a=weight, cublas_b=dy
        */
        OF_CUDA_CHECK(cudaEventRecord(main_stream_event_, cuda_stream->cuda_stream()));
        OF_CUBLAS_CHECK(cublasLtMatmul(
            cuda_stream->cublas_lt_handle(), matmul_grad_cache->operation_desc, &sp_alpha_one,
            weight->dptr(), matmul_grad_cache->cublas_a_desc, dgrad_buf,
            matmul_grad_cache->cublas_b_desc, &sp_beta, d_x->mut_dptr(),
            matmul_grad_cache->cublas_c_desc, d_x->mut_dptr(), matmul_grad_cache->cublas_c_desc,
            nullptr, cuda_stream->cublas_workspace(), cuda_stream->cublas_workspace_size(),
            cuda_stream->cuda_stream()));
      }

      // step1: Get last layer's dbias.
      if (idx == weight_num - 1) {
        user_op::Tensor* d_last_bias = ctx->Tensor4ArgNameAndIndex("d_biases", weight_num - 1);
        DimVector ones_buf_shape(2);
        ones_buf_shape.at(0) = 1;
        ones_buf_shape.at(1) = batch_size;
        epilogue = CUBLASLT_EPILOGUE_DEFAULT;
        InferMatmulCublasMNK(ones_buf_shape, dy_shape,
                             /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                             /*transpose_b=*/ep::primitive::BlasTransposeType::N, &cublas_m,
                             &cublas_n, &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);
        SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/false,
                      /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                      /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue, nullptr,
                      nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);
        OF_CUDA_CHECK(cudaStreamWaitEvent(kernel_state->grad_cuda_stream(), main_stream_event_));
        OF_CUBLAS_CHECK(cublasLtMatmul(
            kernel_state->cublas_lt_handle(), matmul_grad_cache->operation_desc, &sp_alpha_one,
            dgrad_buf, matmul_grad_cache->cublas_a_desc, ones, matmul_grad_cache->cublas_b_desc,
            &sp_beta, d_last_bias->mut_dptr(), matmul_grad_cache->cublas_c_desc,
            d_last_bias->mut_dptr(), matmul_grad_cache->cublas_c_desc, nullptr,
            kernel_state->cublas_workspace(), kernel_state->cublas_workspace_size(),
            kernel_state->grad_cuda_stream()));
      }

      user_op::Tensor* d_weight = ctx->Tensor4ArgNameAndIndex("d_weights", idx);
      epilogue = CUBLASLT_EPILOGUE_DEFAULT;
      if (idx != 0) {
        const user_op::Tensor* hidden = ctx->Tensor4ArgNameAndIndex("hidden", idx - 1);  // here
        hidden->shape_view().ToDimVector(&hidden_shape);
        InferMatmulCublasMNK(dy_shape, hidden_shape,
                             /*transpose_a=*/ep::primitive::BlasTransposeType::T,
                             /*transpose_b=*/ep::primitive::BlasTransposeType::N, &cublas_m,
                             &cublas_n, &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

        SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/false,
                      /*transpose_a=*/ep::primitive::BlasTransposeType::T,
                      /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue, nullptr,
                      nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);
        if (idx != weight_num - 1) {
          // if idx == weight_num - 1, async_stream has wait main_stream_event_ in d_bias.
          OF_CUDA_CHECK(cudaStreamWaitEvent(kernel_state->grad_cuda_stream(), main_stream_event_));
        }
        OF_CUBLAS_CHECK(cublasLtMatmul(
            kernel_state->cublas_lt_handle(), matmul_grad_cache->operation_desc, &sp_alpha_one,
            hidden->dptr(), matmul_grad_cache->cublas_a_desc, dgrad_buf,
            matmul_grad_cache->cublas_b_desc, &sp_beta, d_weight->mut_dptr(),
            matmul_grad_cache->cublas_c_desc, d_weight->mut_dptr(),
            matmul_grad_cache->cublas_c_desc, nullptr, kernel_state->cublas_workspace(),
            kernel_state->cublas_workspace_size(), kernel_state->grad_cuda_stream()));
        OF_CUDA_CHECK(cudaEventRecord(dweight_event_, kernel_state->grad_cuda_stream()));
        // compute dy shape
        dy_shape.at(1) = weight_shape.at(1);
        // compute dybuf
        dgrad_buf = dy_tmp_buf;
        tmp_buf_offset += GetCudaAlignedSize(dy_shape.at(0) * dy_shape.at(1) * sizeof(T));
        CHECK_LE(tmp_buf_offset, tmp_buf_elem_cnt)
            << "Tmp buffer offset should <= Tmp buffer elem_cnt. ";
        dy_tmp_buf = reinterpret_cast<void*>(tmp_buffer->mut_dptr<char>() + tmp_buf_offset);
      } else {
        x->shape_view().ToDimVector(&hidden_shape);
        InferMatmulCublasMNK(dy_shape, hidden_shape,
                             /*transpose_a=*/ep::primitive::BlasTransposeType::T,
                             /*transpose_b=*/ep::primitive::BlasTransposeType::N, &cublas_m,
                             &cublas_n, &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);
        SetCublasAttr(matmul_grad_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/false,
                      /*transpose_a=*/ep::primitive::BlasTransposeType::T,
                      /*transpose_b=*/ep::primitive::BlasTransposeType::N, epilogue, nullptr,
                      nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb, cublas_ldc);
        OF_CUDA_CHECK(cudaStreamWaitEvent(kernel_state->grad_cuda_stream(), main_stream_event_));
        OF_CUBLAS_CHECK(cublasLtMatmul(
            kernel_state->cublas_lt_handle(), matmul_grad_cache->operation_desc, &sp_alpha_one,
            x->dptr(), matmul_grad_cache->cublas_a_desc, dgrad_buf,
            matmul_grad_cache->cublas_b_desc, &sp_beta, d_weight->mut_dptr(),
            matmul_grad_cache->cublas_c_desc, d_weight->mut_dptr(),
            matmul_grad_cache->cublas_c_desc, nullptr, kernel_state->cublas_workspace(),
            kernel_state->cublas_workspace_size(), kernel_state->grad_cuda_stream()));
        OF_CUDA_CHECK(cudaEventRecord(dweight_event_, kernel_state->grad_cuda_stream()));
      }

      if (if_need_comm) {
        // Do Allreduce for d_bias and d_weight.
        // Here we wait wgrad event, and set a ncclGroup to Allreduce d_bias and d_weight.
        OF_CUDA_CHECK(cudaStreamWaitEvent(kernel_state->allreduce_stream(), dweight_event_));
        OF_NCCL_CHECK(ncclGroupStart());
        user_op::Tensor* allreduce_d_bias = ctx->Tensor4ArgNameAndIndex("d_biases", idx);
        OF_NCCL_CHECK(ncclAllReduce(allreduce_d_bias->mut_dptr(), allreduce_d_bias->mut_dptr(),
                                    allreduce_d_bias->shape_view().elem_cnt(),
                                    GetNcclDataType(allreduce_d_bias->data_type()),
                                    ncclRedOp_t::ncclSum, comm, kernel_state->allreduce_stream()));
        OF_NCCL_CHECK(ncclAllReduce(d_weight->mut_dptr(), d_weight->mut_dptr(),
                                    d_weight->shape_view().elem_cnt(),
                                    GetNcclDataType(d_weight->data_type()), ncclRedOp_t::ncclSum,
                                    comm, kernel_state->allreduce_stream()));
        OF_NCCL_CHECK(ncclGroupEnd());
        if (idx == 0) {
          // We should sync allreduce before the kernel finish.
          OF_CUDA_CHECK(cudaEventRecord(allreduce_event_, kernel_state->allreduce_stream()));
        }
      }
    }

    if (if_need_comm) {
      OF_CUDA_CHECK(cudaStreamWaitEvent(cuda_stream->cuda_stream(), allreduce_event_));
    } else {
      OF_CUDA_CHECK(cudaStreamWaitEvent(cuda_stream->cuda_stream(), dweight_event_));
    }
  };

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CUBLAS_FUSED_MLP_GRAD_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("cublas_fused_mlp_grad")                                                  \
      .SetCreateFn<CublasFusedMLPGradKernel<dtype>>()                                            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                           \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value))           \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                                        \
        const int64_t weight_num = ctx->input_size("weights");                                   \
        const Shape& dy_shape = ctx->InputShape("dy", 0);                                        \
        int64_t m = dy_shape.At(0);                                                              \
        int64_t k = dy_shape.At(1);                                                              \
        int64_t tmp_buffer_size = 0;                                                             \
        tmp_buffer_size += GetCudaAlignedSize(m * sizeof(dtype)); /*For last layer's bias grad*/ \
        for (int idx = weight_num - 1; idx > 0; idx--) {                                         \
          const Shape& weight_shape = ctx->InputShape("weights", idx);                           \
          k = weight_shape.At(1);                                                                \
          tmp_buffer_size += GetCudaAlignedSize(m * k * sizeof(dtype));                          \
        }                                                                                        \
        return tmp_buffer_size;                                                                  \
      });

REGISTER_CUBLAS_FUSED_MLP_GRAD_KERNEL(float)
REGISTER_CUBLAS_FUSED_MLP_GRAD_KERNEL(double)
REGISTER_CUBLAS_FUSED_MLP_GRAD_KERNEL(half)

REGISTER_USER_KERNEL_UNIFIED_NCCL_COMM_INIT("cublas_fused_mlp_grad");

}  // namespace

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11060
