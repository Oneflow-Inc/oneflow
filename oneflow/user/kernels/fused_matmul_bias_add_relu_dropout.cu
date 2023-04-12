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
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/cuda/atomic.cuh"
#include "oneflow/user/kernels/cublas_fused_mlp_util.cuh"
#include "oneflow/user/kernels/dropout_kernel.h"
#include "oneflow/user/kernels/random_seed_util.h"

// CUBLAS_AUX_EPILOGUE only support in cuda11.4 or higher version, in cuda11.4 it need static link.
#if CUDA_VERSION >= 11060

namespace oneflow {

namespace {

constexpr int32_t kVecSize = 4;
constexpr int32_t kBlockSize = 256;
constexpr int32_t kWarpSize = 32;

union RandPack4 {
  uint4 storage;
  uint32_t elem[4];  // store curand4 return val.
};

template<int32_t pack_size, typename IndexType>
__device__ void SetCublasBitMask(const IndexType aux_ld, const IndexType row, const IndexType col,
                                 int32_t thread_bitmask, int32_t* mask) {
  IndexType linear_index = row * aux_ld + col;
  IndexType mask_index = linear_index / kWarpSize;
  IndexType mask_offset = linear_index - mask_index * kWarpSize;

  int32_t bitmask = thread_bitmask << mask_offset;
  for (int stride = kWarpSize / (pack_size * 2); stride > 0; stride /= 2) {
    bitmask |= __shfl_down_sync(__activemask(), bitmask, stride, kWarpSize);
  }
  if (mask_offset == 0) { mask[mask_index] = bitmask; }
}

template<typename T, bool relu, typename IndexType>
__global__ void FusedVectorizedReluDropoutKernel(uint64_t seed, uint64_t offset,
                                                 const IndexType elem_cnt, const int32_t aux_ld,
                                                 const IndexType cols, const uint32_t rate,
                                                 float scale, T* x, int32_t* mask) {
  IndexType global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, offset, &state);
  using LoadType = cuda::elementwise::PackType<T, kVecSize>;
  using LoadPack = cuda::elementwise::Pack<T, kVecSize>;

  T t_scale = static_cast<T>(scale);
  RandPack4 rand_uniform_pack4;
  T zero_val = static_cast<T>(0.0);
  for (IndexType linear_index = global_thread_id * kVecSize,
                 step = gridDim.x * blockDim.x * kVecSize;
       linear_index < elem_cnt; linear_index += step) {
    const IndexType row = linear_index / cols;
    const IndexType col = linear_index - row * cols;
    int32_t thread_bitmask = 0;

    rand_uniform_pack4.storage = curand4(&state);

    LoadType* x_load = reinterpret_cast<LoadType*>(x + linear_index);
    LoadPack x_vec;
    x_vec.storage = *x_load;
    LoadPack out_vec;
#pragma unroll
    for (int i = 0; i < kVecSize; i++) {
      bool relu_mask = true;
      if (relu) {
        // Relu
        relu_mask = x_vec.elem[i] >= zero_val;
      }
      // dropout
      bool mask_val = rand_uniform_pack4.elem[i] > rate;
      // Combined relu_mask, dropout_mask together.
      bool combined_mask = relu_mask && mask_val;
      // Cause half/bfloat16 cannot directily convert from bool, here we cast to float type first
      T t_combined_mask = static_cast<T>(static_cast<float>(combined_mask));
      thread_bitmask |= (combined_mask << i);
      out_vec.elem[i] = x_vec.elem[i] * t_combined_mask * t_scale;
    }
    *(reinterpret_cast<LoadType*>(x + linear_index)) = out_vec.storage;
    SetCublasBitMask<kVecSize, IndexType>(aux_ld, row, col, thread_bitmask, mask);
  }
}

template<typename T, bool relu, typename IndexType>
__global__ void FusedPaddedVectorizedReluDropoutKernel(uint64_t seed, uint64_t offset,
                                                       const IndexType aligned32_elem_cnt,
                                                       const int32_t aux_ld,
                                                       const IndexType aligned32_cols,
                                                       const IndexType cols, const uint32_t rate,
                                                       float scale, T* x, int32_t* mask) {
  IndexType global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, offset, &state);
  using LoadType = cuda::elementwise::PackType<T, kVecSize>;
  using LoadPack = cuda::elementwise::Pack<T, kVecSize>;

  T t_scale = static_cast<T>(scale);
  RandPack4 rand_uniform_pack4;
  T zero_val = static_cast<T>(0.0);
  for (IndexType linear_index = global_thread_id * kVecSize,
                 step = gridDim.x * blockDim.x * kVecSize;
       linear_index < aligned32_elem_cnt; linear_index += step) {
    const IndexType row = linear_index / aligned32_cols;
    const IndexType col = linear_index - row * aligned32_cols;
    int32_t thread_bitmask = 0;

    if (col < cols) {
      const IndexType actual_index = row * cols + col;
      rand_uniform_pack4.storage = curand4(&state);

      LoadType* x_load = reinterpret_cast<LoadType*>(x + actual_index);
      LoadPack x_vec;
      x_vec.storage = *x_load;
      LoadPack out_vec;
#pragma unroll
      for (int i = 0; i < kVecSize; i++) {
        bool relu_mask = true;
        if (relu) {
          // Relu
          relu_mask = x_vec.elem[i] >= zero_val;
        }
        // dropout
        bool mask_val = rand_uniform_pack4.elem[i] > rate;
        // Combined relu_mask, dropout_mask together.
        bool combined_mask = relu_mask && mask_val;
        // Cause half/bfloat16 cannot directily convert from bool, here we cast to float type first
        T t_combined_mask = static_cast<T>(static_cast<float>(combined_mask));
        thread_bitmask |= (combined_mask << i);
        out_vec.elem[i] = x_vec.elem[i] * t_combined_mask * t_scale;
      }
      *(reinterpret_cast<LoadType*>(x + actual_index)) = out_vec.storage;
    }
    SetCublasBitMask<kVecSize, IndexType>(aux_ld, row, col, thread_bitmask, mask);
  }
}

template<typename T, bool relu, typename IndexType>
__global__ void FusedWarpReluDropoutKernel(uint64_t seed, uint64_t offset, const IndexType elem_cnt,
                                           const IndexType aux_ld, const IndexType rows,
                                           const IndexType cols, const uint32_t rate, float scale,
                                           T* x, int32_t* mask) {
  const int32_t lane_id = threadIdx.x;
  const IndexType global_warp_id = blockIdx.x * blockDim.y + threadIdx.y;
  const IndexType step = gridDim.x * blockDim.y;
  const IndexType global_thread_id = global_warp_id * kWarpSize + lane_id;

  curandStatePhilox4_32_10_t state;
  curand_init(seed, global_thread_id, offset, &state);

  T t_scale = static_cast<T>(scale);
  T zero_val = static_cast<T>(0.0);
  RandPack4 rand_uniform_pack4;

  for (IndexType row = global_warp_id; row < rows; row += step) {
    for (IndexType col = lane_id; col < cols; col += kWarpSize * kVecSize) {
      const IndexType linear_index = row * cols + col;
      rand_uniform_pack4.storage = curand4(&state);
#pragma unroll
      for (int i = 0; i < kVecSize; i++) {
        int32_t thread_bitmask = 0;
        int32_t cur_col = col + i * kWarpSize;
        int32_t cur_linear_index = linear_index + i * kWarpSize;
        if (cur_col < cols) {
          T x_val = x[cur_linear_index];
          const uint32_t rand_uniform_val = rand_uniform_pack4.elem[i];
          bool relu_mask = true;
          if (relu) {
            // relu
            relu_mask = x_val >= zero_val;
          }
          // dropout
          bool mask_val = rand_uniform_val > rate;
          // Combined relu_mask, dropout_mask together.
          bool combined_mask = relu_mask && mask_val;
          thread_bitmask = combined_mask;
          // Cause half/bfloat16 cannot directily convert from bool, here we cast to float type
          // first
          T t_combined_mask = static_cast<T>(static_cast<float>(combined_mask));
          T out_val = x_val * t_combined_mask * t_scale;
          x[cur_linear_index] = out_val;
        }
        int32_t warp_mask = __ballot_sync(__activemask(), thread_bitmask);
        if (lane_id == 0) { mask[(row * aux_ld + cur_col) / kWarpSize] = warp_mask; }
      }
    }
  }
}

template<typename Func>
unsigned int ComputeGridSize(ep::Stream* stream, Func func, const int64_t elem_cnt,
                             const int32_t block_size) {
  auto* cuda_stream = stream->As<ep::CudaStream>();
  const int64_t pack_num = elem_cnt / kVecSize;
  const int32_t num_blocks = std::max<int64_t>(1, (pack_num + block_size - 1) / block_size);
  const int32_t multi_processor_count = cuda_stream->device_properties().multiProcessorCount;
  int max_active_blocks = 0;
  OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, func, block_size,
                                                              /*shared_memory*/ 0));
  return std::min(num_blocks, max_active_blocks * multi_processor_count);
}

uint64_t RoundUp(uint64_t x, uint64_t y) { return (x + y - 1) / y * y; }

template<typename T, bool relu>
cudaError_t LaunchFusedReluDropoutKernel(
    ep::CudaStream* stream, const std::shared_ptr<one::CUDAGeneratorImpl>& cuda_generator,
    const int64_t elem_cnt, const int32_t aux_ld, const int64_t rows, const int64_t cols,
    float rate, float scale, T* x, int32_t* mask) {
  uint64_t offset = 0;
  uint64_t seed = cuda_generator->current_seed();
  const uint32_t uint_rate = UINT_MAX * rate;
  unsigned int grid_size = 0;
  if (cols % 32 == 0) {
    // Launch Elementwise Vectorized Kernel.
    if (elem_cnt < GetMaxVal<int32_t>()) {
      grid_size = ComputeGridSize(stream, FusedVectorizedReluDropoutKernel<T, relu, int32_t>,
                                  elem_cnt, kBlockSize);
      uint64_t inc_offset = RoundUp((elem_cnt / (kBlockSize * grid_size)), kVecSize);
      offset = cuda_generator->get_philox_offset(inc_offset);
      FusedVectorizedReluDropoutKernel<T, relu, int32_t>
          <<<grid_size, kBlockSize, 0, stream->cuda_stream()>>>(seed, offset, elem_cnt, aux_ld,
                                                                cols, uint_rate, scale, x, mask);
    } else {
      grid_size = ComputeGridSize(stream, FusedVectorizedReluDropoutKernel<T, relu, int64_t>,
                                  elem_cnt, kBlockSize);
      uint64_t inc_offset = RoundUp((elem_cnt / (kBlockSize * grid_size)), kVecSize);
      offset = cuda_generator->get_philox_offset(inc_offset);
      FusedVectorizedReluDropoutKernel<T, relu, int64_t>
          <<<grid_size, kBlockSize, 0, stream->cuda_stream()>>>(seed, offset, elem_cnt, aux_ld,
                                                                cols, uint_rate, scale, x, mask);
    }
  } else {
    if (cols % 4 == 0) {
      // Padding cols to align kWarpSize.
      const int64_t align32_cols = (cols + kWarpSize - 1) / kWarpSize * kWarpSize;
      const int64_t align32_elem_cnt = rows * align32_cols;
      if (align32_elem_cnt < GetMaxVal<int32_t>()) {
        grid_size =
            ComputeGridSize(stream, FusedPaddedVectorizedReluDropoutKernel<T, relu, int32_t>,
                            align32_elem_cnt, kBlockSize);
        uint64_t inc_offset = RoundUp((elem_cnt / (kBlockSize * grid_size)), kVecSize);
        offset = cuda_generator->get_philox_offset(inc_offset);
        FusedPaddedVectorizedReluDropoutKernel<T, relu, int32_t>
            <<<grid_size, kBlockSize, 0, stream->cuda_stream()>>>(seed, offset, align32_elem_cnt,
                                                                  aux_ld, align32_cols, cols,
                                                                  uint_rate, scale, x, mask);
      } else {
        grid_size =
            ComputeGridSize(stream, FusedPaddedVectorizedReluDropoutKernel<T, relu, int64_t>,
                            align32_elem_cnt, kBlockSize);
        uint64_t inc_offset = RoundUp((elem_cnt / (kBlockSize * grid_size)), kVecSize);
        offset = cuda_generator->get_philox_offset(inc_offset);
        FusedPaddedVectorizedReluDropoutKernel<T, relu, int64_t>
            <<<grid_size, kBlockSize, 0, stream->cuda_stream()>>>(seed, offset, align32_elem_cnt,
                                                                  aux_ld, align32_cols, cols,
                                                                  uint_rate, scale, x, mask);
      }
    } else {
      // Process a row by using a warp.
      dim3 block_dim(kWarpSize, kBlockSize / kWarpSize);
      if (elem_cnt < GetMaxVal<int32_t>()) {
        grid_size = ComputeGridSize(stream, FusedWarpReluDropoutKernel<T, relu, int32_t>, elem_cnt,
                                    kBlockSize);
        uint64_t inc_offset = RoundUp((elem_cnt / (kBlockSize * grid_size)), kVecSize);
        offset = cuda_generator->get_philox_offset(inc_offset);
        FusedWarpReluDropoutKernel<T, relu, int32_t>
            <<<grid_size, block_dim, 0, stream->cuda_stream()>>>(
                seed, offset, elem_cnt, aux_ld, rows, cols, uint_rate, scale, x, mask);
      } else {
        grid_size = ComputeGridSize(stream, FusedWarpReluDropoutKernel<T, relu, int32_t>, elem_cnt,
                                    kBlockSize);
        uint64_t inc_offset = RoundUp((elem_cnt / (kBlockSize * grid_size)), kVecSize);
        offset = cuda_generator->get_philox_offset(inc_offset);
        FusedWarpReluDropoutKernel<T, relu, int64_t>
            <<<grid_size, block_dim, 0, stream->cuda_stream()>>>(
                seed, offset, elem_cnt, aux_ld, rows, cols, uint_rate, scale, x, mask);
      }
    }
  }
  return cudaPeekAtLastError();
}

template<typename T>
class FusedMatmulBiasAddReluDropoutKernel final : public user_op::OpKernel {
 public:
  FusedMatmulBiasAddReluDropoutKernel() = default;
  ~FusedMatmulBiasAddReluDropoutKernel() override = default;

  std::shared_ptr<user_op::OpKernelCache> InitOpKernelCache(
      user_op::KernelCacheContext* ctx) const override {
    return CreateCublasFusedMLPKernelCache();
  }

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    const auto& generator = CHECK_JUST(one::MakeGenerator(DeviceType::kCUDA));
    generator->set_current_seed(
        CHECK_JUST(GetOpKernelRandomSeedInCurrentRank(ctx, ctx->Attr<int64_t>("seed"))));
    return std::make_shared<FusedDropoutKernelState>(generator);
  }

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    /*
    Fused DenseActivation Layer. Assume we have two layers:
    A: (m, k)
    B: (n, k) need transpose
    C: (j, n) need transpose
    tmp: A matmul B(transpose), its shape is (m, n)
    out: tmp matmul C(transpose), its shape is (m, j)
    */
    const int32_t weight_size = ctx->input_size("weights");
    const int32_t bias_size = ctx->input_size("biases");
    CHECK_EQ(weight_size, bias_size) << "The number of weight and bias is not equal!. ";
    auto* cuda_stream = ctx->stream()->As<ep::CudaStream>();
    const auto* matmul_cache = CHECK_NOTNULL(dynamic_cast<const CublasFusedMLPKernelCache*>(cache));

    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool skip_final_activation = ctx->Attr<bool>("skip_final_activation");

    auto* fused_dropout_kernel_state = dynamic_cast<FusedDropoutKernelState*>(state);
    CHECK_NOTNULL(fused_dropout_kernel_state);
    const auto& generator = fused_dropout_kernel_state->generator();
    CHECK_NOTNULL(generator);
    const auto device_index = ctx->stream()->device()->device_index();
    std::shared_ptr<one::CUDAGeneratorImpl> cuda_generator =
        CHECK_JUST(generator->Get<one::CUDAGeneratorImpl>(device_index));
    const std::vector<float> dropout_rate_list = ctx->Attr<std::vector<float>>("dropout_rate_list");

    const DataType data_type = out->data_type();
    const cublasComputeType_t cublas_compute_dtype = GetComputeType(data_type);
    const cudaDataType_t cuda_data_type = GetCudaDataType(data_type);
    size_t cublas_m = 0, cublas_n = 0, cublas_k = 0;
    int64_t cublas_lda = 0, cublas_ldb = 0, cublas_ldc = 0;

    const double alpha = 1.0;
    const auto sp_alpha = GetCublasScalarParameter(alpha, cublas_compute_dtype);
    const double beta = 0.0;
    const auto sp_beta = GetCublasScalarParameter(beta, cublas_compute_dtype);

    // Currently only support 2D matmul.
    DimVector in_shape(2);
    x->shape_view().ToDimVector(&in_shape);
    DimVector weight_shape(2);

    const void* in_buf_ptr = x->dptr();
    size_t offset = 0;
    for (int idx = 0; idx < weight_size; idx++) {
      const user_op::Tensor* weight = ctx->Tensor4ArgNameAndIndex("weights", idx);
      const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("biases", idx);
      user_op::Tensor* cublas_aux = ctx->Tensor4ArgNameAndIndex("cublas_aux", idx);

      const int64_t batchsize = in_shape.at(0);
      const int64_t out_feature = weight->shape_view().At(0);
      weight->shape_view().ToDimVector(&weight_shape);
      size_t matmul_out_elem_cnt = batchsize * out_feature;

      InferMatmulCublasMNK(in_shape, weight_shape,
                           /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                           /*transpose_b=*/ep::primitive::BlasTransposeType::T, &cublas_m,
                           &cublas_n, &cublas_k, &cublas_lda, &cublas_ldb, &cublas_ldc);

      cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      void* matmul_out_ptr;

      float rate = dropout_rate_list.at(idx);
      float scale = 0.0;
      const int32_t aux_ld = AlignReluAuxLd(out_feature);
      if (rate < 1.0f) { scale = 1.0f / (1.0f - rate); }

      if (idx == weight_size - 1) {
        matmul_out_ptr = ctx->Tensor4ArgNameAndIndex("out", 0)->mut_dptr();
      } else {
        matmul_out_ptr = ctx->Tensor4ArgNameAndIndex("hidden", idx)->mut_dptr();
      }
      SetCublasAttr(matmul_cache, cublas_compute_dtype, cuda_data_type, /*need_aux=*/false,
                    /*transpose_a=*/ep::primitive::BlasTransposeType::N,
                    /*transpose_b=*/ep::primitive::BlasTransposeType::T, epilogue, bias->dptr(),
                    /*aux_ptr=*/nullptr, cublas_m, cublas_n, cublas_k, cublas_lda, cublas_ldb,
                    cublas_ldc);

      OF_CUBLAS_CHECK(cublasLtMatmul(
          cuda_stream->cublas_lt_handle(), matmul_cache->operation_desc, &sp_alpha, weight->dptr(),
          matmul_cache->cublas_a_desc, in_buf_ptr, matmul_cache->cublas_b_desc, &sp_beta,
          matmul_out_ptr, matmul_cache->cublas_c_desc, matmul_out_ptr, matmul_cache->cublas_c_desc,
          nullptr, cuda_stream->cublas_workspace(), cuda_stream->cublas_workspace_size(),
          cuda_stream->cuda_stream()));

      if (idx != weight_size - 1 || !skip_final_activation || rate != 0.0f) {
        OF_CUDA_CHECK(cudaMemsetAsync(cublas_aux->mut_dptr<int32_t>(), 0,
                                      cublas_aux->shape_view().elem_cnt() * sizeof(int32_t),
                                      cuda_stream->cuda_stream()));
      }

      if (idx != weight_size - 1 || !skip_final_activation) {
        // If it's not last layer or it's last layer but need relu.
        OF_CUDA_CHECK((LaunchFusedReluDropoutKernel<T, true>(
            cuda_stream, cuda_generator, matmul_out_elem_cnt, aux_ld, batchsize, out_feature, rate,
            scale, reinterpret_cast<T*>(matmul_out_ptr),
            reinterpret_cast<int32_t*>(cublas_aux->mut_dptr()))));
        // Set relu_droput_out ptr as next layer's input.
        in_buf_ptr = matmul_out_ptr;
        // Set hidden_layer shape as next layer's input shape.
        in_shape.at(1) = out_feature;
      } else {
        if (rate == 0.0f) {
          // It's last layer and dropout_rate is 0.0f, we do not launch FusedReluDropoutKernel.
          break;
        } else {
          // skip_final_activation but need dropout.
          OF_CUDA_CHECK((LaunchFusedReluDropoutKernel<T, false>(
              cuda_stream, cuda_generator, matmul_out_elem_cnt, aux_ld, batchsize, out_feature,
              rate, scale, reinterpret_cast<T*>(matmul_out_ptr),
              reinterpret_cast<int32_t*>(cublas_aux->mut_dptr()))));
        }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(cpp_type, data_type) \
  REGISTER_USER_KERNEL("fused_matmul_bias_add_relu_dropout")                        \
      .SetCreateFn<FusedMatmulBiasAddReluDropoutKernel<cpp_type>>()                 \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)              \
                       && (user_op::HobDataType("out", 0) == data_type));

REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(float, DataType::kFloat)
REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(half, DataType::kFloat16)
#if CUDA_VERSION >= 11000
REGISTER_FUSED_MATMUL_BIAS_ADD_RELU_DROPOUT_KERNEL_GPU(nv_bfloat16, DataType::kBFloat16)
#endif

}  // namespace

}  // namespace oneflow

#endif  // CUDA_VERSION >= 11060
