
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/ndarray/ndarray_util.h"
#include "oneflow/core/cuda/atomic.cuh"
#include <cub/cub.cuh>
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/ep/include/primitive/fill.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/cuda/apex_layernorm.cuh"

namespace oneflow {


template<typename T, typename U>
class LayerNormGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LayerNormGpuKernel() = default;
  ~LayerNormGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    user_op::Tensor* normalized =
        ctx->has_input("gamma", 0) ? ctx->Tensor4ArgNameAndIndex("normalized", 0) : y;
    const double epsilon = ctx->Attr<double>("epsilon");
    CHECK_GE(epsilon, CUDNN_BN_MIN_EPSILON);
    const int64_t num_instances = mean->shape().elem_cnt();
    const int64_t norm_size = x->shape().elem_cnt() / num_instances;
    const T* gamma_ptr = nullptr;
    const T* beta_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      const user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
      gamma_ptr = gamma->dptr<T>();
      CHECK_EQ(gamma->shape().elem_cnt(), norm_size);
    }
    if (ctx->has_input("beta", 0)) { beta_ptr = ctx->Tensor4ArgNameAndIndex("beta", 0)->dptr<T>(); }
    
    // int32_t n1 = num_instances; 
    // int32_t n2 = norm_size;
    
    int32_t n1 = num_instances; 
    int32_t n2 = norm_size;

    const dim3 threads(32,4,1);
    const uint64_t maxGridY = ctx->stream()->As<ep::CudaStream>()->device_properties().maxGridSize[1];
    
    // printf("N1 is: %d \n", n1); 
    // printf("N2 is: %d \n", n2); 

    // printf("Num instance is: %d \n", num_instances); 
    // printf("Normsize is: %d \n", norm_size); 

    const dim3 blocks(1, std::min((uint64_t)n1, maxGridY), 1);
    int nshared =
        threads.y > 1 ?
	    threads.y*sizeof(U)+(threads.y/2)*sizeof(U) :
	    0;
    cuda::apex_layer_norm::cuApplyLayerNorm<T, U, T><<<blocks, threads, nshared, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
        y->mut_dptr<T>(), mean->mut_dptr<T>(), inv_variance->mut_dptr<T>(), x->dptr<T>(), n1, n2, U(epsilon), gamma_ptr, beta_ptr);
  };
};

#define REGISTER_LAYER_NORM_CUDA_KERNEL(dtype, acc_dtype)                         \
  REGISTER_USER_KERNEL("layer_norm")                                   \
      .SetCreateFn<LayerNormGpuKernel<dtype, acc_dtype>>()                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value));

REGISTER_LAYER_NORM_CUDA_KERNEL(float, float)
// REGISTER_LAYER_NORM_CUDA_KERNEL(double)
// REGISTER_LAYER_NORM_CUDA_KERNEL(half)

template<typename T, typename U>
class LayerNormGradGpuKernel final : public user_op::OpKernel, public user_op::CudaGraphSupport {
 public:
  LayerNormGradGpuKernel() = default;
  ~LayerNormGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mean = ctx->Tensor4ArgNameAndIndex("mean", 0);
    const user_op::Tensor* inv_variance = ctx->Tensor4ArgNameAndIndex("inv_variance", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t num_instances = mean->shape().elem_cnt();
    const int64_t norm_size = x->shape().elem_cnt() / num_instances;
    const T* gamma_ptr = nullptr;
    if (ctx->has_input("gamma", 0)) {
      gamma_ptr = ctx->Tensor4ArgNameAndIndex("gamma", 0)->dptr<T>();
    }
    // const T* add_to_output_ptr = nullptr;
    // if (ctx->has_input("_add_to_output", 0)) {
    //   const user_op::Tensor* add_to_output = ctx->Tensor4ArgNameAndIndex("_add_to_output", 0);
    //   CHECK_EQ(add_to_output->data_type(), dx->data_type());
    //   CHECK_EQ(add_to_output->shape(), dx->shape());
    //   add_to_output_ptr = add_to_output->dptr<T>();
    // }
    
    // compute grad_input
    const double epsilon = 1e-5; // Not be used. 
    int32_t n1 = num_instances; 
    int32_t n2 = norm_size; 
    const uint64_t maxGridY = ctx->stream()->As<ep::CudaStream>()->device_properties().maxGridSize[1];
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
    const dim3 threads1(32,4,1);
    int nshared =
	    threads1.y > 1 ?
	    threads1.y*threads1.x*sizeof(U) :
	    0;
    cuda::apex_layer_norm::cuComputeGradInput<<<blocks1, threads1, nshared, ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
            dy->dptr<T>(),
            x->dptr<T>(),
            n1,n2,
            mean->dptr<T>(),
            inv_variance->dptr<T>(),
            U(epsilon),
            gamma_ptr,
            dx->mut_dptr<T>());

  };
};

#define REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(dtype, acc_type)                                        \
  REGISTER_USER_KERNEL("layer_norm_grad")                                                  \
      .SetCreateFn<LayerNormGradGpuKernel<dtype, acc_type>>()                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                     \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value))    \
      .SetInplaceProposalFn(                                                               \
          [](const user_op::InferContext& ctx,                                             \
             const user_op::AddInplaceArgPair& AddInplaceArgPairFn) -> Maybe<void> {       \
            if (ctx.has_input("_add_to_output", 0)) {                                      \
              OF_RETURN_IF_ERROR(AddInplaceArgPairFn("dx", 0, "_add_to_output", 0, true)); \
            }                                                                              \
            return Maybe<void>::Ok();                                                      \
          });

REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(float, float)
// REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(double)
// REGISTER_LAYER_NORM_GRAD_CUDA_KERNEL(half)

// template<typename T>
// class LayerNormParamGradGpuKernel final : public user_op::OpKernel,
//                                           public user_op::CudaGraphSupport {
//  public:
//   LayerNormParamGradGpuKernel() = default;
//   ~LayerNormParamGradGpuKernel() = default;

//  private:
//   using user_op::OpKernel::Compute;
//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
//   void Compute(user_op::KernelComputeContext* ctx) const override {
//     using NdUtil = NdarrayUtil<DeviceType::kCUDA, T>;
//     auto Val = NdUtil::GetValNdarrayBuilder();
//     auto Var = NdUtil::GetVarNdarrayBuilder();
//     const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
//     user_op::Tensor* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
//     user_op::Tensor* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
//     user_op::Tensor* normalized_diff = ctx->Tensor4ArgNameAndIndex("normalized_diff", 0);
//     user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
//     const bool has_beta_diff = beta_diff != nullptr;
//     const bool has_gamma_diff = gamma_diff != nullptr;
//     const bool has_normalized_diff = normalized_diff != nullptr;
//     const bool has_gamma = gamma != nullptr;
//     const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
//     const int64_t elem_cnt = dy->shape().elem_cnt();
//     const int64_t m = dy->shape().Count(begin_params_axis);
//     int max_active_blocks = 0;
//     OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//         &max_active_blocks, LayerNormParamGradImpl<T, int64_t>, GetLayerNormParamGradBlockSize(),
//         GetParamGradDynamicSharedMemorySize<T>(m)));
//     if (has_gamma_diff && has_beta_diff && has_normalized_diff && max_active_blocks > 0) {
//       const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
//       Memset<DeviceType::kCUDA>(ctx->stream(), gamma_diff->mut_dptr<T>(), 0,
//                                 gamma_diff->shape().elem_cnt() * sizeof(T));
//       Memset<DeviceType::kCUDA>(ctx->stream(), beta_diff->mut_dptr<T>(), 0,
//                                 beta_diff->shape().elem_cnt() * sizeof(T));
//       if (elem_cnt > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
//         LayerNormParamGradImpl<T, int64_t>
//             <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
//                GetParamGradDynamicSharedMemorySize<T>(m),
//                ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
//                 elem_cnt, m, dy->dptr<T>(), normalized->dptr<T>(), gamma->dptr<T>(),
//                 gamma_diff->mut_dptr<T>(), beta_diff->mut_dptr<T>(),
//                 normalized_diff->mut_dptr<T>());
//       } else {
//         LayerNormParamGradImpl<T, int32_t>
//             <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
//                GetParamGradDynamicSharedMemorySize<T>(m),
//                ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
//                 static_cast<int32_t>(elem_cnt), static_cast<int32_t>(m), dy->dptr<T>(),
//                 normalized->dptr<T>(), gamma->dptr<T>(), gamma_diff->mut_dptr<T>(),
//                 beta_diff->mut_dptr<T>(), normalized_diff->mut_dptr<T>());
//       }
//     } else {
//       if (has_beta_diff) {
//         user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
//         CHECK_EQ(m, beta_diff->shape().elem_cnt());
//         CHECK_EQ(dy->shape().elem_cnt() % m, 0);
//         const int64_t n = dy->shape().elem_cnt() / m;
//         NdUtil::ReduceSum(ctx->stream(), Var({1, m}, beta_diff->mut_dptr<T>()),
//                           Val({n, m}, dy->dptr<T>()), Var({n, m}, reduce_buf->mut_dptr<T>()));
//       }
//       if (has_gamma_diff) {
//         const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
//         user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
//         CHECK_EQ(m, gamma_diff->shape().elem_cnt());
//         CHECK_EQ(dy->shape().elem_cnt() % m, 0);
//         const int64_t n = dy->shape().elem_cnt() / m;
//         NdUtil::BroadcastMul(ctx->stream(), Var({n, m}, reduce_buf->mut_dptr<T>()),
//                              Val({n, m}, normalized->dptr<T>()), Val({n, m}, dy->dptr<T>()));
//         NdUtil::ReduceSum(ctx->stream(), Var({1, m}, gamma_diff->mut_dptr<T>()),
//                           Val({n, m}, reduce_buf->dptr<T>()),
//                           Var({n, m}, reduce_buf->mut_dptr<T>()));
//       }
//       if (has_normalized_diff) {
//         if (has_gamma) {
//           CHECK_EQ(m, gamma->shape().elem_cnt());
//           CHECK_EQ(dy->shape().elem_cnt() % m, 0);
//           const int64_t n = dy->shape().elem_cnt() / m;
//           NdUtil::BroadcastMul(ctx->stream(), Var({n, m}, normalized_diff->mut_dptr<T>()),
//                                Val({n, m}, dy->dptr<T>()), Val({1, m}, gamma->dptr<T>()));
//         } else {
//           Memcpy<DeviceType::kCUDA>(ctx->stream(), normalized_diff->mut_dptr<void>(),
//                                     dy->dptr<void>(),
//                                     dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
//         }
//       }
//     }
//   };
// };

// #define REGISTER_LAYER_NORM_PARAM_GRAD_CUDA_KERNEL(dtype)              \
//   REGISTER_USER_KERNEL("layer_norm_param_grad")                        \
//       .SetCreateFn<LayerNormParamGradGpuKernel<dtype>>()               \
//       .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
//                        && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

// REGISTER_LAYER_NORM_PARAM_GRAD_CUDA_KERNEL(float)
// REGISTER_LAYER_NORM_PARAM_GRAD_CUDA_KERNEL(double)

// class LayerNormParamGradGpuHalfKernel final : public user_op::OpKernel,
//                                               public user_op::CudaGraphSupport {
//  public:
//   LayerNormParamGradGpuHalfKernel() = default;
//   ~LayerNormParamGradGpuHalfKernel() = default;

//  private:
//   using user_op::OpKernel::Compute;
//   bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
//   void Compute(user_op::KernelComputeContext* ctx) const override {
//     using NdUtil = NdarrayUtil<DeviceType::kCUDA, float16>;
//     auto Val = NdUtil::GetValNdarrayBuilder();
//     auto Var = NdUtil::GetVarNdarrayBuilder();
//     const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
//     user_op::Tensor* beta_diff = ctx->Tensor4ArgNameAndIndex("beta_diff", 0);
//     user_op::Tensor* gamma_diff = ctx->Tensor4ArgNameAndIndex("gamma_diff", 0);
//     user_op::Tensor* normalized_diff = ctx->Tensor4ArgNameAndIndex("normalized_diff", 0);
//     user_op::Tensor* gamma = ctx->Tensor4ArgNameAndIndex("gamma", 0);
//     const bool has_beta_diff = beta_diff != nullptr;
//     const bool has_gamma_diff = gamma_diff != nullptr;
//     const bool has_normalized_diff = normalized_diff != nullptr;
//     const bool has_gamma = gamma != nullptr;
//     const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
//     const int64_t elem_cnt = dy->shape().elem_cnt();
//     const int64_t m = dy->shape().Count(begin_params_axis);
//     int max_active_blocks = 0;
//     OF_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
//         &max_active_blocks, LayerNormParamGradHalfImpl<int64_t>, GetLayerNormParamGradBlockSize(),
//         GetParamGradDynamicSharedMemorySize<float16>(m)));
//     if (has_gamma_diff && has_beta_diff && has_normalized_diff && max_active_blocks > 0) {
//       const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
//       user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
//       const int64_t num_blocks = GetLayerNormParamGradNumBlocks(dy->shape().elem_cnt());
//       const size_t tmp_diff_size = GetCudaAlignedSize(num_blocks * m * sizeof(float16));
//       float16* tmp_gamma_diff = tmp_buffer->mut_dptr<float16>();
//       float16* tmp_beta_diff =
//           reinterpret_cast<float16*>(tmp_buffer->mut_dptr<char>() + tmp_diff_size);
//       float16* tmp_reduce_buf =
//           reinterpret_cast<float16*>(tmp_buffer->mut_dptr<char>() + 2 * tmp_diff_size);
//       CHECK_GE(tmp_buffer->shape().elem_cnt(), 3 * tmp_diff_size);
//       if (elem_cnt > static_cast<int64_t>(GetMaxVal<int32_t>() / 2)) {
//         LayerNormParamGradHalfImpl<int64_t>
//             <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
//                GetParamGradDynamicSharedMemorySize<float16>(m),
//                ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
//                 elem_cnt, m, dy->dptr<half>(), normalized->dptr<half>(), gamma->dptr<half>(),
//                 reinterpret_cast<half*>(tmp_gamma_diff), reinterpret_cast<half*>(tmp_beta_diff),
//                 normalized_diff->mut_dptr<half>());
//       } else {
//         LayerNormParamGradHalfImpl<int32_t>
//             <<<GetLayerNormParamGradNumBlocks(elem_cnt), GetLayerNormParamGradBlockSize(),
//                GetParamGradDynamicSharedMemorySize<float16>(m),
//                ctx->stream()->As<ep::CudaStream>()->cuda_stream()>>>(
//                 static_cast<int32_t>(elem_cnt), static_cast<int32_t>(m), dy->dptr<half>(),
//                 normalized->dptr<half>(), gamma->dptr<half>(),
//                 reinterpret_cast<half*>(tmp_gamma_diff), reinterpret_cast<half*>(tmp_beta_diff),
//                 normalized_diff->mut_dptr<half>());
//       }
//       NdUtil::ReduceSum(ctx->stream(), Var({1, m}, gamma_diff->mut_dptr<float16>()),
//                         Val({num_blocks, m}, tmp_gamma_diff), Var({num_blocks, m}, tmp_reduce_buf));
//       NdUtil::ReduceSum(ctx->stream(), Var({1, m}, beta_diff->mut_dptr<float16>()),
//                         Val({num_blocks, m}, tmp_beta_diff), Var({num_blocks, m}, tmp_reduce_buf));
//     } else {
//       if (has_beta_diff) {
//         user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
//         CHECK_EQ(m, beta_diff->shape().elem_cnt());
//         CHECK_EQ(dy->shape().elem_cnt() % m, 0);
//         const int64_t n = dy->shape().elem_cnt() / m;
//         NdUtil::ReduceSum(ctx->stream(), Var({1, m}, beta_diff->mut_dptr<float16>()),
//                           Val({n, m}, dy->dptr<float16>()),
//                           Var({n, m}, reduce_buf->mut_dptr<float16>()));
//       }
//       if (has_gamma_diff) {
//         const user_op::Tensor* normalized = ctx->Tensor4ArgNameAndIndex("normalized", 0);
//         user_op::Tensor* reduce_buf = ctx->Tensor4ArgNameAndIndex("reduce_buf", 0);
//         CHECK_EQ(m, gamma_diff->shape().elem_cnt());
//         CHECK_EQ(dy->shape().elem_cnt() % m, 0);
//         const int64_t n = dy->shape().elem_cnt() / m;
//         NdUtil::BroadcastMul(ctx->stream(), Var({n, m}, reduce_buf->mut_dptr<float16>()),
//                              Val({n, m}, normalized->dptr<float16>()),
//                              Val({n, m}, dy->dptr<float16>()));
//         NdUtil::ReduceSum(ctx->stream(), Var({1, m}, gamma_diff->mut_dptr<float16>()),
//                           Val({n, m}, reduce_buf->dptr<float16>()),
//                           Var({n, m}, reduce_buf->mut_dptr<float16>()));
//       }
//       if (has_normalized_diff) {
//         if (has_gamma) {
//           CHECK_EQ(m, gamma->shape().elem_cnt());
//           CHECK_EQ(dy->shape().elem_cnt() % m, 0);
//           const int64_t n = dy->shape().elem_cnt() / m;
//           NdUtil::BroadcastMul(ctx->stream(), Var({n, m}, normalized_diff->mut_dptr<float16>()),
//                                Val({n, m}, dy->dptr<float16>()),
//                                Val({1, m}, gamma->dptr<float16>()));
//         } else {
//           Memcpy<DeviceType::kCUDA>(ctx->stream(), normalized_diff->mut_dptr<void>(),
//                                     dy->dptr<void>(),
//                                     dy->shape().elem_cnt() * GetSizeOfDataType(dy->data_type()));
//         }
//       }
//     }
//   }
// };

// REGISTER_USER_KERNEL("layer_norm_param_grad")
//     .SetCreateFn<LayerNormParamGradGpuHalfKernel>()
//     .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)
//                      && (user_op::HobDataType("dy", 0) == DataType::kFloat16))
//     .SetInferTmpSizeFn([](user_op::InferContext* ctx) {
//       const int64_t begin_params_axis = ctx->Attr<int64_t>("begin_params_axis");
//       const bool has_gamma_diff = ctx->has_output("gamma_diff", 0);
//       const bool has_beta_diff = ctx->has_output("beta_diff", 0);
//       const bool has_normalized_diff = ctx->has_output("normalized_diff", 0);
//       const auto& dy = ctx->InputTensorDesc("dy", 0);
//       const int64_t instance_size = dy.shape().Count(begin_params_axis);
//       size_t tmp_buffer_size = 0;
//       if (has_gamma_diff && has_beta_diff && has_normalized_diff) {
//         const size_t tmp_gamma_diff =
//             GetCudaAlignedSize(GetLayerNormParamGradNumBlocks(dy.shape().elem_cnt()) * instance_size
//                                * sizeof(float16));
//         const size_t tmp_beta_diff = tmp_gamma_diff;
//         const size_t tmp_reduce_buf = tmp_gamma_diff;
//         tmp_buffer_size = tmp_gamma_diff + tmp_beta_diff + tmp_reduce_buf;
//       } else {
//         tmp_buffer_size = 0;
//       }
//       return tmp_buffer_size;
//     });

}  // namespace oneflow
