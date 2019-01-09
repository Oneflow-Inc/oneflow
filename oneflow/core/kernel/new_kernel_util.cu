#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
void GpuInitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                           uint32_t random_seed, Blob* blob, const std::string& data_format) {
  BEFORE_CPU_INITIALIZE();
  // synchronous initialize the host blob
  NewKernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, initializer_conf, random_seed,
                                                         host_blob.get(), data_format);
  AFTER_CPU_INITIALIZE();
}

template<typename T>
void GpuInitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                          const std::string& model_dir, Blob* blob, const std::string& bn_in_op,
                          int32_t dim_num, int64_t num_in_each_dim) {
  BEFORE_CPU_INITIALIZE();
  NewKernelUtil<DeviceType::kCPU, T>::InitializeWithDir(
      ctx, part_id, part_num, model_dir, host_blob.get(), bn_in_op, dim_num, num_in_each_dim);
  AFTER_CPU_INITIALIZE();
}

}  // namespace

#define NEW_KU_IF_GPU_METHOD(type_category) \
  template<typename T>                      \
  void NewKernelUtilIf<DeviceType::kGPU, T, \
                       typename std::enable_if<type_category<T>::value>::type>::

// GPU && Floating
NEW_KU_IF_GPU_METHOD(IsFloating)
InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                   Blob* blob, const std::string& data_format) {
  GpuInitializeWithConf<T>(ctx, initializer_conf, random_seed, blob, data_format);
}
NEW_KU_IF_GPU_METHOD(IsFloating)
InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                   Blob* blob) {
  InitializeWithConf(ctx, initializer_conf, random_seed, blob, "");
}
NEW_KU_IF_GPU_METHOD(IsFloating)
InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_dir,
                  Blob* blob, const std::string& bn_in_op, int32_t dim_num,
                  int64_t num_in_each_dim) {
  GpuInitializeWithDir<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                          num_in_each_dim);
}

// GPU && Integral
NEW_KU_IF_GPU_METHOD(IsIntegral)
InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                   Blob* blob, const std::string& data_format) {
  GpuInitializeWithConf<T>(ctx, initializer_conf, random_seed, blob, data_format);
}
NEW_KU_IF_GPU_METHOD(IsIntegral)
InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                   Blob* blob) {
  InitializeWithConf(ctx, initializer_conf, random_seed, blob, "");
}
NEW_KU_IF_GPU_METHOD(IsIntegral)
InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_dir,
                  Blob* blob, const std::string& bn_in_op, int32_t dim_num,
                  int64_t num_in_each_dim) {
  GpuInitializeWithDir<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                          num_in_each_dim);
}

#undef NEW_KU_IF_GPU_METHOD

#define FLOATING_NEW_KU_IF_GPU_METHOD \
  template<typename T>                \
  void FloatingNewKernelUtilIf<DeviceType::kGPU, T>::

FLOATING_NEW_KU_IF_GPU_METHOD
Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
     const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const T alpha,
     const T* a, const int lda, const T* b, const int ldb, const T beta, T* c, const int ldc) {
  // TODO: wrong CUBLAS_OP_N
  cublasOperation_t cublas_trans_a = cublasOperation_t::CUBLAS_OP_N;
  cublasOperation_t cublas_trans_b = cublasOperation_t::CUBLAS_OP_N;
  cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha, b, ldb,
                 a, lda, &beta, c, ldc);
}

#undef FLOATING_NEW_KU_IF_GPU_METHOD

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

#define INSTANTIATE_FLOATING_KERNEL_UTIL(type_cpp, type_proto) \
  template struct FloatingNewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_FLOATING_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
