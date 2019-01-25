#ifndef ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_

#include "oneflow/core/common/blas.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename U = void>
struct NewKernelUtilIf {
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob);
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim);
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y);
};
struct NewKernelUtilIf;

template<typename T, typename U = void>
struct CpuNewKernelUtilIf {};

template<typename T, typename U = void>
struct GpuNewKernelUtilIf {};

template<DeviceType device_type, typename T>
struct FloatingNewKernelUtilIf {
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c, const int ldc);
};

template<DeviceType device_type, typename T>
struct IntegralNewKernelUtilIf {};

template<DeviceType device_type, typename T>
struct Float16NewKernelUtilIf {
  static void HGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                    const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                    const int m, const int n, const int k, const T alpha, const T* a, const int lda,
                    const T* b, const int ldb, const T beta, T* c, const int ldc);
};

template<DeviceType device_type, typename T, typename U = void>
struct NewKernelUtil;

// CPU && Floating
template<typename T>
struct NewKernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public NewKernelUtilIf<DeviceType::kCPU, T>,
      public CpuNewKernelUtilIf<T>,
      public FloatingNewKernelUtilIf<DeviceType::kCPU, T> {};

// CPU && Integral
template<typename T>
struct NewKernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsIntegral<T>::value>::type>
    : public NewKernelUtilIf<DeviceType::kCPU, T>,
      public CpuNewKernelUtilIf<T>,
      public IntegralNewKernelUtilIf<DeviceType::kCPU, T> {};

// CPU && Float16
template<typename T>
struct NewKernelUtil<DeviceType::kCPU, T, typename std::enable_if<IsFloat16<T>::value>::type>
    : public NewKernelUtilIf<DeviceType::kCPU, T>,
      public CpuNewKernelUtilIf<T>,
      public Float16NewKernelUtilIf<DeviceType::kCPU, T> {};

// GPU && Floating
template<typename T>
struct NewKernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloating<T>::value>::type>
    : public NewKernelUtilIf<DeviceType::kGPU, T>,
      public GpuNewKernelUtilIf<T>,
      public FloatingNewKernelUtilIf<DeviceType::kGPU, T> {};

// GPU && Integral
template<typename T>
struct NewKernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsIntegral<T>::value>::type>
    : public NewKernelUtilIf<DeviceType::kGPU, T>,
      public GpuNewKernelUtilIf<T>,
      public IntegralNewKernelUtilIf<DeviceType::kGPU, T> {};

// GPU && Float16
template<typename T>
struct NewKernelUtil<DeviceType::kGPU, T, typename std::enable_if<IsFloat16<T>::value>::type>
    : public NewKernelUtilIf<DeviceType::kGPU, T>,
      public CpuNewKernelUtilIf<T>,
      public Float16NewKernelUtilIf<DeviceType::kGPU, T> {};

// Functions that do not fit other classes stay here
template<DeviceType device_type, typename T, typename U>
struct NewKernelUtilIf {
  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
    const int m = c->shape().At(0);
    const int n = c->shape().Count(1);
    const int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

    OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
           c->mut_dptr<T>());
  }
  static void OFGemmTrans(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                          enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                          const T alpha, const T* a, const T* b, const T beta, T* c) {
    trans_a = (trans_a == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    trans_b = (trans_b == CblasNoTrans) ? CblasTrans : CblasNoTrans;
    OFGemm(ctx, trans_b, trans_a, n, m, k, alpha, b, a, beta, c);
  }
  static void InitializeWithProperConf(DeviceCtx* ctx, const InitializerConf* initializer_conf,
                                       uint32_t random_seed, Blob* blob,
                                       const std::string& data_format = "") {
    if (initializer_conf == nullptr) {
      initializer_conf = Global<JobDesc>::Get()->DefaultInitializerConf();
    }
    InitializeWithConf(ctx, *initializer_conf, random_seed, blob, data_format);
  }
  static void InitializeWithProperConf(DeviceCtx* ctx, const PbMessage* initializer_conf,
                                       uint32_t random_seed, Blob* blob,
                                       const std::string& data_format = "") {
    InitializeWithProperConf(ctx, static_cast<const InitializerConf*>(initializer_conf),
                             random_seed, blob, data_format);
  }
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob) {
    InitializeWithConf(ctx, initializer_conf, random_seed, blob, "");
  }
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c);
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format);
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim);
  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx);
  static void TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx);
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y);
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx);
  static void Set(DeviceCtx* ctx, const T value, T* addr);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_UTIL_H_
