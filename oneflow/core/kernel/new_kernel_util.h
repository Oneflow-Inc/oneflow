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

namespace oneflow {

template<DeviceType>
class NewKernelUtil;

template<>
class NewKernelUtil<DeviceType::kCPU> {
 public:
  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       float alpha, float beta, const Blob* a, const Blob* b, Blob* c);
  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       double alpha, double beta, const Blob* a, const Blob* b, Blob* c) ;
  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       float16 alpha, float16 beta, const Blob* a, const Blob* b, Blob* c);
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float alpha, const float* a, const float* b,
                     const float beta, float* c);
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const double alpha, const double* a, const double* b,
                     const double beta, double* c);
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
                     const float16 beta, float16* c);
};

template<>
class NewKernelUtil<DeviceType::kGPU> {
 public:
  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       float alpha, float beta, const Blob* a, const Blob* b, Blob* c);
  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       double alpha, double beta, const Blob* a, const Blob* b, Blob* c) ;
  static void BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       float16 alpha, float16 beta, const Blob* a, const Blob* b, Blob* c);
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float alpha, const float* a, const float* b,
                     const float beta, float* c);
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const double alpha, const double* a, const double* b,
                     const double beta, double* c);
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
                     const float16 beta, float16* c);
};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
