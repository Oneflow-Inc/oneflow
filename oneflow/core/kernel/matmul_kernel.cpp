#include "oneflow/core/kernel/matmul_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
struct Dim2MatMulUtil {
  static void Calc(const JobDesc& job_desc, DeviceCtx* ctx, const Blob* a,
                   CBLAS_TRANSPOSE blas_trans_a, const Blob* b, CBLAS_TRANSPOSE blas_trans_b,
                   Blob* c) {
    NewKernelUtil<device_type>::BlobGemm(ctx, blas_trans_a, blas_trans_b, GetOneVal<T>(),
                                         GetZeroVal<T>(), a, b, c);
  }
};  // namespace oneflow

template<>
struct Dim2MatMulUtil<DeviceType::kGPU, float16> {
  static void Calc(const JobDesc& job_desc, DeviceCtx* ctx, const Blob* a,
                   CBLAS_TRANSPOSE blas_trans_a, const Blob* b, CBLAS_TRANSPOSE blas_trans_b,
                   Blob* c) {
    if (job_desc.enable_float_compute_for_half_gemm()) {
      NewKernelUtil<DeviceType::kGPU>::BlobHGemmWithFloat(
          ctx, blas_trans_a, blas_trans_b, GetOneVal<float>(), GetZeroVal<float>(), a, b, c);
    } else {
      NewKernelUtil<DeviceType::kGPU>::BlobGemm(
          ctx, blas_trans_a, blas_trans_b, GetOneVal<float16>(), GetZeroVal<float16>(), a, b, c);
    }
  }
};

template<DeviceType device_type, typename T>
struct BatchMatMulUtil {
  static void Calc(const JobDesc& job_desc, DeviceCtx* ctx, CBLAS_TRANSPOSE blas_trans_a,
                   CBLAS_TRANSPOSE blas_trans_b, int32_t batch_size, int m, int n, int k,
                   const T* a_dptr, const T* b_dptr, T* c_dptr, T** buf_dptr) {
    NewKernelUtil<device_type>::OFBatchedGemm(ctx, blas_trans_a, blas_trans_b, batch_size, m, n, k,
                                              GetOneVal<T>(), a_dptr, b_dptr, GetZeroVal<T>(),
                                              c_dptr, buf_dptr);
  }
};  // namespace oneflow

template<>
struct BatchMatMulUtil<DeviceType::kGPU, float16> {
  static void Calc(const JobDesc& job_desc, DeviceCtx* ctx, CBLAS_TRANSPOSE blas_trans_a,
                   CBLAS_TRANSPOSE blas_trans_b, int32_t batch_size, int m, int n, int k,
                   const float16* a_dptr, const float16* b_dptr, float16* c_dptr,
                   float16** buf_dptr) {
    if (job_desc.enable_float_compute_for_half_gemm()) {
      NewKernelUtil<DeviceType::kGPU>::OFBatchedHGemmWithFloat(
          ctx, blas_trans_a, blas_trans_b, batch_size, m, n, k, GetOneVal<float>(), a_dptr, b_dptr,
          GetZeroVal<float>(), c_dptr, buf_dptr);
    } else {
      NewKernelUtil<DeviceType::kGPU>::OFBatchedGemm(ctx, blas_trans_a, blas_trans_b, batch_size, m,
                                                     n, k, GetOneVal<float16>(), a_dptr, b_dptr,
                                                     GetZeroVal<float16>(), c_dptr, buf_dptr);
    }
  }
};

}  // namespace

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a_blob = BnInOp2Blob("a");
  const Blob* b_blob = BnInOp2Blob("b");
  Blob* fw_buf_blob = BnInOp2Blob("fw_buf");
  Blob* out_blob = BnInOp2Blob("out");
  bool transpose_a = this->op_conf().matmul_conf().transpose_a();
  bool transpose_b = this->op_conf().matmul_conf().transpose_b();
  if (a_blob->static_shape().dim_vec().size() == 2) {
    Calc2DMatMul(ctx.device_ctx, a_blob, transpose_a, b_blob, transpose_b, out_blob);
  } else {
    CalcBatchMatMul(ctx.device_ctx, a_blob, transpose_a, b_blob, transpose_b, out_blob,
                    fw_buf_blob);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& MatmulKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().matmul_conf();
}

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::Calc2DMatMul(DeviceCtx* ctx, const Blob* a, bool trans_a,
                                                const Blob* b, bool trans_b, Blob* c) const {
  CBLAS_TRANSPOSE blas_trans_a = trans_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE blas_trans_b = trans_b ? CblasTrans : CblasNoTrans;
  Dim2MatMulUtil<device_type, T>::Calc(this->job_desc(), ctx, a, blas_trans_a, b, blas_trans_b, c);
}

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::CalcBatchMatMul(DeviceCtx* ctx, const Blob* a, bool trans_a,
                                                   const Blob* b, bool trans_b, Blob* c,
                                                   Blob* buf) const {
  CBLAS_TRANSPOSE blas_trans_a = trans_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE blas_trans_b = trans_b ? CblasTrans : CblasNoTrans;
  int32_t num_axes = a->shape().NumAxes();
  int32_t batch_size = a->shape().Count(0, num_axes - 2);
  int m = trans_a ? a->shape().At(num_axes - 1) : a->shape().At(num_axes - 2);
  int k = trans_a ? a->shape().At(num_axes - 2) : a->shape().At(num_axes - 1);
  int n = trans_b ? b->shape().At(num_axes - 2) : b->shape().At(num_axes - 1);
  const T* a_dptr = a->dptr<T>();
  const T* b_dptr = b->dptr<T>();
  T* c_dptr = c->mut_dptr<T>();
  T** buf_dptr = reinterpret_cast<T**>(buf->mut_dptr<int64_t>());
  BatchMatMulUtil<device_type, T>::Calc(this->job_desc(), ctx, blas_trans_a, blas_trans_b,
                                        batch_size, m, n, k, a_dptr, b_dptr, c_dptr, buf_dptr);
}

#define REGISTER_MATMUL_KERNEL(dev, dtype)                                     \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kMatmulConf, dev, dtype, \
                                        MatmulKernel<dev, dtype>)

REGISTER_MATMUL_KERNEL(DeviceType::kGPU, float);
REGISTER_MATMUL_KERNEL(DeviceType::kGPU, double);
REGISTER_MATMUL_KERNEL(DeviceType::kCPU, float);
REGISTER_MATMUL_KERNEL(DeviceType::kCPU, double);
REGISTER_MATMUL_KERNEL(DeviceType::kGPU, float16);

}  // namespace oneflow
