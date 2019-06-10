#include "oneflow/core/kernel/matmul_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
struct Dim2MatMulUtil {
  static void Calc(DeviceCtx* ctx, const Blob* a, CBLAS_TRANSPOSE blas_trans_a, const Blob* b,
                   CBLAS_TRANSPOSE blas_trans_b, Blob* c) {
    NewKernelUtil<device_type>::BlobGemm(ctx, blas_trans_a, blas_trans_b, OneVal<T>::value,
                                         ZeroVal<T>::value, a, b, c);
  }
};  // namespace oneflow

template<>
struct Dim2MatMulUtil<DeviceType::kGPU, float16> {
  static void Calc(DeviceCtx* ctx, const Blob* a, CBLAS_TRANSPOSE blas_trans_a, const Blob* b,
                   CBLAS_TRANSPOSE blas_trans_b, Blob* c) {
    if (Global<JobDesc>::Get()->enable_cublashgemm_when_matmul()) {
      NewKernelUtil<DeviceType::kGPU>::BlobGemm(ctx, blas_trans_a, blas_trans_b,
                                                OneVal<float16>::value, ZeroVal<float16>::value, a,
                                                b, c);
    } else {
      NewKernelUtil<DeviceType::kGPU>::BlobHGemmWithFloat(
          ctx, blas_trans_a, blas_trans_b, OneVal<float>::value, ZeroVal<float>::value, a, b, c);
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
  Dim2MatMulUtil<device_type, T>::Calc(ctx, a, blas_trans_a, b, blas_trans_b, c);
}

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::CalcBatchMatMul(DeviceCtx* ctx, const Blob* a, bool trans_a,
                                                   const Blob* b, bool trans_b, Blob* c,
                                                   Blob* buf) const {
  CBLAS_TRANSPOSE blas_trans_a = trans_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE blas_trans_b = trans_b ? CblasTrans : CblasNoTrans;
  int32_t dim_num = a->shape().dim_vec().size();
  int32_t batch_size = a->shape().Count(0, dim_num - 2);
  int m = trans_a ? a->shape().At(dim_num - 1) : a->shape().At(dim_num - 2);
  int k = trans_a ? a->shape().At(dim_num - 2) : a->shape().At(dim_num - 1);
  int n = trans_b ? b->shape().At(dim_num - 2) : b->shape().At(dim_num - 1);
  const T* a_dptr = a->dptr<T>();
  const T* b_dptr = b->dptr<T>();
  T* c_dptr = c->mut_dptr<T>();
  T** buf_dptr = reinterpret_cast<T**>(buf->mut_dptr<int64_t>());
  NewKernelUtil<device_type>::OFBatchedGemm(ctx, blas_trans_a, blas_trans_b, batch_size, m, n, k,
                                            GetOneVal<T>(), a_dptr, b_dptr, GetZeroVal<T>(), c_dptr,
                                            buf_dptr);
}

ADD_GPU_HALF_KERNEL_CREATOR(OperatorConf::kMatmulConf, MatmulKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
