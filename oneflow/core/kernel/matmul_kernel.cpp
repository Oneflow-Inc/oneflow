#include "oneflow/core/kernel/matmul_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a_blob = BnInOp2Blob("a");
  const Blob* b_blob = BnInOp2Blob("b");
  Blob* out_blob = BnInOp2Blob("out");
  bool transpose_a = this->op_conf().matmul_conf().transpose_a();
  bool transpose_b = this->op_conf().matmul_conf().transpose_b();
  if (a_blob->static_shape().dim_vec().size() == 2) {
    Calc2DMatMul(ctx.device_ctx, a_blob, transpose_a, b_blob, transpose_b, out_blob, false);
  } else {
    CalcBatchMatMul(ctx.device_ctx, a_blob, transpose_a, b_blob, transpose_b, out_blob, false);
  }
}

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* a_blob = BnInOp2Blob("a");
  const Blob* b_blob = BnInOp2Blob("b");
  const Blob* out_diff_blob = BnInOp2Blob("out");
  Blob* a_diff_blob = BnInOp2Blob("a_diff");
  Blob* b_diff_blob = BnInOp2Blob("b_diff");
  bool transpose_a = this->op_conf().matmul_conf().transpose_a();
  bool transpose_b = this->op_conf().matmul_conf().transpose_b();
  // trans_a  trans_b  a_diff b_diff
  //   T        T       b'g'   g'a'
  //   T        F       bg'    ag
  //   F        T       gb     g'a
  //   F        F       gb'    a'g
  if (a_blob->static_shape().dim_vec().size() == 2) {
    Calc2DMatMul(ctx.device_ctx, b_blob, !(transpose_a ^ transpose_b), out_diff_blob, transpose_a,
                 a_diff_blob, !transpose_a);
    Calc2DMatMul(ctx.device_ctx, a_blob, !(transpose_a ^ transpose_b), out_diff_blob, transpose_b,
                 b_diff_blob, transpose_b);
  } else {
    CalcBatchMatMul(ctx.device_ctx, b_blob, !(transpose_a ^ transpose_b), out_diff_blob,
                    transpose_a, a_diff_blob, !transpose_a);
    CalcBatchMatMul(ctx.device_ctx, a_blob, !(transpose_a ^ transpose_b), out_diff_blob,
                    transpose_b, b_diff_blob, transpose_b);
  }
}

template<DeviceType device_type, typename T>
const PbMessage& MatmulKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().matmul_conf();
}

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::Calc2DMatMul(DeviceCtx* ctx, const Blob* a, bool trans_a,
                                                const Blob* b, bool trans_b, Blob* c,
                                                bool swap_in) const {
  CBLAS_TRANSPOSE blas_trans_a = trans_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE blas_trans_b = trans_b ? CblasTrans : CblasNoTrans;
  if (swap_in) {
    KernelUtil<device_type, T>::BlobGemm(ctx, blas_trans_b, blas_trans_a, OneVal<T>::value,
                                         ZeroVal<T>::value, b, a, c);
  } else {
    KernelUtil<device_type, T>::BlobGemm(ctx, blas_trans_a, blas_trans_b, OneVal<T>::value,
                                         ZeroVal<T>::value, a, b, c);
  }
}

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::CalcBatchMatMul(DeviceCtx* ctx, const Blob* a, bool trans_a,
                                                   const Blob* b, bool trans_b, Blob* c,
                                                   bool swap_in) const {
  if (swap_in) {
    CalcBatchMatMul(ctx, b, trans_b, a, trans_a, c);
  } else {
    CalcBatchMatMul(ctx, a, trans_a, b, trans_b, c);
  }
}

template<DeviceType device_type, typename T>
void MatmulKernel<device_type, T>::CalcBatchMatMul(DeviceCtx* ctx, const Blob* a, bool trans_a,
                                                   const Blob* b, bool trans_b, Blob* c) const {
  CBLAS_TRANSPOSE blas_trans_a = trans_a ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE blas_trans_b = trans_b ? CblasTrans : CblasNoTrans;
  int32_t dim_num = a->shape().dim_vec().size();
  int32_t mul_num = a->shape().Count(0, dim_num - 3);
  int32_t stride_a = a->shape().Count(dim_num - 3);
  int32_t stride_b = b->shape().Count(dim_num - 3);
  int32_t stride_c = c->shape().Count(dim_num - 3);
  int batch_size = a->shape().At(dim_num - 3);
  int m = trans_a ? a->shape().At(dim_num - 1) : a->shape().At(dim_num - 2);
  int k = trans_a ? a->shape().At(dim_num - 2) : a->shape().At(dim_num - 1);
  int n = trans_b ? b->shape().At(dim_num - 2) : b->shape().At(dim_num - 1);
  const T* a_dptr = a->dptr<T>();
  const T* b_dptr = b->dptr<T>();
  T* c_dptr = c->mut_dptr<T>();
  FOR_RANGE(int32_t, i, 0, mul_num) {
    KernelUtil<device_type, T>::OFBatchedGemm(
        ctx, blas_trans_a, blas_trans_b, batch_size, m, n, k, OneVal<T>::value,
        a_dptr + i * stride_a, b_dptr + i * stride_b, ZeroVal<T>::value, c_dptr + i * stride_c);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMatmulConf, MatmulKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
