#include "oneflow/core/kernel/innerproduct_kernel.h"
#include <string>
#include "oneflow/core/common/cblas.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace {

template<typename floating_point_type>
void BlasMatrixMult(
    Channel<std::function<void()>>* cpu_stream,
    const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    const floating_point_type alpha, const floating_point_type beta,
    Blob* A, Blob* B, Blob* C) {
}

// C = alpha * A * B + beta * C
template<>
void BlasMatrixMult<float>(
    Channel<std::function<void()>>* cpu_stream,
    const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    const float alpha, const float beta, Blob* A, Blob* B, Blob* C) {
  const enum CBLAS_ORDER Order = CblasRowMajor;
  const int M = C->shape().NumAxes();  // rows number of C
  const int N = C->shape().At(0);  // columns number of C
  const int K = A->shape().At(0);  // columns number of A
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransA == CblasNoTrans) ? N : K;
  const int ldc = N;

  std::function<void()> fp = [=]() {
    cblas_sgemm(
        Order, TransA, TransB, M, N, K, alpha,
        static_cast<const float*>(A->dptr()), lda,
        static_cast<const float*>(B->dptr()), ldb, beta,
        static_cast<float*>(C->mut_dptr()), ldc);
  };
  cpu_stream->Send(fp);
}

template<>
void BlasMatrixMult<double>(
    Channel<std::function<void()>>* cpu_stream,
    const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    const double alpha, const double beta, Blob* A, Blob* B, Blob* C) {
  const enum CBLAS_ORDER Order = CblasRowMajor;
  const int M = C->shape().NumAxes();
  const int N = C->shape().At(0);
  const int K = A->shape().At(0);
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransA == CblasNoTrans) ? N : K;
  const int ldc = N;

  std::function<void()> fp = [=]() {
    cblas_dgemm(Order, TransA, TransB, M, N, K, alpha,
        static_cast<const double*>(A->dptr()), lda,
        static_cast<const double*>(B->dptr()), ldb, beta,
        static_cast<double*>(C->mut_dptr()), ldc);
  };
  cpu_stream->Send(fp);
}

}  // namespace

template<typename floating_point_type>
void InnerProductKernel<DeviceType::kCPU, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_data  = BnInOp2BlobPtr(op()->SoleIbn());
  Blob* out_data = BnInOp2BlobPtr(op()->SoleObn());

  Blob* weight = BnInOp2BlobPtr(op()->model_bns().at(0));
  Blob* bias = BnInOp2BlobPtr(op()->model_bns().at(1));
  Blob* bias_multiplier = BnInOp2BlobPtr(op()->model_tmp_bns().at(0));

  // out_data = in_data * weight.t
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasNoTrans, CblasTrans,
      (floating_point_type)1., (floating_point_type)0.,
      in_data, weight, out_data);

  // out_data = bias_multiplier * bias + out_data
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasNoTrans, CblasNoTrans,
      (floating_point_type)1., (floating_point_type)1.,
      bias_multiplier, bias, out_data);
}

template<typename floating_point_type>
void InnerProductKernel<DeviceType::kCPU, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_data = BnInOp2BlobPtr(op()->SoleIbn());

  Blob* out_diff = BnInOp2BlobPtr(op()->SoleOdbn());
  Blob* in_diff = BnInOp2BlobPtr(op()->SoleIdbn());

  Blob* weight = BnInOp2BlobPtr(op()->model_bns().at(0));
  Blob* bias_multiplier = BnInOp2BlobPtr(op()->model_tmp_bns().at(0));

  Blob* weight_diff = BnInOp2BlobPtr(op()->model_diff_bns().at(0));
  Blob* bias_diff = BnInOp2BlobPtr(op()->model_diff_bns().at(1));

  // in_diff = out_diff * weight
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasNoTrans, CblasNoTrans,
      (floating_point_type)1., (floating_point_type)0.,
      out_diff, weight, in_diff);

  // weight_diff = out_diff.t * in_data
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasTrans, CblasNoTrans,
      (floating_point_type)1., (floating_point_type)0.,
      out_diff, in_data, weight_diff);

  // bias_diff = out_diff.t * bias_multiplier
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasTrans, CblasNoTrans,
      (floating_point_type)1., (floating_point_type)0.,
      out_diff, bias_multiplier, bias_diff);
}

INSTANTIATE_CPU_KERNEL_CLASS(InnerProductKernel);
REGISTER_CPU_KERNEL(OperatorConf::kInnerproductConf, InnerProductKernel);

}  // namespace oneflow
