#include "oneflow/core/kernel/innerproduct_kernel.h"

namespace oneflow {

namespace {

template<typename floating_point_type>
void BlasMatrixMult(
    Channel<std::function<void()>>* cpu_stream,
    const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    const floating_point_type alpha, const floating_point_type beta,
    Blob* A, Blob* B, Blob* C) {
  const int M = C->shape().NumAxes();  // rows number of C
  const int N = C->shape().At(0);  // columns number of C
  const int K = A->shape().At(0);  // columns number of A
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransA == CblasNoTrans) ? N : K;
  const int ldc = N;

  cpu_stream->Send([=]() {
    cblas_gemm(
        TransA, TransB, M, N, K, alpha,
        static_cast<const floating_point_type*>(A->dptr()), lda,
        static_cast<const floating_point_type*>(B->dptr()), ldb, beta,
        static_cast<floating_point_type*>(C->mut_dptr()), ldc);
  });
}

}  // namespace

template<typename floating_point_type>
void InnerProductKernel<DeviceType::kCPU, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_data  = BnInOp2BlobPtr("in");
  Blob* out_data = BnInOp2BlobPtr("out");

  Blob* weight = BnInOp2BlobPtr("weight");
  Blob* bias = BnInOp2BlobPtr("bias");
  Blob* bias_multiplier = BnInOp2BlobPtr("bias_multiplier");

  // out_data = in_data * weight.t
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasNoTrans, CblasTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      in_data, weight, out_data);

  // out_data = bias_multiplier * bias + out_data
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasNoTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(1.0),
      bias_multiplier, bias, out_data);
}

template<typename floating_point_type>
void InnerProductKernel<DeviceType::kCPU, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_data = BnInOp2BlobPtr("in");

  Blob* out_diff = BnInOp2BlobPtr("out_diff");
  Blob* in_diff = BnInOp2BlobPtr("in_diff");

  Blob* weight = BnInOp2BlobPtr("weight");
  Blob* bias_multiplier = BnInOp2BlobPtr("bias_multiplier");

  Blob* weight_diff = BnInOp2BlobPtr("weight_diff");
  Blob* bias_diff = BnInOp2BlobPtr("bias_diff");

  // in_diff = out_diff * weight
  if (in_diff != nullptr) {
    BlasMatrixMult<floating_point_type>(
        ctx.device_ctx->cpu_stream(), CblasNoTrans, CblasNoTrans,
        static_cast<floating_point_type>(1.0),
        static_cast<floating_point_type>(0.0),
        out_diff, weight, in_diff);
  }

  // weight_diff = out_diff.t * in_data
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      out_diff, in_data, weight_diff);

  // bias_diff = out_diff.t * bias_multiplier
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      out_diff, bias_multiplier, bias_diff);
}

INSTANTIATE_CPU_KERNEL_CLASS(InnerProductKernel);
REGISTER_CPU_KERNEL(OperatorConf::kInnerproductConf, InnerProductKernel);

}  // namespace oneflow
