#include "oneflow/core/kernel/innerproduct_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace {

template<typename floating_point_type>
void BlasMatrixMatrix(
    Channel<std::function<void()>>* cpu_stream,
    const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    const floating_point_type alpha, const floating_point_type beta,
    Blob* A, Blob* B, Blob* C) {
  const int M = C->shape().At(0);  // rows number of C
  const int N = C->shape().At(1);  // colms of C
  // colms of op(A)
  const int K = (TransA == CblasNoTrans) ? A->shape().At(1) : A->shape().At(0);

  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  const int ldc = N;

  cpu_stream->Send([=]() {
    cblas_gemm<floating_point_type>(
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
  BlasMatrixMatrix<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasNoTrans, CblasTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      in_data, weight, out_data);
  
  // out_data = bias_multiplier * bias + out_data
  BlasMatrixMatrix<floating_point_type>(
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
    BlasMatrixMatrix<floating_point_type>(
        ctx.device_ctx->cpu_stream(), CblasNoTrans, CblasNoTrans,
        static_cast<floating_point_type>(1.0),
        static_cast<floating_point_type>(0.0),
        out_diff, weight, in_diff);
  }

  // weight_diff = out_diff.t * in_data
  BlasMatrixMatrix<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      out_diff, in_data, weight_diff);
  
  // bias_diff = bias_multiplier.t * out_diff 
  BlasMatrixMatrix<floating_point_type>(
      ctx.device_ctx->cpu_stream(), CblasTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      bias_multiplier, out_diff, bias_diff);
}

INSTANTIATE_CPU_KERNEL_CLASS(InnerProductKernel);
REGISTER_CPU_KERNEL(OperatorConf::kInnerproductConf, InnerProductKernel);

}  // namespace oneflow
