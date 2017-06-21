#include "oneflow/core/kernel/innerproduct_kernel.h"
#include <string>
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

namespace {

template<typename floating_point_type>
void BlasMatrixMult(
    const cublasHandle_t& cublas_handle, const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB, const floating_point_type alpha,
    const floating_point_type beta, Blob* A, Blob* B, Blob* C) {
  cublasOperation_t cuTransA =
    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int M = C->shape().NumAxes();
  const int N = C->shape().At(0);
  const int K = A->shape().At(0);
  const int lda = (TransA == CblasNoTrans) ? K : M;
  const int ldb = (TransB == CblasNoTrans) ? N : K;
  const int ldc = N;

  cublas_gemm<floating_point_type>(
      cublas_handle, cuTransA, cuTransB, M, N, K, &alpha,
      static_cast<const floating_point_type*>(A->dptr()), lda,
      static_cast<const floating_point_type*>(B->dptr()), ldb, &beta,
      static_cast<floating_point_type*>(C->mut_dptr()), ldc);
}

}  // namespace

template<typename floating_point_type>
void InnerProductKernel<DeviceType::kGPU, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_data = BnInOp2BlobPtr("in");
  Blob* out_data = BnInOp2BlobPtr("out");

  Blob* weight = BnInOp2BlobPtr("weight");
  Blob* bias = BnInOp2BlobPtr("bias");
  Blob* bias_multiplier = BnInOp2BlobPtr("bias_multiplier");

  // out_data = in_data * weight.t
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cublas_handle(), CblasNoTrans, CblasTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      in_data, weight, out_data);

  // out_data = bias_multiplier * bias + out_data
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cublas_handle(), CblasNoTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      bias_multiplier, bias, out_data);
}

template<typename floating_point_type>
void InnerProductKernel<DeviceType::kGPU, floating_point_type>::Backward(
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
        ctx.device_ctx->cublas_handle(), CblasNoTrans, CblasNoTrans,
        static_cast<floating_point_type>(1.0),
        static_cast<floating_point_type>(0.0),
        out_diff, weight, in_diff);
  }

  // weight_diff = out_diff.t * in_data
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cublas_handle(), CblasTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      out_diff, in_data, weight_diff);

  // bias_diff = out_diff.t * bias_multiplier
  BlasMatrixMult<floating_point_type>(
      ctx.device_ctx->cublas_handle(), CblasTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      out_diff, bias_multiplier, bias_diff);
}

INSTANTIATE_GPU_KERNEL_CLASS(InnerProductKernel);
REGISTER_GPU_KERNEL(OperatorConf::kInnerproductConf, InnerProductKernel);

}  // namespace oneflow
