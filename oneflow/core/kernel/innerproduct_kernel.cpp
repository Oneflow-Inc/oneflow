#include "oneflow/core/kernel/innerproduct_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename floating_point_type>
void BlasMatrixMatrix(
    const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans_a,
    const enum CBLAS_TRANSPOSE trans_b, const floating_point_type alpha,
    const floating_point_type beta, Blob* a, Blob* b, Blob* c) {
  const int m = c->shape().At(0);
  const int n = c->shape().At(1);
  const int k = (trans_a == CblasNoTrans) ? a->shape().At(1) : a->shape().At(0);

  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  KernelUtil<device_type, floating_point_type>::BlasGemm(
      ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha,
      static_cast<const floating_point_type*>(a->dptr()), lda,
      static_cast<const floating_point_type*>(b->dptr()), ldb, beta,
      static_cast<floating_point_type*>(c->mut_dptr()), ldc);
}

}  // namespace

template<DeviceType device_type, typename floating_point_type>
void InnerProductKernel<device_type, floating_point_type>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);

  const InnerProductOpConf& inner_product_conf =
    op()->op_conf().innerproduct_conf();
  has_bias_term_ = inner_product_conf.has_bias_term();
}

template<DeviceType device_type, typename floating_point_type>
void InnerProductKernel<device_type, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_data  = BnInOp2BlobPtr("in");
  Blob* out_data = BnInOp2BlobPtr("out");
  Blob* weight = BnInOp2BlobPtr("weight");

  // out_data = in_data * weight.t
  BlasMatrixMatrix<device_type, floating_point_type>(
      ctx, CblasNoTrans, CblasTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      in_data, weight, out_data);
  
  if (has_bias_term_) {
    Blob* bias = BnInOp2BlobPtr("bias");
    Blob* bias_multiplier = BnInOp2BlobPtr("bias_multiplier");
    
    // out_data = bias_multiplier * bias + out_data
    BlasMatrixMatrix<device_type, floating_point_type>(
        ctx, CblasNoTrans, CblasNoTrans,
        static_cast<floating_point_type>(1.0),
        static_cast<floating_point_type>(1.0),
        bias_multiplier, bias, out_data);
  }
}

template<DeviceType device_type, typename floating_point_type>
void InnerProductKernel<device_type, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_data = BnInOp2BlobPtr("in");
  Blob* out_diff = BnInOp2BlobPtr("out_diff");
  Blob* in_diff = BnInOp2BlobPtr("in_diff");

  Blob* weight = BnInOp2BlobPtr("weight");
  Blob* weight_diff = BnInOp2BlobPtr("weight_diff");

  // in_diff = out_diff * weight
  if (in_diff != nullptr) {
    BlasMatrixMatrix<device_type, floating_point_type>(
        ctx, CblasNoTrans, CblasNoTrans,
        static_cast<floating_point_type>(1.0),
        static_cast<floating_point_type>(0.0),
        out_diff, weight, in_diff);
  }

  // weight_diff = out_diff.t * in_data
  BlasMatrixMatrix<device_type, floating_point_type>(
      ctx, CblasTrans, CblasNoTrans,
      static_cast<floating_point_type>(1.0),
      static_cast<floating_point_type>(0.0),
      out_diff, in_data, weight_diff);
  
  if (has_bias_term_) {
    Blob* bias_diff = BnInOp2BlobPtr("bias_diff");
    Blob* bias_multiplier = BnInOp2BlobPtr("bias_multiplier");

    // bias_diff = bias_multiplier.t * out_diff 
    BlasMatrixMatrix<device_type, floating_point_type>(
        ctx, CblasTrans, CblasNoTrans,
        static_cast<floating_point_type>(1.0),
        static_cast<floating_point_type>(0.0),
        bias_multiplier, out_diff, bias_diff);
  }
}

template<DeviceType device_type, typename floating_point_type>
void InnerProductKernel<device_type, floating_point_type>::
  InitModelAndModelTmpBlobsWithoutSnapshot(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
}

INSTANTIATE_KERNEL_CLASS(InnerProductKernel);
REGISTER_KERNEL(OperatorConf::kInnerproductConf, InnerProductKernel);

}  // namespace oneflow
