#include "oneflow/core/kernel/innerproduct_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
void BlasMatrixMatrix(const KernelCtx& ctx, const enum CBLAS_TRANSPOSE trans_a,
                      const enum CBLAS_TRANSPOSE trans_b, const T alpha,
                      const T beta, const Blob* a, const Blob* b, Blob* c) {
  const int m = c->shape().At(0);
  const int n = c->shape().Count(1);
  const int k =
      (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  KernelUtil<device_type, T>::BlasGemm(
      ctx.device_ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha,
      a->dptr<T>(), lda, b->dptr<T>(), ldb, beta, c->mut_dptr<T>(), ldc);
}

}  // namespace

template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* in = BnInOp2BlobPtr("in");
  const Blob* weight = BnInOp2BlobPtr("weight");
  Blob* out = BnInOp2BlobPtr("out");

  // out = in * weight
  BlasMatrixMatrix<device_type, T>(ctx, CblasNoTrans, CblasTrans,
                                   static_cast<T>(1.0), static_cast<T>(0.0), in,
                                   weight, out);

  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    const Blob* bias = BnInOp2BlobPtr("bias");
    const Blob* bias_multiplier = BnInOp2BlobPtr("bias_multiplier");

    // out = bias_multiplier * bias + out
    BlasMatrixMatrix<device_type, T>(ctx, CblasNoTrans, CblasNoTrans,
                                     static_cast<T>(1.0), static_cast<T>(1.0),
                                     bias_multiplier, bias, out);
  }
}

template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* in = BnInOp2BlobPtr("in");
  const Blob* out_diff = BnInOp2BlobPtr("out_diff");
  Blob* in_diff = BnInOp2BlobPtr("in_diff");

  const Blob* weight = BnInOp2BlobPtr("weight");
  Blob* weight_diff = BnInOp2BlobPtr("weight_diff");

  // in_diff = out_diff * weight
  if (in_diff != nullptr) {
    BlasMatrixMatrix<device_type, T>(ctx, CblasNoTrans, CblasNoTrans,
                                     static_cast<T>(1.0), static_cast<T>(0.0),
                                     out_diff, weight, in_diff);
  }

  // weight_diff = out_diff * in
  BlasMatrixMatrix<device_type, T>(ctx, CblasTrans, CblasNoTrans,
                                   static_cast<T>(1.0), static_cast<T>(0.0),
                                   out_diff, in, weight_diff);

  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    const Blob* bias_multiplier = BnInOp2BlobPtr("bias_multiplier");
    Blob* bias_diff = BnInOp2BlobPtr("bias_diff");

    // bias_diff = bias_multiplier * out_diff
    BlasMatrixMatrix<device_type, T>(ctx, CblasTrans, CblasNoTrans,
                                     static_cast<T>(1.0), static_cast<T>(0.0),
                                     bias_multiplier, out_diff, bias_diff);
  }
}

template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::InitModelBlobsWithRandomSeed(
    const KernelCtx& ctx, std::mt19937 random_seed_gen,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  KernelUtil<device_type, T>::FillWithProperConf(
      ctx.device_ctx,
      OF_PB_POINTER_GET(op()->op_conf().innerproduct_conf(), weight_fill),
      random_seed_gen(), BnInOp2Blob("weight"));

  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    KernelUtil<device_type, T>::FillWithProperConf(
        ctx.device_ctx,
        OF_PB_POINTER_GET(op()->op_conf().innerproduct_conf(), bias_fill),
        random_seed_gen(), BnInOp2Blob("bias"));
  }
}
template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::InitModelBlobsWithSnapshot(
    const KernelCtx& ctx, int32_t part_id, int32_t part_num,
    const Snapshot* snapshot,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* weight_blob = BnInOp2Blob("weight");
  int32_t dim_num = op()->GetInt32FromSpecialConf("out_num");
  KernelUtil<device_type, T>::FillWithSnapshot(
      ctx.device_ctx, part_id, part_num, snapshot, weight_blob,
      op()->Lbn4BnInOp("weight"), dim_num, weight_blob->shape().Count(1));
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    KernelUtil<device_type, T>::FillWithSnapshot(
        ctx.device_ctx, part_id, part_num, snapshot, BnInOp2Blob("bias"),
        op()->Lbn4BnInOp("bias"), dim_num, 1);
  }
}

template<DeviceType device_type, typename T>
void InnerProductKernel<device_type, T>::InitModelTmpBlobs(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (op()->GetBoolFromSpecialConf("has_bias_term")) {
    FillConf bias_multiplier_fill_conf;
    bias_multiplier_fill_conf.mutable_constant_conf()->set_value(1.0f);
    KernelUtil<device_type, T>::Fill(ctx.device_ctx, bias_multiplier_fill_conf,
                                     0, BnInOp2Blob("bias_multiplier"));
  }
}

namespace {

template<DeviceType device_type>
Kernel* CreateInnerProductKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define MACRO_PAIR(type_cpp, type_proto) \
  {type_proto, []() { return new InnerProductKernel<device_type, type_cpp>; }},
      FLOATING_DATA_TYPE_PAIR()
#undef MACRO_PAIR
  };
  return data_type2creator.at(op_conf.innerproduct_conf().in().data_type())();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kInnerproductConf, DeviceType::kCPU,
                         CreateInnerProductKernel<DeviceType::kCPU>);
        AddKernelCreator(OperatorConf::kInnerproductConf, DeviceType::kGPU,
                         CreateInnerProductKernel<DeviceType::kGPU>));

}  // namespace oneflow
