#include "oneflow/core/kernel/float_to_half_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void FloatToHalfKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK(in_blob->shape() == out_blob->shape());
  CHECK(in_blob->data_type() == DataType::kFloat);
  CHECK(out_blob->data_type() == DataType::kFloat16);
  NewKernelUtil<device_type, float16>::Float2Half(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                                  in_blob->dptr<float>(),
                                                  out_blob->mut_dptr<float16>());
}

template<DeviceType device_type, typename T>
void FloatToHalfKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  CHECK(out_diff_blob->shape() == in_diff_blob->shape());
  CHECK(in_diff_blob->data_type() == DataType::kFloat);
  CHECK(out_diff_blob->data_type() == DataType::kFloat16);
  NewKernelUtil<device_type, float16>::Half2Float(ctx.device_ctx, in_diff_blob->shape().elem_cnt(),
                                                  out_diff_blob->dptr<float16>(),
                                                  in_diff_blob->mut_dptr<float>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFloatToHalfConf, FloatToHalfKernel,
                           OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat));

}  // namespace oneflow
