#include "oneflow/core/kernel/cast_inited_model_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void CastInitedModelKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t next_model_version_id = *static_cast<int64_t*>(ctx.other);
  if (next_model_version_id > 0) { return; }
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK(in_blob->shape() == out_blob->shape());
  CHECK(in_blob->data_type() == DataType::kFloat16);
  CHECK(out_blob->data_type() == DataType::kFloat);
  NewKernelUtil<device_type, float16>::Half2Float(ctx.device_ctx, in_blob->shape().elem_cnt(),
                                                  in_blob->dptr<float16>(),
                                                  out_blob->mut_dptr<float>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kCastInitedModelConf, CastInitedModelKernel,
                           OF_PP_MAKE_TUPLE_SEQ(float, DataType::kFloat));

}  // namespace oneflow
