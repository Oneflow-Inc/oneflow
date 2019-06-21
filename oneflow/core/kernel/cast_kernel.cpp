#include "oneflow/core/kernel/cast_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename U>
void CopyBlob(DeviceCtx* ctx, const Blob* src, Blob* dst) {
  CHECK_EQ(src->shape(), dst->shape());
  if (device_type == DeviceType::kCPU) {
    CopyElem(src->dptr<T>(), dst->mut_dptr<U>(), src->shape().elem_cnt());
  } else if (device_type == DeviceType::kGPU) {
    CopyElemOnGpu(ctx, src->dptr<T>(), dst->mut_dptr<U>(), src->shape().elem_cnt());
  } else {
    UNIMPLEMENTED();
  }
}

}  // namespace

#define MAKE_CAST_SWITCH_ENTRY(func_name, T, U) func_name<device_type, T, U>
template<DeviceType device_type>
struct CastUtil final {
  DEFINE_STATIC_SWITCH_FUNC(void, CopyBlob, MAKE_CAST_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ),
                            MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ));
};

template<DeviceType device_type>
void CastKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CastUtil<device_type>::SwitchCopyBlob(SwitchCase(in_blob->data_type(), out_blob->data_type()),
                                        ctx.device_ctx, in_blob, out_blob);
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kCastConf, CastKernel);

}  // namespace oneflow
