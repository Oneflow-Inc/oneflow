#include "oneflow/core/kernel/cast_kernel.h"
#include "oneflow/core/record/ofrecord_decoder.h"
#include "oneflow/core/common/switch_func.h"

namespace oneflow {

namespace {

template<typename T, typename U>
void CopyBlob(const Blob* src, Blob* dst) {
  CHECK_EQ(src->shape(), dst->shape());
  CopyElem(src->dptr<T>(), dst->mut_dptr<U>(), src->shape().elem_cnt());
}

#define MAKE_CAST_SWITCH_ENTRY(func_name, T, U) func_name<T, U>
DEFINE_STATIC_SWITCH_FUNC(void, CopyBlob, MAKE_CAST_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ),
                          MAKE_DATA_TYPE_CTRV_SEQ(POD_DATA_TYPE_SEQ));

}  // namespace

void CastKernel::ForwardDataContent(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  SwitchCopyBlob(SwitchCase(in_blob->data_type(), out_blob->data_type()), in_blob, out_blob);
}

REGISTER_KERNEL(OperatorConf::kCastConf, CastKernel);

}  // namespace oneflow
