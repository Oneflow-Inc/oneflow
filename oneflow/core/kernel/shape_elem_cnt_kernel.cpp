#include "oneflow/core/kernel/shape_elem_cnt_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ShapeElemCntKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int32_t elem_cnt = GetShapePartialElemCnt(BnInOp2Blob("x")->shape());
  KernelUtil<device_type, int32_t>::Set(ctx.device_ctx, elem_cnt,
                                        BnInOp2Blob("y")->mut_dptr<int32_t>());
}

template<DeviceType device_type>
int32_t ShapeElemCntKernel<device_type>::GetShapePartialElemCnt(const Shape& shape) const {
  int32_t ret = 1;
  for (int32_t axis : this->kernel_conf().shape_elem_cnt_conf().axis()) { ret *= shape.At(axis); }
  return ret;
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kShapeElemCntConf, ShapeElemCntKernel);

}  // namespace oneflow
