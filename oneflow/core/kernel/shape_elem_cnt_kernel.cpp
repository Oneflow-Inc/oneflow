#include "oneflow/core/kernel/shape_elem_cnt_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ShapeElemCntKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const T elem_cnt = GetShapePartialElemCnt(BnInOp2Blob("x")->shape());
  KernelUtil<device_type, T>::Set(ctx.device_ctx, elem_cnt, BnInOp2Blob("y")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
int32_t ShapeElemCntKernel<device_type, T>::GetShapePartialElemCnt(
    const DenseShapeView& shape) const {
  int32_t ret = 1;
  for (int32_t axis : this->kernel_conf().shape_elem_cnt_conf().axis()) { ret *= shape.At(axis); }
  return ret;
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kShapeElemCntConf, ShapeElemCntKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
