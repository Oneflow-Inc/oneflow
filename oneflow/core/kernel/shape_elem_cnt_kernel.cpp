#include "oneflow/core/kernel/shape_elem_cnt_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ShapeElemCntKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int32_t begin_axis = this->op_conf().shape_elem_cnt_conf().begin_axis();
  int32_t end_axis = this->op_conf().shape_elem_cnt_conf().end_axis();
  if (end_axis < 0) { end_axis += BnInOp2Blob("x")->shape().NumAxes(); }
  const int32_t elem_cnt = BnInOp2Blob("x")->shape().Count(begin_axis, end_axis);
  KernelUtil<device_type, int32_t>::Set(ctx.device_ctx, elem_cnt,
                                        BnInOp2Blob("y")->mut_dptr<int32_t>());
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kShapeElemCntConf, ShapeElemCntKernel);

}  // namespace oneflow
