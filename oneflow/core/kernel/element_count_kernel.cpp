#include "oneflow/core/kernel/element_count_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ElementCountKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const ElementCountOpConf& conf = this->op_conf().element_count_conf();
  const Blob* in = BnInOp2Blob("in");
  Blob* out = BnInOp2Blob("out");
  const int64_t num_axes = in->shape().NumAxes();
  const int64_t begin_axis = conf.has_begin_axis() ? conf.begin_axis() : 0;
  const int64_t end_axis = conf.has_end_axis() ? conf.end_axis() : num_axes;
  CHECK_GE(begin_axis, 0);
  CHECK_LT(begin_axis, num_axes);
  CHECK_GT(end_axis, begin_axis);
  CHECK_LE(end_axis, num_axes);
  const int64_t cnt = in->shape().Count(begin_axis, end_axis);
  KernelUtil<device_type, T>::Set(ctx.device_ctx, static_cast<T>(cnt),
                                         out->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kElementCountConf, ElementCountKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}  // namespace oneflow
