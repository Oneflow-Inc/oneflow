#include "oneflow/core/kernel/gather_grad_kernel.h"
#include "oneflow/core/kernel/gather_kernel.h"

namespace oneflow {

namespace {

Shape GetFlatShape(const Shape& shape, int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<DeviceType device_type, typename T, typename K>
void GatherForward(DeviceCtx* ctx, const Blob* indices, const Blob* in, int64_t axis, Blob* out) {
  const Shape flat_in_shape = GetFlatShape(in->shape(), axis);
  GatherKernelUtil<device_type, T, K>::Forward(ctx, indices->dptr<K>(), indices->shape().elem_cnt(),
                                               in->dptr<T>(), flat_in_shape, out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename K>
void GatherBackward(DeviceCtx* ctx, const Blob* indices, const Blob* out_diff, int64_t axis,
                    Blob* in_diff) {
  Memset<device_type>(ctx, in_diff->mut_dptr<T>(), 0, in_diff->ByteSizeOfDataContentField());
  const Shape flat_in_shape = GetFlatShape(in_diff->shape(), axis);
  GatherKernelUtil<device_type, T, K>::Backward(ctx, indices->dptr<K>(),
                                                indices->shape().elem_cnt(), out_diff->dptr<T>(),
                                                flat_in_shape, in_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct GatherSwitchUtil final {
#define MAKE_GATHER_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
#define DEFINE_GATHER_STATIC_SWITCH_FUNC(func_name)                    \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_GATHER_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
  DEFINE_GATHER_STATIC_SWITCH_FUNC(GatherForward);
  DEFINE_GATHER_STATIC_SWITCH_FUNC(GatherBackward);
#undef DEFINE_GATHER_STATIC_SWITCH_FUNC
#undef MAKE_GATHER_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& GatherGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_grad_conf();
}

template<DeviceType device_type, typename T>
void GatherGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GatherSwitchUtil<device_type, T>::SwitchGatherBackward(
      SwitchCase(BnInOp2Blob("indices")->data_type()), ctx.device_ctx, BnInOp2Blob("indices"),
      BnInOp2Blob("out_diff"), this->op_conf().gather_grad_conf().axis(), BnInOp2Blob("in_diff"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherGradConf, GatherGradKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
