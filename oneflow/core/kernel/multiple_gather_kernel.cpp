#include "oneflow/core/kernel/multiple_gather_kernel.h"
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
const PbMessage& MultipleGatherKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().multiple_gather_conf();
}

template<DeviceType device_type, typename T>
void MultipleGatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const MultipleGatherOpConf& conf = this->op_conf().multiple_gather_conf();
  CHECK_GT(conf.indices().size(), 0);
  CHECK_EQ(conf.indices().size(), conf.out().size());
  FOR_RANGE(int32_t, i, 0, conf.indices().size()) {
    const Blob* indices = BnInOp2Blob(GenRepeatedBn("indices", i));
    const Blob* in = BnInOp2Blob("in");
    Blob* out = BnInOp2Blob(GenRepeatedBn("out", i));
    GatherSwitchUtil<device_type, T>::SwitchGatherForward(SwitchCase(indices->data_type()),
                                                          ctx.device_ctx, indices, in, 0, out);
  }
}

template<DeviceType device_type, typename T>
void MultipleGatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const MultipleGatherOpConf& conf = this->op_conf().multiple_gather_conf();
  CHECK_GT(conf.indices().size(), 0);
  CHECK_EQ(conf.indices().size(), conf.out().size());
  Blob* in_diff = BnInOp2Blob(GenDiffBn("in"));
  Memset<device_type>(ctx.device_ctx, in_diff->mut_dptr<T>(), 0,
                      in_diff->ByteSizeOfDataContentField());
  FOR_RANGE(int32_t, i, 0, conf.indices().size()) {
    const Blob* indices = BnInOp2Blob(GenRepeatedBn("indices", i));
    const Blob* out_diff = BnInOp2Blob(GenDiffBn(GenRepeatedBn("out", i)));
    if (out_diff == nullptr) { continue; }
    GatherSwitchUtil<device_type, T>::SwitchGatherBackward(
        SwitchCase(indices->data_type()), ctx.device_ctx, indices, out_diff, 0, in_diff);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kMultipleGatherConf, MultipleGatherKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
