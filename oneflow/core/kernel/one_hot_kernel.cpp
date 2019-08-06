#include "oneflow/core/kernel/one_hot_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T, typename K>
void OneHot(DeviceCtx* ctx, const Blob* indices, Blob* out) {
  const int64_t depth = out->shape().At(out->shape().NumAxes() - 1);
  OneHotKernelUtil<device_type, T, K>::Encode(ctx, indices->dptr<K>(), indices->shape().elem_cnt(),
                                              depth, out->mut_dptr<T>());
}

}  // namespace

template<DeviceType device_type, typename T>
struct OneHotUtil final {
#define MAKE_ONE_HOT_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
  DEFINE_STATIC_SWITCH_FUNC(void, OneHot, MAKE_ONE_HOT_SWITCH_ENTRY,
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
#undef MAKE_ONE_HOT_SWITCH_ENTRY
};

template<DeviceType device_type, typename T>
const PbMessage& OneHotKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().one_hot_conf();
}

template<DeviceType device_type, typename T>
void OneHotKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  Blob* out = BnInOp2Blob("out");
  OneHotUtil<device_type, T>::SwitchOneHot(SwitchCase(indices->data_type()), ctx.device_ctx,
                                           indices, out);
}

template<typename T, typename K>
struct OneHotKernelUtil<DeviceType::kCPU, T, K> final {
  static void Encode(DeviceCtx* ctx, const K* indices, int64_t num_indices, int64_t depth, T* out);
};

template<typename T, typename K>
void OneHotKernelUtil<DeviceType::kCPU, T, K>::Encode(DeviceCtx* ctx, const K* indices,
                                                      int64_t num_indices, int64_t depth, T* out) {
  Memset<kCPU>(ctx, out, 0, num_indices * depth * sizeof(T));
  FOR_RANGE(int64_t, i, 0, num_indices) {
    const K idx = indices[i];
    CHECK_GE(idx, 0);
    CHECK_LT(idx, depth);
    out[i * depth + idx] = GetOneVal<T>();
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kOneHotConf, OneHotKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
