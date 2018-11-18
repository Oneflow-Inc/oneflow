#include "oneflow/core/kernel/one_hot_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& OneHotKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().one_hot_conf();
}

template<DeviceType device_type, typename T>
void OneHotKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* indices = BnInOp2Blob("indices");
  Blob* out = BnInOp2Blob("out");
  const int64_t num_indices = indices->shape().elem_cnt();
  const int64_t depth = out->shape().At(out->shape().NumAxes() - 1);
  switch (indices->data_type()) {
#define ONE_HOT_FORWARD_CASE(type_cpp, type_proto)                                                 \
  case type_proto:                                                                                 \
    OneHotKernelUtil<device_type, T, type_cpp>::Forward(ctx.device_ctx, indices->dptr<type_cpp>(), \
                                                        num_indices, depth, out->mut_dptr<T>());   \
    break;
    OF_PP_FOR_EACH_TUPLE(ONE_HOT_FORWARD_CASE, INT_DATA_TYPE_SEQ);
#undef ONE_HOT_FORWARD_CASE
    default: UNIMPLEMENTED();
  }
}

template<typename T, typename IndexT>
struct OneHotKernelUtil<DeviceType::kCPU, T, IndexT> final {
  static void Forward(DeviceCtx* ctx, const IndexT* indices, int64_t num_indices, int64_t depth,
                      T* out);
};

template<typename T, typename IndexT>
void OneHotKernelUtil<DeviceType::kCPU, T, IndexT>::Forward(DeviceCtx* ctx, const IndexT* indices,
                                                            int64_t num_indices, int64_t depth,
                                                            T* out) {
  Memset<kCPU>(ctx, out, 0, num_indices * depth * sizeof(T));
  FOR_RANGE(int64_t, i, 0, num_indices) {
    const IndexT idx = indices[i];
    CHECK_GE(idx, 0);
    CHECK_LT(idx, depth);
    out[i * depth + idx] = OneVal<T>::value;
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kOneHotConf, OneHotKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
