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
const PbMessage& GatherKernel<device_type, T>::GetCustomizedOpConf() const {
  if (this->op_conf().has_gather_conf()) {
    return this->op_conf().gather_conf();
  } else if (this->op_conf().has_local_gather_conf()) {
    return this->op_conf().local_gather_conf();
  } else {
    UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GatherSwitchUtil<device_type, T>::SwitchGatherForward(
      SwitchCase(BnInOp2Blob("indices")->data_type()), ctx.device_ctx, BnInOp2Blob("indices"),
      BnInOp2Blob("in"), this->kernel_conf().gather_conf().axis(), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  GatherSwitchUtil<device_type, T>::SwitchGatherBackward(
      SwitchCase(BnInOp2Blob("indices")->data_type()), ctx.device_ctx, BnInOp2Blob("indices"),
      BnInOp2Blob(GenDiffBn("out")), this->kernel_conf().gather_conf().axis(),
      BnInOp2Blob(GenDiffBn("in")));
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* indices_blob = BnInOp2Blob("indices");
  Blob* out_blob = BnInOp2Blob("out");
  const int64_t axis = this->kernel_conf().local_gather_conf().axis();
  if (axis == 0 && indices_blob->has_dim0_valid_num_field()) {
    out_blob->CopyDim0ValidNumFrom(ctx.device_ctx, indices_blob);
  } else if (axis > 0 && in_blob->has_dim0_valid_num_field()) {
    out_blob->CopyDim0ValidNumFrom(ctx.device_ctx, in_blob);
  } else {
    UNIMPLEMENTED();
  }
}

template<typename T, typename K>
struct GatherKernelUtil<DeviceType::kCPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out);
  static void Backward(DeviceCtx* ctx, const K* indices, int64_t num_indices, const T* out_diff,
                       const Shape& flat_in_shape, T* in_diff);
};

template<typename T, typename K>
void GatherKernelUtil<DeviceType::kCPU, T, K>::Forward(DeviceCtx* ctx, const K* indices,
                                                       int64_t num_indices, const T* in,
                                                       const Shape& flat_in_shape, T* out) {
  const int64_t outer_dim_size = flat_in_shape.At(0);
  const int64_t gather_dim_size = flat_in_shape.At(1);
  const int64_t inner_dim_size = flat_in_shape.At(2);
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = indices[i];
      CHECK(idx >= 0 && idx < gather_dim_size);
      const T* from = in + outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size;
      T* to = out + outer_idx * num_indices * inner_dim_size + i * inner_dim_size;
      std::copy(from, from + inner_dim_size, to);
    }
  }
}

template<typename T, typename K>
void GatherKernelUtil<DeviceType::kCPU, T, K>::Backward(DeviceCtx* ctx, const K* indices,
                                                        int64_t num_indices, const T* out_diff,
                                                        const Shape& flat_in_shape, T* in_diff) {
  const int64_t outer_dim_size = flat_in_shape.At(0);
  const int64_t gather_dim_size = flat_in_shape.At(1);
  const int64_t inner_dim_size = flat_in_shape.At(2);
  FOR_RANGE(int64_t, outer_idx, 0, outer_dim_size) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = indices[i];
      CHECK(idx >= 0 && idx < gather_dim_size);
      const T* from = out_diff + outer_idx * num_indices * inner_dim_size + i * inner_dim_size;
      T* to = in_diff + outer_idx * gather_dim_size * inner_dim_size + idx * inner_dim_size;
      std::transform(from, from + inner_dim_size, to, to, std::plus<T>());
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel, FLOATING_DATA_TYPE_SEQ);
ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLocalGatherConf, GatherKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
