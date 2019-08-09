#include "oneflow/core/kernel/batch_gather_kernel.h"

namespace oneflow {

namespace {

Shape GetFlatShape(const Shape& shape, const int64_t axis) {
  CHECK_GT(shape.NumAxes(), 0);
  CHECK_GE(axis, 0);
  CHECK_LT(axis, shape.NumAxes());
  return Shape({shape.Count(0, axis), shape.At(axis), shape.Count(axis + 1)});
}

template<DeviceType device_type, typename T, typename K>
void BatchGatherForward(DeviceCtx* ctx, const Blob* in, const Blob* indices, Blob* out) {
  const int64_t axis = indices->shape().NumAxes() - 1;
  const Shape flat_out_shape = GetFlatShape(out->shape(), axis);
  BatchGatherKernelUtil<device_type, T, K>::Forward(ctx, in->dptr<T>(), indices->dptr<K>(),
                                                    flat_out_shape, in->shape().At(axis),
                                                    out->mut_dptr<T>());
}

template<DeviceType device_type, typename T, typename K>
void BatchGatherBackward(DeviceCtx* ctx, const Blob* out_diff, const Blob* indices, Blob* in_diff) {
  Memset<device_type>(ctx, in_diff->mut_dptr<T>(), 0, in_diff->ByteSizeOfDataContentField());
  const int64_t axis = indices->shape().NumAxes() - 1;
  const Shape flat_out_diff_shape = GetFlatShape(out_diff->shape(), axis);
  BatchGatherKernelUtil<device_type, T, K>::Backward(ctx, out_diff->dptr<T>(), indices->dptr<K>(),
                                                     flat_out_diff_shape, in_diff->shape().At(axis),
                                                     in_diff->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
struct BatchGatherSwitchUtil final {
#define MAKE_BATCH_GATHER_SWITCH_ENTRY(func_name, K) func_name<device_type, T, K>
#define DEFINE_BATCH_GATHER_STATIC_SWITCH_FUNC(func_name)                    \
  DEFINE_STATIC_SWITCH_FUNC(void, func_name, MAKE_BATCH_GATHER_SWITCH_ENTRY, \
                            MAKE_DATA_TYPE_CTRV_SEQ(INT_DATA_TYPE_SEQ));
  DEFINE_BATCH_GATHER_STATIC_SWITCH_FUNC(BatchGatherForward);
  DEFINE_BATCH_GATHER_STATIC_SWITCH_FUNC(BatchGatherBackward);
#undef DEFINE_BATCH_GATHER_STATIC_SWITCH_FUNC
#undef MAKE_BATCH_GATHER_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type, typename T>
const PbMessage& BatchGatherKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().batch_gather_conf();
}

template<DeviceType device_type, typename T>
void BatchGatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* indices_blob = BnInOp2Blob("indices");
  const std::vector<int64_t>& in_dim_vec = in_blob->shape().dim_vec();
  const std::vector<int64_t>& indices_dim_vec = indices_blob->shape().dim_vec();
  CHECK_LE(indices_dim_vec.size(), in_dim_vec.size());
  FOR_RANGE(int64_t, i, 0, indices_dim_vec.size() - 1) {
    CHECK_EQ(indices_dim_vec.at(i), in_dim_vec.at(i));
  }
  BatchGatherSwitchUtil<device_type, T>::SwitchBatchGatherForward(
      SwitchCase(BnInOp2Blob("indices")->data_type()), ctx.device_ctx, BnInOp2Blob("in"),
      BnInOp2Blob("indices"), BnInOp2Blob("out"));
}

template<DeviceType device_type, typename T>
void BatchGatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BatchGatherSwitchUtil<device_type, T>::SwitchBatchGatherBackward(
      SwitchCase(BnInOp2Blob("indices")->data_type()), ctx.device_ctx,
      BnInOp2Blob(GenDiffBn("out")), BnInOp2Blob("indices"), BnInOp2Blob(GenDiffBn("in")));
}

template<DeviceType device_type, typename T>
void BatchGatherKernel<device_type, T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("out")->set_dim0_valid_num(0, BnInOp2Blob("in")->shape().At(0));
}

template<DeviceType device_type, typename T>
void BatchGatherKernel<device_type, T>::ForwardInstanceShape(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const std::vector<int64_t> in_dim_vec = in_blob->shape().dim_vec();
  const Blob* indices_blob = BnInOp2Blob("indices");
  const std::vector<int64_t> indices_dim_vec = indices_blob->shape().dim_vec();

  std::vector<int64_t> out_dim_vec(indices_dim_vec);
  out_dim_vec.insert(out_dim_vec.end(), in_dim_vec.begin() + out_dim_vec.size(), in_dim_vec.end());
  BnInOp2Blob("out")->set_instance_shape(
      Shape(std::vector<int64_t>(out_dim_vec.begin() + 1, out_dim_vec.end())));
}

template<typename T, typename K>
struct BatchGatherKernelUtil<DeviceType::kCPU, T, K> final {
  static void Forward(DeviceCtx* ctx, const T* in, const K* indices, const Shape& flat_out_shape,
                      const int64_t gather_dim_size, T* out);
  static void Backward(DeviceCtx* ctx, const T* out_diff, const K* indices,
                       const Shape& flat_out_diff_shape, const int64_t gather_dim_size, T* in_diff);
};

template<typename T, typename K>
void BatchGatherKernelUtil<DeviceType::kCPU, T, K>::Forward(DeviceCtx* ctx, const T* in,
                                                            const K* indices,
                                                            const Shape& flat_out_shape,
                                                            const int64_t gather_dim_size, T* out) {
  const int64_t batch_num = flat_out_shape.At(0);
  const int64_t indices_num = flat_out_shape.At(1);
  const int64_t instance_size = flat_out_shape.At(2);
  FOR_RANGE(int64_t, batch_idx, 0, batch_num) {
    FOR_RANGE(int64_t, i, 0, indices_num) {
      const K idx = indices[batch_idx * indices_num + i];
      CHECK(idx >= 0 && idx < gather_dim_size);
      const T* from = in + batch_idx * gather_dim_size * instance_size + idx * instance_size;
      T* to = out + batch_idx * indices_num * instance_size + i * instance_size;
      std::copy(from, from + instance_size, to);
    }
  }
}

template<typename T, typename K>
void BatchGatherKernelUtil<DeviceType::kCPU, T, K>::Backward(DeviceCtx* ctx, const T* out_diff,
                                                             const K* indices,
                                                             const Shape& flat_out_diff_shape,
                                                             const int64_t gather_dim_size,
                                                             T* in_diff) {
  const int64_t batch_num = flat_out_diff_shape.At(0);
  const int64_t indices_num = flat_out_diff_shape.At(1);
  const int64_t instance_size = flat_out_diff_shape.At(2);
  FOR_RANGE(int64_t, batch_idx, 0, batch_num) {
    FOR_RANGE(int64_t, i, 0, indices_num) {
      const int64_t idx = indices[batch_idx * indices_num + i];
      CHECK(idx >= 0 && idx < gather_dim_size);
      const T* from = out_diff + batch_idx * indices_num * instance_size + i * instance_size;
      T* to = in_diff + batch_idx * gather_dim_size * instance_size + idx * instance_size;
      std::transform(from, from + instance_size, to, to, std::plus<T>());
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBatchGatherConf, BatchGatherKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
