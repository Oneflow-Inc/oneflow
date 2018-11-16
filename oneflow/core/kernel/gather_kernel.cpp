#include "oneflow/core/kernel/gather_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& GatherKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gather_conf();
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t axis = this->op_conf().gather_conf().axis();
  const Blob* indices_blob = BnInOp2Blob("indices");
  const int64_t num_indices = indices_blob->shape().elem_cnt();
  const Blob* in_blob = BnInOp2Blob("in");
  const int64_t in_rows = in_blob->shape().At(axis);
  const int64_t in_cols = in_blob->shape().Count(axis + 1);
  const int64_t in_blocks = in_blob->shape().Count(0, axis);
  Blob* out = BnInOp2Blob("out");
  switch (indices_blob->data_type()) {
#define LOOKUP_FORWARD_CASE(type_cpp, type_proto)                                        \
  case type_proto:                                                                       \
    LookupKernelUtil<device_type, T, type_cpp>::Forward(                                 \
        ctx.device_ctx, indices_blob->dptr<type_cpp>(), num_indices, in_blob->dptr<T>(), \
        in_blocks, in_rows, in_cols, out->mut_dptr<T>());                                \
    break;
    OF_PP_FOR_EACH_TUPLE(LOOKUP_FORWARD_CASE, INT_DATA_TYPE_SEQ)
#undef LOOKUP_FORWARD_CASE
    default: UNIMPLEMENTED();
  }
}

template<DeviceType device_type, typename T>
void GatherKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t axis = this->op_conf().gather_conf().axis();
  const Blob* indices_blob = BnInOp2Blob("indices");
  const int64_t num_indices = indices_blob->shape().elem_cnt();
  const Blob* out_diff_blob = BnInOp2Blob(GenDiffBn("out"));
  Blob* in_diff_blob = BnInOp2Blob(GenDiffBn("in"));
  const int64_t in_rows = in_diff_blob->shape().At(axis);
  const int64_t in_cols = in_diff_blob->shape().Count(axis + 1);
  const int64_t in_blocks = in_diff_blob->shape().Count(0, axis);
  Memset<device_type>(ctx.device_ctx, in_diff_blob->mut_dptr<T>(), 0,
                      in_diff_blob->ByteSizeOfDataContentField());
  switch (indices_blob->data_type()) {
#define LOOKUP_BACKWARD_CASE(type_cpp, type_proto)                                             \
  case type_proto:                                                                             \
    LookupKernelUtil<device_type, T, type_cpp>::Backward(                                      \
        ctx.device_ctx, indices_blob->dptr<type_cpp>(), num_indices, out_diff_blob->dptr<T>(), \
        in_blocks, in_rows, in_cols, in_diff_blob->mut_dptr<T>());                             \
    break;
    OF_PP_FOR_EACH_TUPLE(LOOKUP_BACKWARD_CASE, INT_DATA_TYPE_SEQ)
#undef LOOKUP_BACKWARD_CASE
    default: UNIMPLEMENTED();
  }
}

template<typename T, typename IndexT>
struct LookupKernelUtil<DeviceType::kCPU, T, IndexT> final {
  static void Forward(DeviceCtx* ctx, const IndexT* indices, int64_t num_indices, const T* in,
                      int64_t in_blocks, int64_t in_rows, int64_t in_cols, T* out);
  static void Backward(DeviceCtx* ctx, const IndexT* indices, int64_t num_indices,
                       const T* out_diff, int64_t in_blocks, int64_t in_rows, int64_t in_cols,
                       T* in_diff);
};

template<typename T, typename IndexT>
void LookupKernelUtil<DeviceType::kCPU, T, IndexT>::Forward(DeviceCtx* ctx, const IndexT* indices,
                                                            int64_t num_indices, const T* in,
                                                            int64_t in_blocks, int64_t in_rows,
                                                            int64_t in_cols, T* out) {
  FOR_RANGE(int64_t, b, 0, in_blocks) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = indices[i];
      CHECK(idx >= 0 && idx < in_rows);
      const T* from = in + b * in_rows * in_cols + idx * in_cols;
      T* to = out + b * num_indices * in_cols + i * in_cols;
      std::copy(from, from + in_cols, to);
    }
  }
}

template<typename T, typename IndexT>
void LookupKernelUtil<DeviceType::kCPU, T, IndexT>::Backward(DeviceCtx* ctx, const IndexT* indices,
                                                             int64_t num_indices, const T* out_diff,
                                                             int64_t in_blocks, int64_t in_rows,
                                                             int64_t in_cols, T* in_diff) {
  FOR_RANGE(int64_t, b, 0, in_blocks) {
    FOR_RANGE(int64_t, i, 0, num_indices) {
      const int64_t idx = indices[i];
      CHECK(idx >= 0 && idx < in_rows);
      const T* from = out_diff + b * num_indices * in_cols + i * in_cols;
      T* to = in_diff + b * in_rows * in_cols + idx * in_cols;
      std::transform(from, from + in_cols, to, to, std::plus<T>());
    }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kGatherConf, GatherKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
