#include "oneflow/core/kernel/concat_kernel.h"

namespace oneflow {

namespace {

int64_t Count2LastDim(const int32_t axis, Blob* blob_tmp) {
  if ((axis != (blob_tmp->shape().NumAxes() - 1)) && (axis != -1)) {
    return blob_tmp->shape().Count(axis + 1);
  } else {
    return 1;
  }
}

int64_t CountFrom1stDim(const int32_t axis, Blob* blob_tmp) {
  if ((axis != (-blob_tmp->shape().NumAxes())) && (axis != 0)) {
    return blob_tmp->shape().Count(0, axis);
  } else {
    return 1;
  }
}

template<typename FloatingPointType>
FloatingPointType* CalcCopyAddr(FloatingPointType* start_addr,
                                const int64_t cp_times, const int64_t axis_dim,
                                const int64_t offset, const int64_t elem_cnt) {
  return start_addr + (cp_times * axis_dim + offset) * elem_cnt;
}

}  // namespace

template<DeviceType device_type, typename FloatingPointType>
void ConcatKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const std::vector<std::string>& ibns = op()->input_bns();
  if (ibns.size() == 0) { return; }
  Blob* out_blob = BnInOp2BlobPtr(op()->SoleObn());
  const int32_t concat_axis = op()->op_conf().concat_conf().axis();
  const int64_t concat_element_cnt = Count2LastDim(concat_axis, out_blob);
  const int64_t concat_num_each_blob = CountFrom1stDim(concat_axis, out_blob);
  const int64_t out_concat_axis_dim = out_blob->shape().At(concat_axis);
  FloatingPointType* obn_dptr = out_blob->mut_dptr<FloatingPointType>();
  int64_t offset_concat_axis = 0;
  cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyHostToHost;
  if (device_type == DeviceType::kGPU) {
    kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
  }

  for (const std::string& ibn : ibns) {
    const Blob* ibn_blob = BnInOp2BlobPtr(ibn);
    const FloatingPointType* ibn_dptr = ibn_blob->dptr<FloatingPointType>();
    const int64_t in_concat_axis_dim = ibn_blob->shape().At(concat_axis);
    const int64_t cp_size =
        in_concat_axis_dim * concat_element_cnt * sizeof(FloatingPointType);

    for (int64_t concat_idx = 0; concat_idx < concat_num_each_blob;
         ++concat_idx) {
      FloatingPointType* obn_dst_adr =
          CalcCopyAddr(obn_dptr, concat_idx, out_concat_axis_dim,
                       offset_concat_axis, concat_element_cnt);
      const FloatingPointType* ibn_src_adr = CalcCopyAddr(
          ibn_dptr, concat_idx, in_concat_axis_dim, 0, concat_element_cnt);
      KernelUtil<device_type, FloatingPointType>::Memcpy(
          ctx, obn_dst_adr, ibn_src_adr, cp_size, kind);
    }

    offset_concat_axis += in_concat_axis_dim;
  }
}

template<DeviceType device_type, typename FloatingPointType>
void ConcatKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* odbn_blob = BnInOp2BlobPtr(op()->SoleOdbn());
  const std::vector<std::string>& idbns = op()->input_diff_bns();
  const int32_t split_axis = op()->op_conf().concat_conf().axis();
  const int64_t split_element_cnt = Count2LastDim(split_axis, odbn_blob);
  const int64_t split_num_each_blob = CountFrom1stDim(split_axis, odbn_blob);
  const int64_t out_diff_split_axis_dim = odbn_blob->shape().At(split_axis);
  const FloatingPointType* odbn_dptr = odbn_blob->dptr<FloatingPointType>();
  int64_t offset_split_axis = 0;
  cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyHostToHost;
  if (device_type == DeviceType::kGPU) {
    kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
  }

  for (const std::string& idbn : idbns) {
    Blob* idbn_blob = BnInOp2BlobPtr(idbn);
    FloatingPointType* idbn_dptr = idbn_blob->mut_dptr<FloatingPointType>();
    const int64_t in_diff_split_axis_dim = idbn_blob->shape().At(split_axis);
    const int64_t cp_size =
        in_diff_split_axis_dim * split_element_cnt * sizeof(FloatingPointType);

    for (int64_t split_idx = 0; split_idx < split_num_each_blob; ++split_idx) {
      const FloatingPointType* odbn_src_adr =
          CalcCopyAddr(odbn_dptr, split_idx, out_diff_split_axis_dim,
                       offset_split_axis, split_element_cnt);
      FloatingPointType* idbn_dst_adr = CalcCopyAddr(
          idbn_dptr, split_idx, in_diff_split_axis_dim, 0, split_element_cnt);
      KernelUtil<device_type, FloatingPointType>::Memcpy(
          ctx, idbn_dst_adr, odbn_src_adr, cp_size, kind);
    }

    offset_split_axis += in_diff_split_axis_dim;
  }
}

INSTANTIATE_KERNEL_CLASS(ConcatKernel);
REGISTER_KERNEL(OperatorConf::kConcatConf, ConcatKernel);

}  // namespace oneflow
