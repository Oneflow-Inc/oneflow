#include "oneflow/core/kernel/concat_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void ConcatKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const std::vector<std::string>& ibns = op()->input_bns();
  if (ibns.size() == 0) return;
  Blob* out_blob = BnInOp2BlobPtr(op()->SoleObn());
  const int32_t concat_axis = op()->op_conf().concat_conf().axis();
  // TODO, concat_axis might be negative.
  const int64_t concat_num_each_blob = out_blob->shape().Count(0, concat_axis);
  const int64_t concat_element_size = out_blob->shape().Count(concat_axis + 1);
  const int64_t out_concat_axis_size = out_blob->shape().At(concat_axis);
  int64_t offset_concat_axis = 0;
  for (size_t ibn_idx = 0; ibn_idx < ibns.size(); ++ibn_idx) {
    const Blob* ibn_blob = BnInOp2BlobPtr(ibns[ibn_idx]);
    const int64_t in_concat_axis_size = ibn_blob->shape().At(concat_axis);
    for (int64_t concat_idx = 0; concat_idx < concat_num_each_blob;
         ++concat_idx) {
      KernelUtil<device_type, FloatingPointType>::Memcpy(
          ctx,
          (static_cast<FloatingPointType*>(out_blob->mut_dptr()))
              + (concat_idx * out_concat_axis_size + offset_concat_axis)
                    * concat_element_size,
          (static_cast<const FloatingPointType*>(ibn_blob->dptr()))
              + concat_idx * in_concat_axis_size * concat_element_size,
          in_concat_axis_size * concat_element_size
              * sizeof(FloatingPointType));
    }
    offset_concat_axis += in_concat_axis_size;
  }
}

template<DeviceType device_type, typename FloatingPointType>
void ConcatKernel<device_type, FloatingPointType>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* odbn_blob = BnInOp2BlobPtr(op()->SoleOdbn());
  const std::vector<std::string>& idbns = op()->input_diff_bns();
  const int32_t split_axis = op()->op_conf().concat_conf().axis();
  // TODO, split_axis might be negative.
  const int64_t split_num_each_blob = odbn_blob->shape().Count(0, split_axis);
  const int64_t split_element_size = odbn_blob->shape().Count(split_axis + 1);
  const int64_t out_diff_split_axis_size = odbn_blob->shape().At(split_axis);
  int64_t offset_split_axis = 0;
  for (size_t idbns_idx = 0; idbns_idx < idbns.size(); ++idbns_idx) {
    Blob* idbn_blob = BnInOp2BlobPtr(idbns[idbns_idx]);
    const int64_t in_diff_split_axis_size = idbn_blob->shape().At(split_axis);
    for (int64_t split_idx = 0; split_idx < split_num_each_blob; ++split_idx) {
      KernelUtil<device_type, FloatingPointType>::Memcpy(
          ctx,
          static_cast<FloatingPointType*>(idbn_blob->mut_dptr())
              + split_idx * in_diff_split_axis_size * split_element_size,
          static_cast<const FloatingPointType*>(odbn_blob->dptr())
              + (split_idx * out_diff_split_axis_size + offset_split_axis)
                    * split_element_size,
          in_diff_split_axis_size * split_element_size
              * sizeof(FloatingPointType));
    }
    offset_split_axis += in_diff_split_axis_size;
  }
}

INSTANTIATE_KERNEL_CLASS(ConcatKernel);
REGISTER_KERNEL(OperatorConf::kConcatConf, ConcatKernel);

}  // namespace oneflow
