#include "oneflow/core/kernel/concat_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
void ConcatKernel<device_type, FloatingPointType>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const std::vector<std::string>& ibns = op()->input_bns();
  if (ibns.size() == 0) { return; }
  Blob* out_blob = BnInOp2BlobPtr(op()->SoleObn());
  const int32_t concat_axis = op()->op_conf().concat_conf().axis();
  int64_t concat_element_size = 1;
  if (out_blob->shape().NumAxes() != (concat_axis + 1)) {
    concat_element_size = out_blob->shape().Count(concat_axis + 1);
  }
  const int64_t concat_num_each_blob = out_blob->shape().Count(0, concat_axis);
  const int64_t out_concat_axis_size = out_blob->shape().At(concat_axis);
  FloatingPointType* obn_dptr = out_blob->mut_dptr<FloatingPointType>();
  int64_t offset_concat_axis = 0;

  for (const std::string& ibn : ibns) {
    const Blob* ibn_blob = BnInOp2BlobPtr(ibn);
    const FloatingPointType* ibn_dptr = ibn_blob->dptr<FloatingPointType>();
    const int64_t in_concat_axis_size = ibn_blob->shape().At(concat_axis);
    const int64_t cp_size =
        in_concat_axis_size * concat_element_size * sizeof(FloatingPointType);

    for (int64_t concat_idx = 0; concat_idx < concat_num_each_blob;
         ++concat_idx) {
      FloatingPointType* obn_dst_adr =
          obn_dptr
          + (concat_idx * out_concat_axis_size + offset_concat_axis)
                * concat_element_size;
      const FloatingPointType* ibn_src_adr =
          ibn_dptr + concat_idx * in_concat_axis_size * concat_element_size;
      KernelUtil<device_type, FloatingPointType>::Memcpy(
          ctx, obn_dst_adr, ibn_src_adr, cp_size,
          cudaMemcpyKind::cudaMemcpyDeviceToDevice);
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
  int64_t split_element_size = 1;
  if (odbn_blob->shape().NumAxes() != (split_axis + 1)) {
    split_element_size = odbn_blob->shape().Count(split_axis + 1);
  }
  const int64_t split_num_each_blob = odbn_blob->shape().Count(0, split_axis);
  const int64_t out_diff_split_axis_size = odbn_blob->shape().At(split_axis);
  const FloatingPointType* odbn_dptr = odbn_blob->dptr<FloatingPointType>();
  int64_t offset_split_axis = 0;

  for (const std::string& idbn : idbns) {
    Blob* idbn_blob = BnInOp2BlobPtr(idbn);
    FloatingPointType* idbn_dptr = idbn_blob->mut_dptr<FloatingPointType>();
    const int64_t in_diff_split_axis_size = idbn_blob->shape().At(split_axis);
    const int64_t cp_size = in_diff_split_axis_size * split_element_size
                            * sizeof(FloatingPointType);

    for (int64_t split_idx = 0; split_idx < split_num_each_blob; ++split_idx) {
      const FloatingPointType* odbn_src_adr =
          odbn_dptr
          + (split_idx * out_diff_split_axis_size + offset_split_axis)
                * split_element_size;
      FloatingPointType* idbn_dst_adr =
          idbn_dptr + split_idx * in_diff_split_axis_size * split_element_size;
      KernelUtil<device_type, FloatingPointType>::Memcpy(
          ctx, idbn_dst_adr, odbn_src_adr, cp_size,
          cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    }

    offset_split_axis += in_diff_split_axis_size;
  }
}

INSTANTIATE_KERNEL_CLASS(ConcatKernel);
REGISTER_KERNEL(OperatorConf::kConcatConf, ConcatKernel);

}  // namespace oneflow
