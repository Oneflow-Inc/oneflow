#include "oneflow/core/kernel/concat_gt_proposals_kernel.h"

namespace oneflow {

template<typename T>
void ConcatGtProposalsKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void ConcatGtProposalsKernel<T>::ForwardDim1ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void ConcatGtProposalsKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void ConcatGtProposalsKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_proposals_blob = BnInOp2Blob("in");
  const Blob* gt_boxes_blob = BnInOp2Blob("gt_boxes");
  Blob* out_proposals_blob = BnInOp2Blob("out");
  const bool has_record_id_header = out_proposals_blob->has_record_id_in_device_piece_field();

  CHECK_GE(out_proposals_blob->ByteSizeOfBlobHeader(), in_proposals_blob->ByteSizeOfBlobHeader());
  Memcpy<DeviceType::kCPU>(ctx.device_ctx, out_proposals_blob->mut_header_ptr(),
                           in_proposals_blob->header_ptr(),
                           in_proposals_blob->ByteSizeOfBlobHeader());
  out_proposals_blob->CopyDataContentFrom(ctx.device_ctx, in_proposals_blob);

  auto* out_proposals_bbox_ptr = BBox::Cast(out_proposals_blob->mut_dptr<T>());
  auto* gt_bbox_ptr = GtBBox::Cast(gt_boxes_blob->dptr<T>());
  const size_t max_num_gt_boxes_per_im = gt_boxes_blob->static_shape().At(1);
  size_t num_proposals = out_proposals_blob->dim0_valid_num(0);
  FOR_RANGE(int32_t, i, 0, gt_boxes_blob->static_shape().At(0)) {
    FOR_RANGE(int32_t, j, 0, gt_boxes_blob->dim1_valid_num(i)) {
      auto* cur_gt_bbox_ptr = gt_bbox_ptr + (i * max_num_gt_boxes_per_im + j);
      out_proposals_bbox_ptr[num_proposals].set_ltrb(
          cur_gt_bbox_ptr->left(), cur_gt_bbox_ptr->top(), cur_gt_bbox_ptr->right(),
          cur_gt_bbox_ptr->bottom());
      out_proposals_bbox_ptr[num_proposals].set_index(i);
      if (has_record_id_header) {
        out_proposals_blob->set_record_id_in_device_piece(num_proposals, i);
      }
      num_proposals += 1;
    }
  }
  CHECK_LE(num_proposals, out_proposals_blob->static_shape().At(0));
  out_proposals_blob->set_dim0_valid_num(0, num_proposals);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kConcatGtProposalsConf, ConcatGtProposalsKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
