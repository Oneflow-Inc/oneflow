#include "oneflow/core/kernel/proposal_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"

namespace oneflow {

template<typename T>
void ProposalKernel<T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  FasterRcnnUtil<T>::GenerateAnchors(op_conf().proposal_conf().anchors_generator_conf(),
                                     BnInOp2Blob("anchors"));
}

template<typename T>
void ProposalKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // input blob
  const Blob* bbox_pred_blob = BnInOp2Blob("bbox_pred");
  const Blob* class_prob_blob = BnInOp2Blob("class_prob");
  // output blob
  Blob* rois_blob = BnInOp2Blob("rois");
  Blob* roi_probs_blob = BnInOp2Blob("roi_probs");
  Blob* proposals_blob = BnInOp2Blob("proposals");
  const ProposalOpConf& conf = op_conf().proposal_conf();
  const AnchorGeneratorConf& anchor_generator_conf = conf.anchors_generator_conf();
  const BBoxRegressionWeights& bbox_reg_ws = conf.bbox_reg_weights();
  const int64_t num_images = class_prob_blob->shape().At(0);
  const int64_t height = class_prob_blob->shape().At(1);
  const int64_t width = class_prob_blob->shape().At(2);
  const int64_t num_anchors =
      anchor_generator_conf.aspect_ratios_size() * anchor_generator_conf.anchor_scales_size();
  const int64_t num_proposals = height * width * num_anchors;

  const T* anchors_ptr = BnInOp2Blob("anchors")->dptr<T>();
  const T* const_proposals_ptr = proposals_blob->dptr<T>();
  T* proposals_ptr = proposals_blob->mut_dptr<T>();
  int32_t* pre_nms_slice_ptr = BnInOp2Blob("pre_nms_slice")->mut_dptr<int32_t>();
  int32_t* post_nms_slice_ptr = BnInOp2Blob("post_nms_slice")->mut_dptr<int32_t>();
  FOR_RANGE(int64_t, i, 0, num_images) {
    const T* bbox_pred_ptr = bbox_pred_blob->dptr<T>(i);
    const T* class_prob_ptr = class_prob_blob->dptr<T>(i);

    FasterRcnnUtil<T>::BboxTransform(num_proposals, anchors_ptr, bbox_pred_ptr, bbox_reg_ws,
                                     proposals_ptr);
    FasterRcnnUtil<T>::ClipBoxes(num_proposals, anchor_generator_conf.image_height(),
                                 anchor_generator_conf.image_width(), proposals_ptr);

    ScoredBBoxSlice<T> pre_nms_slice(num_proposals, const_proposals_ptr, class_prob_ptr,
                                     pre_nms_slice_ptr);
    pre_nms_slice.DescSortByScore();
    pre_nms_slice.Filter([&](const T score, const BBox<T>* bbox) {
      return (bbox->width() < conf.min_size()) || (bbox->height() < conf.min_size());
    });
    pre_nms_slice.Truncate(conf.pre_nms_top_n());

    ScoredBBoxSlice<T> post_nms_slice(conf.post_nms_top_n(), const_proposals_ptr, class_prob_ptr,
                                      post_nms_slice_ptr);
    post_nms_slice.NmsFrom(conf.nms_threshold(), pre_nms_slice);

    CopyRoI(i, post_nms_slice, rois_blob);
    FOR_RANGE(int32_t, j, 0, post_nms_slice.available_len()) {
      roi_probs_blob->mut_dptr<T>(i)[j] = post_nms_slice.GetScore(j);
    }
  }
}

template<typename T>
void ProposalKernel<T>::CopyRoI(const int64_t im_index, const ScoredBBoxSlice<T>& slice,
                                Blob* rois_blob) const {
  FOR_RANGE(int32_t, i, 0, slice.available_len()) {
    BBox<T>* roi_bbox = BBox<T>::MutCast(rois_blob->mut_dptr<T>(im_index, i));
    const BBox<T>* proposal_bbox = slice.GetBBox(i);
    roi_bbox->set_x1(proposal_bbox->x1());
    roi_bbox->set_y1(proposal_bbox->y1());
    roi_bbox->set_x2(proposal_bbox->x2());
    roi_bbox->set_y2(proposal_bbox->y2());
  }
}

template<typename T>
void ProposalKernel<T>::ForwardDataId(const KernelCtx& ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("rois")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("bbox_pred"));
  BnInOp2Blob("roi_probs")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("bbox_pred"));
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kProposalConf, ProposalKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
