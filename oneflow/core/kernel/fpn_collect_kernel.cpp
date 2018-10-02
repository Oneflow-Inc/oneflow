#include "oneflow/core/kernel/fpn_collect_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void FpnCollectKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // to do :
  //       *better sort method :nth element + sort

  const int32_t level = this->op_conf().fpn_collect_conf().level();
  const int32_t post_nms_topn = this->op_conf().fpn_collect_conf().post_nms_top_n();
  ConcatAllRoisAndScores(ctx, level, BnInOp2Blob);
  SortAndSelectTopnRois(post_nms_topn, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
void FpnCollectKernel<device_type, T>::ConcatAllRoisAndScores(
    const KernelCtx& ctx, const int32_t level,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* roi_inputs_blob = BnInOp2Blob("roi_inputs");
  Blob* score_inputs_blob = BnInOp2Blob("score_inputs");
  int64_t N = roi_inputs_blob->shape().At(0);
  int64_t R = roi_inputs_blob->shape().At(1);
  const int64_t row_num = N * R;
  const int64_t roi_out_col_num = roi_inputs_blob->shape().Count(1);
  const int64_t score_out_col_num = score_inputs_blob->shape().Count(1);
  int64_t roi_col_offset = 0;
  int64_t prob_col_offset = 0;

  for (int32_t i = 0; i < level; i++) {
    std::string roi_bn = "rpn_rois_fpn" + std::to_string(i);
    std::string prob_bn = "rpn_roi_probs_fpn" + std::to_string(i);

    const Blob* roi_blob = BnInOp2Blob(roi_bn);
    const Blob* prob_blob = BnInOp2Blob(prob_bn);
    const int64_t roi_in_col_num = roi_blob->shape().Count(1);
    const int64_t prob_in_col_num = prob_blob->shape().Count(1);

    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, roi_in_col_num, roi_blob->dptr<T>(), 0, roi_in_col_num,
        roi_inputs_blob->mut_dptr<T>(), roi_col_offset, roi_out_col_num);
    roi_col_offset += roi_in_col_num;

    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, prob_in_col_num, prob_blob->dptr<T>(), 0, prob_in_col_num,
        score_inputs_blob->mut_dptr<T>(), prob_col_offset, score_out_col_num);
    prob_col_offset += roi_in_col_num;
  }
}

template<DeviceType device_type, typename T>
void FpnCollectKernel<device_type, T>::SortAndSelectTopnRois(
    const int32_t topn, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* roi_inputs_blob = BnInOp2Blob("roi_inputs");
  const Blob* score_inputs_blob = BnInOp2Blob("score_inputs");
  Blob* index_blob = BnInOp2Blob("index");
  Blob* out_blob = BnInOp2Blob("out");
  size_t index_size = roi_inputs_blob->shape().At(0) * roi_inputs_blob->shape().At(1);
  auto scored_index = GenScoresIndex(index_size, index_blob->mut_dptr<int32_t>(),
                                     score_inputs_blob->dptr<T>(), false);
  scored_index.SortByScore([](T lhs_score, T rhs_score) { return lhs_score > rhs_score; });
  for (int64_t i = 0; i < topn; i++) {
    const size_t si = scored_rois.GetIndex(i);
    for (int64_t j = 0; j < 5; j++) { 
        out_blob->mut_dptr<T>()[i * 5 + j] = roi_inputs_blob->dptr<T>()[si * 5 + j]; }
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnCollectConf, FpnCollectKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
