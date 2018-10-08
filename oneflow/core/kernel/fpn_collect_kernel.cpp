#include "oneflow/core/kernel/fpn_collect_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void FpnCollectKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t level = this->op_conf().fpn_collect_conf().level();
  const int64_t post_nms_topn = this->op_conf().fpn_collect_conf().post_nms_top_n();
  int64_t available_num = ConcatAllRoisAndScores(ctx, level, BnInOp2Blob);
  if(available_num < post_nms_topn){post_nms_topn = available_num;}
  SortAndSelectTopnRois(post_nms_topn, BnInOp2Blob);
}

template<DeviceType device_type, typename T>
int64_t FpnCollectKernel<device_type, T>::ConcatAllRoisAndScores(
    const KernelCtx& ctx, const int32_t level,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* roi_inputs_blob = BnInOp2Blob("roi_inputs");
  Blob* score_inputs_blob = BnInOp2Blob("score_inputs");
  const int64_t row_num = roi_inputs_blob->shape().At(0);
  const int64_t roi_out_col_num = roi_inputs_blob->shape().Count(1);
  const int64_t score_out_col_num = score_inputs_blob->shape().Count(1);
  int64_t roi_col_offset = 0;
  int64_t prob_col_offset = 0;
  int64_t available_roi_num = 0;
  int64_t available_prob_num = 0;


  FOR_RANGE(size_t, i, 0, level) {
    std::string roi_bn = "rpn_rois_fpn_" + std::to_string(i);
    std::string prob_bn = "rpn_roi_probs_fpn_" + std::to_string(i);

    const Blob* roi_blob = BnInOp2Blob(roi_bn);
    const Blob* prob_blob = BnInOp2Blob(prob_bn);
    const int64_t roi_in_col_num = roi_blob->shape().Count(1);
    const int64_t prob_in_col_num = prob_blob->shape().Count(1);
    
    available_roi_num += roi_blob->instance_available_elem_cnt();
    available_prob_num += roi_blob->instance_available_elem_cnt();
    CHECK_EQ(available_roi_num, available_prob_num);    
 
    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, roi_in_col_num, roi_blob->dptr<T>(), 0, roi_in_col_num,
        roi_inputs_blob->mut_dptr<T>(), roi_col_offset, roi_out_col_num);
    roi_col_offset += roi_in_col_num;

    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, prob_in_col_num, prob_blob->dptr<T>(), 0, prob_in_col_num,
        score_inputs_blob->mut_dptr<T>(), prob_col_offset, score_out_col_num);
    prob_col_offset += prob_in_col_num;
  }
  LOG(INFO) << "TEST COLLECT BREAK POINT 1";
  return available_roi_num;
}

template<DeviceType device_type, typename T>
void FpnCollectKernel<device_type, T>::SortAndSelectTopnRois(
    const size_t topn, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* roi_inputs_blob = BnInOp2Blob("roi_inputs");
  const Blob* score_inputs_blob = BnInOp2Blob("score_inputs");
  Blob* index_blob = BnInOp2Blob("index");
  Blob* out_blob = BnInOp2Blob("out");
  size_t index_size = roi_inputs_blob->shape().At(0) * roi_inputs_blob->shape().At(1);
  auto scored_index = GenScoresIndex(index_size, index_blob->mut_dptr<int32_t>(),
                                     score_inputs_blob->dptr<T>(), true);

  auto comp = [](T lhs_score, T rhs_score) { return lhs_score > rhs_score; };
  scored_index.NthElementByScore(topn, comp);
  scored_index.Truncate(topn);
  scored_index.SortByScore(comp);

  FOR_RANGE(size_t, i, 0, topn) {
    const size_t si = scored_index.GetIndex(i);
    FOR_RANGE(size_t, j, 0, 5) {
      out_blob->mut_dptr<T>()[i * 5 + j] = roi_inputs_blob->dptr<T>()[si * 5 + j];
    }
  }
  LOG(INFO) << "TEST COLLECT BREAK POINT 2";
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnCollectConf, FpnCollectKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
