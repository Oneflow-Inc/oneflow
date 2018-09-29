#include "oneflow/core/kernel/fpn_collect_kernel.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void FpnCollectKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  //to do :
  //       *in two function
  //       *repeated blob?
  //       *best sort method :nth element + sort
  //       *blob shape
  
  
  //1. concat all rois and probs
  level = this->kernel_conf().fpn_collect_conf().level();
  Blob* roi_inputs_blob = BnInOp2Blob("roi_inputs");
  Blob* score_inputs_blob = BnInOp2Blob("score_inputs");
  const int64_t row_num = roi_inputs_blob->shape().At(0);
  const int64_t roi_out_col_num = roi_inputs_blob->shape().Count(0);
  const int64_t score_out_col_num = score_inputs_blob->shape().Count(0);
  int64_t roi_col_offset = 0;
  int64_t prob_col_offset = 0;
  for (int32_t i = 2; i <= level; i++){
    std::string roi_bn = "rpn_rois_fpn_" + std::to_string(i);
    std::string prob_bn = "rpn_roi_probs_fpn_" + std::to_string(i);
    const Blob* roi_blob = BnInOp2Blob(roi_bn);
    const Blob* prob_blob = BnInOp2Blob(prob_bn);
    const int64_t roi_in_col_num = roi_blob->shape().Count(0);
    const int64_t prob_in_col_num = prob_blob->shape().Count(0);

    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, roi_in_col_num, roi_blob->dptr<T>(), 0, roi_in_col_num,
        roi_inputs_blob->mut_dptr<T>(), roi_col_offset, roi_col_num);
    roi_col_offset += roi_in_col_num;

    KernelUtil<device_type, T>::CopyColsRegion(
        ctx.device_ctx, row_num, prob_in_col_num, prob_blob->dptr<T>(), 0, prob_in_col_num,
        score_inputs_blob->mut_dptr<T>(), prob_col_offset, out_col_num);
    prob_col_offset += roi_in_col_num;
  }
  //2. sort and select topn
  post_nms_topn = this->kernel_conf().fpn_collect_conf().post_nms_topn();
  Blob* index_blob = BnInOp2Blob("index");
  Blob* out_blob = BnInOp2Blob("box");
  index_ptr = index_blob->dptr<T>();
  score_ptr = score_inputs_blob->mut_dptr<T>();
  scoreindex = ScoresIndex<T>(Indexes(row_num, index_ptr, init_index = false),score_ptr);
  scoreindex.SortByScore([](T lhs_score, T rhs_score) { return lhs_score > rhs_score; });
  for(int32_t i = 0; i < post_nms_topn ; i++){
    si=scoreindex.GetIndex(i);
    for(int32_t j = 0 ; j < 5 ; j++)
      out_blob->mut_dptr<T>()[i * 5 + j] = roi_inputs_blob->dptr<T>()[si * 5 + j];
  }
  
}



ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnCollectConf, FpnCollectKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
