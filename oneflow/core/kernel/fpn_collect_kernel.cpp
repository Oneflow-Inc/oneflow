#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void FpnCollectKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  //to do :
  //       *in one function?
  //       *repeated blob?
  
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

  //sort method :nth element + sort
  
}



ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kFpnCollectConf, FpnCollectKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
