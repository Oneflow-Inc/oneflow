#include "oneflow/core/kernel/bbox_nms_and_limit_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include <math.h>

namespace oneflow {

namespace {

void IndexMemContinuous(int32_t* post_nms_index_slice_ptr, const int32_t class_num, const int32_t box_num, const int32_t* post_nms_keep_num_ptr) {
  int32_t keep_index = 0;
  FOR_RANGE(int32_t, i, 1, class_num) {
    FOR_RANGE(int32_t, j, 0, post_nms_keep_num_ptr[i]) {
      post_nms_index_slice_ptr[keep_index++] = post_nms_index_slice_ptr[i * box_num + j];
    }
  }
}

}  // namespace

template<typename T>
void BboxNmsAndLimitKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int32_t image_num = BnInOp2Blob("bbox")->shape().At(0);    
  FOR_RANGE(int32_t, i, 0, image_num) {
    BroadCastBboxTransform(i, BnInOp2Blob);
    ClipBox(BnInOp2Blob("bbox"));
    NmsAndTryVote(i, BnInOp2Blob);
    int32_t keep_num_per_im = Limit(i, BnInOp2Blob);
    WriteOutputToOFRecord(i, keep_num_per_im, BnInOp2Blob);
  }
}
template<typename T>
void BboxNmsAndLimitKernel<T>::SortClassBoxIndex(const T* score_ptr, int32_t* pre_nms_index_slice, int32_t box_num, int32_t class_num,int32_t class_index){
  FOR_RANGE(int32_t, i, 0, box_num){
    pre_nms_index_slice[i] = class_index + i * class_num;
  }
  std::sort(pre_nms_index_slice, pre_nms_index_slice + box_num,
            [&](int32_t lhs, int32_t rhs) { return score_ptr[lhs] > score_ptr[rhs]; });
}
template<typename T>
void BboxNmsAndLimitKernel<T>::BroadCastBboxTransform(const int32_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob){
  // bbox_delta_blob: (im, box_num, class*4)
  const Blob* bbox_delta_blob = BnInOp2Blob("bbox_delta");
  int32_t box_num = bbox_delta_blob->shape().At(1);
  int32_t class_num = bbox_delta_blob->shape().At(2) / 4;
  int32_t rois_offset_per_im = im_index * box_num * 4;
  int32_t delta_offset_per_im = im_index * box_num * class_num * 4;
  // rois_blob: (im, box_num, 4)
  const T* rois_ptr = BnInOp2Blob("rois")->dptr<T>() + rois_offset_per_im;
  const T* bbox_delta_ptr = BnInOp2Blob("bbox_delta")->dptr<T>() + delta_offset_per_im;
  // bbox_blob: (box_num, class*4)
  T* bbox_ptr = BnInOp2Blob("bbox")->mut_dptr<T>(); 
  FOR_RANGE(int32_t, i, 1, box_num) {
    int32_t rois_offset = i * 4;    
    float w = rois_ptr[rois_offset + 2] - rois_ptr[rois_offset + 0] + 1.0f;
    float h = rois_ptr[rois_offset + 3] - rois_ptr[rois_offset + 1] + 1.0f;
    float ctr_x = rois_ptr[rois_offset + 0] + 0.5f * w;
    float ctr_y = rois_ptr[rois_offset + 1] + 0.5f * h;
    FOR_RANGE(int32_t, j, 1, class_num) {
      int32_t delta_offset = (i * class_num + j) * 4;
      float pred_ctr_x = bbox_delta_ptr[delta_offset + 0] * w + ctr_x;
      float pred_ctr_y = bbox_delta_ptr[delta_offset + 1] * h + ctr_y;
      float pred_w = std::exp(bbox_delta_ptr[delta_offset + 2]) * w;
      float pred_h = std::exp(bbox_delta_ptr[delta_offset + 3]) * h;
      bbox_ptr[delta_offset + 0] = pred_ctr_x - 0.5f * pred_w;
      bbox_ptr[delta_offset + 1] = pred_ctr_y - 0.5f * pred_h;
      bbox_ptr[delta_offset + 2] = pred_ctr_x + 0.5f * pred_w - 1.f;
      bbox_ptr[delta_offset + 3] = pred_ctr_y + 0.5f * pred_h - 1.f;
    }
  }
}
template<typename T>
void BboxNmsAndLimitKernel<T>::ClipBox(Blob* bbox_blob){
  // bbox_blob: (box_num, class * 4)
  T* bbox_ptr = bbox_blob->mut_dptr<T>();
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  const int32_t bbox_buf_size = bbox_blob->shape().elem_cnt();
  for (int64_t i = 0; i < bbox_buf_size; i += 4) {
    bbox_ptr[i + 0] = std::max<T>(std::min<T>(bbox_ptr[i + 0], conf.image_width()), 0);
    bbox_ptr[i + 1] = std::max<T>(std::min<T>(bbox_ptr[i + 1], conf.image_height()), 0);
    bbox_ptr[i + 2] = std::max<T>(std::min<T>(bbox_ptr[i + 2], conf.image_width()), 0);
    bbox_ptr[i + 3] = std::max<T>(std::min<T>(bbox_ptr[i + 3], conf.image_height()), 0);
  }
}
template<typename T>
int32_t BboxNmsAndLimitKernel<T>::FilterSortedIndexByThreshold(const T* scores_ptr,const T score_thresh,const int32_t* pre_nms_index_slice,const int32_t num){
  FOR_RANGE(int32_t, i, 0, num){
    int32_t scores_index = pre_nms_index_slice[i];
    if(scores_ptr[scores_index] <= score_thresh){
      return i;
    }
  }
  return num;
}
template<typename T>
void BboxNmsAndLimitKernel<T>::NmsAndTryVote(int32_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob){
  int32_t score_offset = im_index * box_num * class_num;
  // scores_blob: (im, box_num, class_num)  pre_nms_index_slice_blob: (cls_num, roi)
  // post_nms_index_slice_blob: (cls_num, roi) post_nms_keep_num_blob: (cls_num) nms_area_tmp_blob: (cls_num)
  const Blob* scores_blob = BnInOp2Blob("scores");
  const T* scores_ptr = scores_blob->dptr<T>() + score_offset;  
  Blob* pre_nms_index_slice_blob = BnInOp2Blob("pre_nms_index_slice"); 
  Blob* post_nms_index_slice_blob = BnInOp2Blob("post_nms_index_slice"); 
  int32_t* post_nms_keep_num_ptr = BnInOp2Blob("post_nms_keep_num")->mut_dptr<int32_t>();
  int32_t* nms_area_tmp_ptr = BnInOp2Blob("nms_area_tmp")->mut_dptr<int32_t>();
  int32_t box_num = scores_blob->shape().At(1);
  int32_t class_num = scores_blob->shape().At(2);
  FOR_RANGE(int32_t, i, 1, class_num) {  // just foreground, class index start from 1
    int32_t index_offset = i * box_num;
    int32_t* pre_nms_index_slice_ptr = pre_nms_index_slice_blob->mut_dptr<int32_t>() + index_offset;
    int32_t* post_nms_index_slice_ptr = post_nms_index_slice_blob->mut_dptr<int32_t>() + index_offset;      
    SortClassBoxIndex(scores_ptr, pre_nms_index_slice_ptr, box_num, class_num, i);  
    int32_t pre_nms_keep_num = FilterSortedIndexByThreshold(scores_ptr, op_conf().bbox_nms_and_limit_conf().score_thresh(), pre_nms_index_slice_ptr, box_num);  
    post_nms_keep_num_ptr[i] = FasterRcnnUtil<T>::Nms(boxes_ptr, pre_nms_index_slice_ptr,
                               pre_nms_keep_num, pre_nms_keep_num, conf.nms_threshold(), nms_area_tmp_ptr,
                               post_nms_index_slice_ptr);
    if (conf.bbox_vote_enabled()) {
        BboxVoting();//todo
    }
  }
}
template<typename T>
int32_t BboxNmsAndLimitKernel<T>::Limit(std::function<Blob*(const std::string&)> BnInOp2Blob){  
  int32_t class_num = BnInOp2Blob("post_nms_keep_num")->shape().elem_cnt();
  int32_t* post_nms_keep_num_ptr = BnInOp2Blob("post_nms_keep_num")->mut_dptr<int32_t>();
  // votting_score_blob: (im, box_num, class_num)
  T* votting_score_ptr = BnInOp2Blob("votting_score")->mut_dptr<T>();
  int32_t keep_num_per_im = 0;
  FOR_RANGE(int32_t, i, 1, class_num) {
    keep_num_per_im += post_nms_keep_num_ptr[i];
  } 
  if (conf.detections_per_im() > 0 && keep_num_per_im > conf.detections_per_im()) {
      int32_t* post_nms_index_slice_ptr = BnInOp2Blob("post_nms_index_slice")->mut_dptr<int32_t>();
      IndexMemContinuous(post_nms_index_slice_ptr, post_nms_keep_num_ptr, class_num, box_num);
      std::sort(pre_nms_index_slice, pre_nms_index_slice + keep_num_per_im,
            [&](int32_t lhs, int32_t rhs) { return votting_score_ptr[lhs] > votting_score_ptr[rhs]; });
      keep_num_per_im = conf.detections_per_im();
  }
  return keep_num_per_im;
}
void ResultsWithNmsKernel<T>::WriteOutputToOFRecord(int64_t image_index, in32_t limit_num, std::function<Blob*(const std::string&)> BnInOp2Blob){

}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
