#include "oneflow/core/kernel/bbox_nms_and_limit_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
//#include "oneflow/core/kernel/faster_rcnn_util.h"
#include <math.h>

namespace oneflow {

namespace {

template<typename T>
inline int32_t BBoxArea(const T* box) {
  return (box[2] - box[0] + 1) * (box[3] - box[1] + 1);
}
template<typename T>
inline float InterOverUnion(const T* box0, const int32_t area0, const T* box1,
                            const int32_t area1) {
  const int32_t iw = std::min(box0[2], box1[2]) - std::max(box0[0], box1[0]) + 1;
  if (iw <= 0) { return 0; }
  const int32_t ih = std::min(box0[3], box1[3]) - std::max(box0[1], box1[1]) + 1;
  if (ih <= 0) { return 0; }
  const float inter = iw * ih;
  return inter / (area0 + area1 - inter);
}
// to delete
template<typename T>
int32_t Nms(const T* img_proposal_ptr, const int32_t* sorted_score_slice_ptr,
            const int32_t pre_nms_top_n, const int32_t post_nms_top_n, const float nms_threshold,
            int32_t* area_ptr, int32_t* post_nms_slice_ptr) {
  CHECK_NE(sorted_score_slice_ptr, post_nms_slice_ptr);
  FOR_RANGE(int32_t, i, 0, pre_nms_top_n) {
    area_ptr[i] = BBoxArea(img_proposal_ptr + sorted_score_slice_ptr[i] * 4);
  }
  int32_t keep_num = 0;
  auto IsSuppressed = [&](int32_t index) -> bool {
    FOR_RANGE(int32_t, post_nms_slice_i, 0, keep_num) {
      const int32_t keep_index = post_nms_slice_ptr[post_nms_slice_i];
      const int32_t area0 = area_ptr[keep_index];
      const int32_t area1 = area_ptr[index];
      if (area0 >= area1 * nms_threshold && area1 >= area0 * nms_threshold) {
        const T* box0 = img_proposal_ptr + sorted_score_slice_ptr[keep_index] * 4;
        const T* box1 = img_proposal_ptr + sorted_score_slice_ptr[index] * 4;
        if (InterOverUnion(box0, area0, box1, area1) >= nms_threshold) { return true; }
      }
    }
    return false;
  };
  FOR_RANGE(int32_t, sorted_score_slice_i, 0, pre_nms_top_n) {
    if (IsSuppressed(sorted_score_slice_i)) { continue; }
    post_nms_slice_ptr[keep_num++] = sorted_score_slice_i;
    if (keep_num == post_nms_top_n) { break; }
  }
  FOR_RANGE(int32_t, i, 0, keep_num) {
    post_nms_slice_ptr[i] = sorted_score_slice_ptr[post_nms_slice_ptr[i]];
  }
  return keep_num;
}
}  // namespace

template<typename T>
void BboxNmsAndLimitKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t image_num = BnInOp2Blob("bbox")->shape().At(0);
  FOR_RANGE(int64_t, i, 0, image_num) {
    BroadCastBboxTransform(i, BnInOp2Blob);
    ClipBox(BnInOp2Blob("bbox"));
    NmsAndTryVote(i, BnInOp2Blob);
    int64_t keep_num_per_im = Limit(BnInOp2Blob);
    WriteOutputToOFRecord(i, keep_num_per_im, BnInOp2Blob);
  }
}
template<typename T>
void BboxNmsAndLimitKernel<T>::SortClassBoxIndexByScore(const T* score_ptr, const int64_t box_num,
                                                        const int64_t class_num,
                                                        const int64_t class_index,
                                                        int32_t* pre_nms_index_slice) const {
  // score_ptr : roi_num * class_num, stores score
  // pre_nms_index_slice : class_num * roi_num, stores index in score_ptr
  FOR_RANGE(int64_t, i, 0, box_num) { pre_nms_index_slice[i] = class_index + i * class_num; }
  std::sort(pre_nms_index_slice, pre_nms_index_slice + box_num,
            [&](int32_t lhs, int32_t rhs) { return score_ptr[lhs] > score_ptr[rhs]; });
}
template<typename T>
void BboxNmsAndLimitKernel<T>::BroadCastBboxTransform(
    const int64_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // bbox_delta_blob: (im, box_num, class*4)
  const Blob* bbox_delta_blob = BnInOp2Blob("bbox_delta");
  int64_t box_num = bbox_delta_blob->shape().At(1);
  int64_t class_num = bbox_delta_blob->shape().At(2) / 4;
  int64_t rois_offset_per_im = im_index * box_num * 4;
  int64_t delta_offset_per_im = im_index * box_num * class_num * 4;
  // rois_blob: (im, box_num, 4)
  const T* rois_ptr = BnInOp2Blob("rois")->dptr<T>() + rois_offset_per_im;
  const T* bbox_delta_ptr = BnInOp2Blob("bbox_delta")->dptr<T>() + delta_offset_per_im;
  // bbox_blob: (box_num, class*4)
  T* bbox_ptr = BnInOp2Blob("bbox")->mut_dptr<T>();
  FOR_RANGE(int64_t, i, 1, box_num) {
    int64_t rois_offset = i * 4;
    float w = rois_ptr[rois_offset + 2] - rois_ptr[rois_offset + 0] + 1.0f;
    float h = rois_ptr[rois_offset + 3] - rois_ptr[rois_offset + 1] + 1.0f;
    float ctr_x = rois_ptr[rois_offset + 0] + 0.5f * w;
    float ctr_y = rois_ptr[rois_offset + 1] + 0.5f * h;
    FOR_RANGE(int64_t, j, 1, class_num) {
      int64_t delta_offset = (i * class_num + j) * 4;
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
void BboxNmsAndLimitKernel<T>::ClipBox(Blob* bbox_blob) const {
  // bbox_blob: (box_num, class * 4)
  T* bbox_ptr = bbox_blob->mut_dptr<T>();
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  const int64_t bbox_buf_size = bbox_blob->shape().elem_cnt();
  for (int64_t i = 0; i < bbox_buf_size; i += 4) {
    bbox_ptr[i + 0] = std::max<T>(std::min<T>(bbox_ptr[i + 0], conf.image_width()), 0);
    bbox_ptr[i + 1] = std::max<T>(std::min<T>(bbox_ptr[i + 1], conf.image_height()), 0);
    bbox_ptr[i + 2] = std::max<T>(std::min<T>(bbox_ptr[i + 2], conf.image_width()), 0);
    bbox_ptr[i + 3] = std::max<T>(std::min<T>(bbox_ptr[i + 3], conf.image_height()), 0);
  }
}
template<typename T>
int64_t BboxNmsAndLimitKernel<T>::FilterSortedIndexByThreshold(const T* scores_ptr,
                                                               const float score_thresh,
                                                               const int32_t* pre_nms_index_slice,
                                                               const int64_t num) const {
  FOR_RANGE(int64_t, i, 0, num) {
    int64_t scores_index = pre_nms_index_slice[i];
    if (scores_ptr[scores_index] <= score_thresh) { return i; }
  }
  return num;
}
template<typename T>
void BboxNmsAndLimitKernel<T>::BboxVoting(
    int64_t class_index, int32_t voter_num,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  TODO();
}

template<typename T>
void BboxNmsAndLimitKernel<T>::NmsAndTryVote(
    int64_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // scores_blob: (im, box_num, class_num)  pre_nms_index_slice_blob: (cls_num, roi)
  // post_nms_index_slice_blob: (cls_num, roi) post_nms_keep_num_blob: (cls_num) nms_area_tmp_blob:
  // (cls_num)
  const Blob* scores_blob = BnInOp2Blob("scores");
  int64_t box_num = scores_blob->shape().At(1);
  int64_t class_num = scores_blob->shape().At(2);
  int64_t score_offset = im_index * box_num * class_num;
  const T* scores_ptr = scores_blob->dptr<T>() + score_offset;
  T* bbox_ptr = BnInOp2Blob("bbox")->mut_dptr<T>();
  Blob* pre_nms_index_slice_blob = BnInOp2Blob("pre_nms_index_slice");
  Blob* post_nms_index_slice_blob = BnInOp2Blob("post_nms_index_slice");
  int32_t* post_nms_keep_num_ptr = BnInOp2Blob("post_nms_keep_num")->mut_dptr<int32_t>();
  int32_t* nms_area_tmp_ptr = BnInOp2Blob("nms_area_tmp")->mut_dptr<int32_t>();

  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  FOR_RANGE(int64_t, i, 1, class_num) {  // just foreground, class index start from 1
    int64_t index_offset = i * box_num;
    int32_t* pre_nms_index_slice_ptr = pre_nms_index_slice_blob->mut_dptr<int32_t>() + index_offset;
    int32_t* post_nms_index_slice_ptr =
        post_nms_index_slice_blob->mut_dptr<int32_t>() + index_offset;
    SortClassBoxIndexByScore(scores_ptr, box_num, class_num, i, pre_nms_index_slice_ptr);
    int64_t pre_nms_keep_num = FilterSortedIndexByThreshold(scores_ptr, conf.score_thresh(),
                                                            pre_nms_index_slice_ptr, box_num);
    // post_nms_keep_num_ptr[i] = FasterRcnnUtil<T>::Nms(
    //     boxes_ptr, pre_nms_index_slice_ptr, pre_nms_keep_num, pre_nms_keep_num,
    //     conf.nms_threshold(), nms_area_tmp_ptr, post_nms_index_slice_ptr);
    post_nms_keep_num_ptr[i] =
        Nms(bbox_ptr, pre_nms_index_slice_ptr, pre_nms_keep_num, pre_nms_keep_num,
            conf.nms_threshold(), nms_area_tmp_ptr, post_nms_index_slice_ptr);
    if (conf.bbox_vote_enabled()) { BboxVoting(i, post_nms_keep_num_ptr[i], BnInOp2Blob); }
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::IndexMemContinuous(const int32_t* post_nms_keep_num_ptr,
                                                  const int64_t class_num, const int64_t box_num,
                                                  int32_t* post_nms_index_slice_ptr) const {
  int64_t keep_index = 0;
  FOR_RANGE(int64_t, i, 1, class_num) {
    FOR_RANGE(int32_t, j, 0, post_nms_keep_num_ptr[i]) {
      post_nms_index_slice_ptr[keep_index++] = post_nms_index_slice_ptr[i * box_num + j];
    }
  }
}

template<typename T>
int64_t BboxNmsAndLimitKernel<T>::Limit(
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t class_num = BnInOp2Blob("post_nms_keep_num")->shape().elem_cnt();
  int64_t box_num = BnInOp2Blob("votting_score")->shape().At(0);
  int32_t* post_nms_keep_num_ptr = BnInOp2Blob("post_nms_keep_num")->mut_dptr<int32_t>();
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  // votting_score_blob: (box_num, class_num)
  T* votting_score_ptr = BnInOp2Blob("votting_score")->mut_dptr<T>();
  int32_t keep_num_per_im = 0;
  FOR_RANGE(int64_t, i, 1, class_num) { keep_num_per_im += post_nms_keep_num_ptr[i]; }
  if (conf.detections_per_im() > 0 && keep_num_per_im > conf.detections_per_im()) {
    int32_t* post_nms_index_slice_ptr = BnInOp2Blob("post_nms_index_slice")->mut_dptr<int32_t>();
    IndexMemContinuous(post_nms_index_slice_ptr, class_num, box_num, post_nms_keep_num_ptr);
    std::sort(
        post_nms_index_slice_ptr, post_nms_index_slice_ptr + keep_num_per_im,
        [&](int32_t lhs, int32_t rhs) { return votting_score_ptr[lhs] > votting_score_ptr[rhs]; });
    keep_num_per_im = conf.detections_per_im();
  }
  return keep_num_per_im;
}
template<typename T>
void BboxNmsAndLimitKernel<T>::WriteOutputToOFRecord(
    int64_t image_index, int64_t limit_num,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const T* bbox_ptr = BnInOp2Blob("bbox")->dptr<T>();
  const T* score_ptr = BnInOp2Blob("voting_score")->dptr<T>();
  const int32_t* post_num_keep_index_ptr = BnInOp2Blob("post_nms_index_slice")->dptr<int32_t>();
  OFRecord* labeled_bbox_record = BnInOp2Blob("labeled_bbox")->mut_dptr<OFRecord>() + image_index;
  Feature& labeled_bbox_feature =
      labeled_bbox_record->mutable_feature()->at(kOFRecordMapDefaultKey);
  OFRecord* score_record = BnInOp2Blob("bbox_score")->mut_dptr<OFRecord>() + image_index;
  Feature& score_feature = score_record->mutable_feature()->at(kOFRecordMapDefaultKey);
  FOR_RANGE(int64_t, i, 0, limit_num) {
    int32_t index = post_num_keep_index_ptr[i];
    labeled_bbox_feature.mutable_int32_list()->add_value(bbox_ptr[index * 4 + 0]);
    labeled_bbox_feature.mutable_int32_list()->add_value(bbox_ptr[index * 4 + 1]);
    labeled_bbox_feature.mutable_int32_list()->add_value(bbox_ptr[index * 4 + 2]);
    labeled_bbox_feature.mutable_int32_list()->add_value(bbox_ptr[index * 4 + 3]);
    score_feature.mutable_float_list()->add_value(score_ptr[index]);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
