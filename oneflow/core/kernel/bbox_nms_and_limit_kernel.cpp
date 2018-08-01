#include "oneflow/core/kernel/bbox_nms_and_limit_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include <math.h>

namespace oneflow {

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
void BboxNmsAndLimitKernel<T>::BroadCastBboxTransform(
    const int64_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* bbox_delta_blob = BnInOp2Blob("bbox_delta");
  const int64_t class_num = BnInOp2Blob("scores")->shape().At(2);
  const int64_t rois_num = rois_blob->shape().At(1);
  // data ptr
  int64_t rois_offset_per_im = im_index * rois_num * 4;
  int64_t delta_offset_per_im = im_index * rois_num * class_num * 4;
  const T* rois_ptr = rois_blob->dptr<T>() + rois_offset_per_im;
  const T* bbox_delta_ptr = bbox_delta_blob->dptr<T>() + delta_offset_per_im;
  // bbox broadcast
  T* bbox_ptr = BnInOp2Blob("bbox")->mut_dptr<T>();
  FOR_RANGE(int64_t, i, 1, rois_num) {
    const T* cur_roi_ptr = rois_ptr + i * 4;
    FOR_RANGE(int64_t, j, 1, class_num) {
      int64_t bbox_offset = (i * class_num + j) * 4;
      const T* cur_bbox_delta_ptr = bbox_delta_ptr + bbox_offset;
      T* cur_bbox_ptr = bbox_ptr + bbox_offset;
      FasterRcnnUtil<T>::BboxTransform(1, cur_roi_ptr, cur_bbox_delta_ptr, cur_bbox_ptr);
    }
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ClipBox(Blob* bbox_blob) const {
  T* bbox_ptr = bbox_blob->mut_dptr<T>();
  const int64_t boxes_num = bbox_blob->shape().elem_cnt() / 4;
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  FasterRcnnUtil<T>::ClipBoxes(boxes_num, conf.image_height(), conf.image_width(), bbox_ptr);
}

template<typename T>
void BboxNmsAndLimitKernel<T>::NmsAndTryVote(
    int64_t im_index, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* scores_blob = BnInOp2Blob("scores");
  int64_t boxes_num = scores_blob->shape().At(1);
  int64_t class_num = scores_blob->shape().At(2);
  int64_t score_offset = im_index * boxes_num * class_num;
  const T* scores_ptr = scores_blob->dptr<T>() + score_offset;
  T* bbox_ptr = BnInOp2Blob("bbox")->mut_dptr<T>();
  int32_t* pre_nms_index_slice_ptr = BnInOp2Blob("pre_nms_index_slice")->mut_dptr<int32_t>();
  int32_t* post_nms_index_slice_ptr = BnInOp2Blob("post_nms_index_slice")->mut_dptr<int32_t>();
  int32_t* post_nms_keep_num_ptr = BnInOp2Blob("post_nms_keep_num")->mut_dptr<int32_t>();
  int32_t* nms_area_tmp_ptr = BnInOp2Blob("nms_area_tmp")->mut_dptr<int32_t>();
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();

  FOR_RANGE(int64_t, i, 1, class_num) {
    int64_t cur_class_offset = i * boxes_num;
    int32_t* pre_nms_indics = pre_nms_index_slice_ptr + cur_class_offset;
    int32_t* post_nms_indics = post_nms_index_slice_ptr + cur_class_offset;
    SortClassBoxIndexByScore(scores_ptr, boxes_num, class_num, i, pre_nms_indics);
    int64_t pre_nms_keep_num =
        FilterSortedIndexByThreshold(boxes_num, scores_ptr, pre_nms_indics, conf.score_thresh());
    post_nms_keep_num_ptr[i] =
        FasterRcnnUtil<T>::Nms(bbox_ptr, pre_nms_indics, pre_nms_keep_num, pre_nms_keep_num,
                               conf.nms_threshold(), nms_area_tmp_ptr, post_nms_indics);
    if (conf.bbox_vote_enabled()) { BboxVoting(i, post_nms_keep_num_ptr[i], BnInOp2Blob, false); }
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::SortClassBoxIndexByScore(const T* scores_ptr,
                                                        const int64_t boxes_num,
                                                        const int64_t class_num,
                                                        const int64_t class_index,
                                                        int32_t* indics) const {
  FOR_RANGE(int64_t, i, 0, boxes_num) { indics[i] = class_index + i * class_num; }
  std::sort(indics, indics + boxes_num,
            [&](int32_t lhs, int32_t rhs) { return scores_ptr[lhs] > scores_ptr[rhs]; });
}

template<typename T>
int64_t BboxNmsAndLimitKernel<T>::FilterSortedIndexByThreshold(const int64_t num,
                                                               const T* scores_ptr,
                                                               const int32_t* indics,
                                                               const float thresh) const {
  FOR_RANGE(int64_t, i, 0, num) {
    int64_t index = indics[i];
    if (scores_ptr[index] <= thresh) { return i; }
  }
  return num;
}

template<typename T>
void BboxNmsAndLimitKernel<T>::BboxVoting(int64_t class_index, int32_t voter_num,
                                          std::function<Blob*(const std::string&)> BnInOp2Blob,
                                          bool need_calc_area) const {
  Blob* bbox_blob = BnInOp2Blob("bbox");
  int64_t bbox_num = bbox_blob->shape().At(1);
  const int32_t votee_num = BnInOp2Blob("post_nms_keep_num")->dptr<int32_t>()[class_index];
  const int32_t* voter_index_ptr =
      BnInOp2Blob("pre_nms_index_slice")->dptr<int32_t>() + class_index * bbox_num;
  const int32_t* votee_index_ptr =
      BnInOp2Blob("post_nms_index_slice")->dptr<int32_t>() + class_index * bbox_num;
  const BboxVoteConf& vote_conf = op_conf().bbox_nms_and_limit_conf().bbox_vote();
  float voting_thresh = vote_conf.thresh();
  float beta = vote_conf.beta();
  ScoringMethod scoring_method = vote_conf.scoring_method();
  // calc area as needed
  int32_t* area_ptr = BnInOp2Blob("nms_area_tmp")->mut_dptr<int32_t>();
  const T* const_bbox_ptr = bbox_blob->dptr<T>();
  if (need_calc_area) {
    FOR_RANGE(int32_t, i, 0, voter_num) {
      int32_t index = voter_index_ptr[i];
      const T* bbox = const_bbox_ptr + index * 4;
      area_ptr[i] = FasterRcnnUtil<T>::BBoxArea(bbox);
    }
  }
  // voting bbox and score
  const T* score_ptr = BnInOp2Blob("scores")->dptr<T>();
  T* voting_score_ptr = BnInOp2Blob("voting_score")->mut_dptr<T>();
  T* bbox_ptr = bbox_blob->mut_dptr<T>();
  FOR_RANGE(int64_t, i, 0, votee_num) {
    const int32_t votee_index = votee_index_ptr[i];
    T* votee_bbox = bbox_ptr + votee_index * 4;
    const int32_t votee_area = FasterRcnnUtil<T>::BBoxArea(votee_bbox);
    int32_t valid_voter_num = 0;
    T score_weighted_bbox_sum[4] = {0, 0, 0, 0};
    T score_sum = 0;
    // used by iou avg scoring
    float iou_weighted_score_sum = 0;
    float iou_sum = 0;
    // used by generalized avg scoring
    float generalized_score_sum = 0;
    FOR_RANGE(int64_t, j, 0, voter_num) {
      const int32_t voter_index = voter_index_ptr[j];
      const int32_t voter_area = area_ptr[j];
      const T voter_score = score_ptr[voter_index];
      const T* voter_bbox = const_bbox_ptr + voter_index * 4;
      const float iou =
          FasterRcnnUtil<T>::InterOverUnion(votee_bbox, votee_area, voter_bbox, voter_area);
      if (iou >= voting_thresh) {
        ++valid_voter_num;
        FOR_RANGE(int32_t, k, 0, 4) { score_weighted_bbox_sum[k] += voter_bbox[k] * voter_score; }
        score_sum += voter_score;
        if (scoring_method == kIouAvg) {
          iou_weighted_score_sum += voter_score * iou;
          iou_sum += iou;
        } else if (scoring_method == kGeneralizedAvg) {
          generalized_score_sum += std::pow<float>(voter_score, beta);
        }
      }
    }
    FOR_RANGE(int32_t, k, 0, 4) { votee_bbox[k] += score_weighted_bbox_sum[k] / score_sum; }
    T voting_score = score_ptr[votee_index];
    switch (scoring_method) {
      case kIdentity:
        // do nothing
        break;
      case kTempAvg: TODO(); break;
      case kAvg: voting_score = score_sum / valid_voter_num; break;
      case kIouAvg: voting_score = static_cast<T>(iou_weighted_score_sum / iou_sum); break;
      case kGeneralizedAvg:
        voting_score = std::pow<T>(generalized_score_sum / valid_voter_num, 1.f / beta);
        break;
      case kQuasiSum:
        voting_score = static_cast<T>(score_sum / std::pow<float>(valid_voter_num, beta));
        break;
      default: UNIMPLEMENTED(); break;
    }
    voting_score_ptr[votee_index] = voting_score;
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
