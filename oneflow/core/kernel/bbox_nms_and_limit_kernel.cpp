#include "oneflow/core/kernel/bbox_nms_and_limit_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/faster_rcnn_util.h"
#include "oneflow/core/common/auto_registration_factory.h"

namespace oneflow {

namespace {

#define REGISTER_SCORING_METHOD_CLASS(k, class_name, data_type_pair)   \
  REGISTER_CLASS(k, ScoringMethodIf<OF_PP_PAIR_FIRST(data_type_pair)>, \
                 class_name<OF_PP_PAIR_FIRST(data_type_pair)>);

#define REGISTER_SCORING_METHOD(k, class_name)                                       \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SCORING_METHOD_CLASS, (k), (class_name), \
                                   FLOATING_DATA_TYPE_SEQ)

using ForEachType = std::function<void(const std::function<void(int32_t, float)>&)>;

// clang-format off
#define DEFINE_SCORING_METHOD(k, score_ptr, votee_index, for_each_nearby)                \
class OF_PP_CAT(ScoreMethod, __LINE__) final : public ScoringMethodIf<T> {               \
   public:                                                                               \
   T scoring(const T* score_ptr, const int32_t votee_index,                              \
	     const ForEachType& for_each_nearby) const override;	                 \
};                                                                                       \
REGISTER_SCORING_METHOD(k, OF_PP_CAT(ScoreMethod, __LINE__));                            \
template<typename T>                                                                     \
T OF_PP_CAT(ScoreMethod, __LINE__)<T>::scoring(const T* score_ptr,                       \
					       const int32_t votee_index,                \
                                               const ForEachType& for_each_nearby) const
// clang-format on

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kId, score_ptr, votee_index, ForEach) {
  return score_ptr[votee_index];
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kAvg, score_ptr, votee_index, ForEach) {
  T score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t index, float iou) {
    score_sum += score_ptr[index];
    ++num;
  });
  return score_sum / num;
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kIouAvg, score_ptr, votee_index, ForEach) {
  T iou_weighted_score_sum = 0;
  T iou_sum = 0;
  ForEach([&](int32_t index, float iou) {
    iou_weighted_score_sum += score_ptr[index] * iou;
    iou_sum += iou;
  });
  return static_cast<T>(iou_weighted_score_sum / iou_sum);
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kGeneralizedAvg, score_ptr, votee_index, ForEach) {
  const float beta = this->conf().beta();
  T generalized_score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t index, float iou) {
    generalized_score_sum += std::pow<T>(score_ptr[index], beta);
    ++num;
  });
  return std::pow<T>(generalized_score_sum / num, 1.f / beta);
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kQuasiSum, score_ptr, votee_index, ForEach) {
  const float beta = this->conf().beta();
  T score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t index, float iou) {
    score_sum += score_ptr[index];
    ++num;
  });
  return static_cast<T>(score_sum / std::pow<T>(num, beta));
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kTempAvg, score_ptr, votee_index, ForEach) {
  TODO();
  return 0;
}

}  // namespace

template<typename T>
void BboxNmsAndLimitKernel<T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  if (conf.has_bbox_vote()) {
    scoring_method_.reset(NewObj<ScoringMethodIf<T>>(conf.bbox_vote().scoring_method()));
    scoring_method_->Init(conf.bbox_vote());
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const int64_t image_num = BnInOp2Blob("rois")->shape().At(0);
  FOR_RANGE(int64_t, i, 0, image_num) {
    BroadCastBboxTransform(i, BnInOp2Blob);
    ClipBox(BnInOp2Blob("bbox"));
    NmsAndTryVote(i, BnInOp2Blob);
    const int64_t limit_num = Limit(BnInOp2Blob);
    WriteOutputToOFRecord(i, limit_num, BnInOp2Blob);
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::BroadCastBboxTransform(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* bbox_delta_blob = BnInOp2Blob("bbox_delta");
  Blob* bbox_blob = BnInOp2Blob("bbox");
  const int64_t rois_num = bbox_blob->shape().At(0);
  const int64_t class_num = bbox_blob->shape().At(1);
  CHECK_EQ(rois_blob->shape().At(1), rois_num);
  CHECK_EQ(bbox_delta_blob->shape().elem_cnt(), rois_blob->shape().elem_cnt() * class_num);
  CHECK_EQ(bbox_delta_blob->shape().At(0), rois_blob->shape().At(0) * rois_num);
  // bbox broadcast
  FOR_RANGE(int64_t, i, 1, rois_num) {
    const BBox<T>* roi_bbox = BBox<T>::Cast(rois_blob->dptr<T>(im_index, i));
    FOR_RANGE(int64_t, j, 1, class_num) {
      const BBoxDelta<T>* delta =
          BBoxDelta<T>::Cast(bbox_delta_blob->dptr<T>(im_index * rois_num + i, j * 4));
      BBox<T>::MutCast(bbox_blob->mut_dptr<T>(i, j))->Transform(roi_bbox, delta);
    }
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ClipBox(Blob* bbox_blob) const {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  FasterRcnnUtil<T>::ClipBoxes(bbox_blob->shape().elem_cnt() / 4, conf.image_height(),
                               conf.image_width(), bbox_blob->mut_dptr<T>());
}

template<typename T>
void BboxNmsAndLimitKernel<T>::NmsAndTryVote(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* scores_blob = BnInOp2Blob("scores");
  Blob* bbox_blob = BnInOp2Blob("bbox");
  Blob* voting_score_blob = BnInOp2Blob("voting_score");
  const int64_t boxes_num = bbox_blob->shape().At(0);
  const int64_t class_num = scores_blob->shape().At(1);
  const int64_t score_offset = im_index * boxes_num * class_num;
  const T* scores_ptr = scores_blob->dptr<T>() + score_offset;
  int32_t* pre_nms_index_slice_ptr = BnInOp2Blob("pre_nms_index_slice")->mut_dptr<int32_t>();
  int32_t* post_nms_index_slice_ptr = BnInOp2Blob("post_nms_index_slice")->mut_dptr<int32_t>();
  int32_t* post_nms_keep_num_ptr = BnInOp2Blob("post_nms_keep_num")->mut_dptr<int32_t>();
  int32_t* nms_area_tmp_ptr = BnInOp2Blob("nms_area_tmp")->mut_dptr<int32_t>();
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();

  FOR_RANGE(int64_t, i, 1, class_num) {
    const int64_t cur_class_offset = i * boxes_num;
    int32_t* cls_pre_nms_idx_ptr = pre_nms_index_slice_ptr + cur_class_offset;
    int32_t* cls_post_nms_idx_ptr = post_nms_index_slice_ptr + cur_class_offset;
    SortClassBoxIndexByScore(scores_ptr, boxes_num, class_num, i, cls_pre_nms_idx_ptr);
    const int64_t pre_nms_keep_num = FilterSortedIndexByThreshold(
        boxes_num, scores_ptr, cls_pre_nms_idx_ptr, conf.score_threshold());
    post_nms_keep_num_ptr[i] = FasterRcnnUtil<T>::Nms(
        bbox_blob->dptr<T>(), cls_pre_nms_idx_ptr, pre_nms_keep_num, pre_nms_keep_num,
        conf.nms_threshold(), nms_area_tmp_ptr, cls_post_nms_idx_ptr);
    if (conf.bbox_vote_enabled()) {
      BboxVoting(im_index, i, pre_nms_keep_num, post_nms_keep_num_ptr[i], pre_nms_index_slice_ptr,
                 post_nms_index_slice_ptr, nms_area_tmp_ptr, scores_blob, voting_score_blob,
                 bbox_blob);
    } else {
      std::memcpy(voting_score_blob->mut_dptr<T>(), scores_ptr, boxes_num * class_num);
    }
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::SortClassBoxIndexByScore(const T* scores_ptr,
                                                        const int64_t boxes_num,
                                                        const int64_t class_num,
                                                        const int64_t class_index,
                                                        int32_t* idx_ptr) const {
  FOR_RANGE(int64_t, i, 0, boxes_num) { idx_ptr[i] = class_index + i * class_num; }
  std::sort(idx_ptr, idx_ptr + boxes_num,
            [&](int32_t lhs, int32_t rhs) { return scores_ptr[lhs] > scores_ptr[rhs]; });
}

template<typename T>
int64_t BboxNmsAndLimitKernel<T>::FilterSortedIndexByThreshold(const int64_t num,
                                                               const T* scores_ptr,
                                                               const int32_t* idx_ptr,
                                                               const float thresh) const {
  FOR_RANGE(int64_t, i, 0, num) {
    if (scores_ptr[idx_ptr[i]] <= thresh) { return i; }
  }
  return num;
}

template<typename T>
void BboxNmsAndLimitKernel<T>::BboxVoting(int64_t im_index, int64_t class_index, int32_t voter_num,
                                          int32_t votee_num, const int32_t* pre_nms_index_slice_ptr,
                                          const int32_t* post_nms_index_slice_ptr,
                                          const int32_t* area_ptr, const Blob* score_blob,
                                          Blob* voting_score_blob, Blob* bbox_blob) const {
  const int64_t bbox_num = bbox_blob->shape().At(0);
  const int32_t* voter_index_ptr = pre_nms_index_slice_ptr + class_index * bbox_num;
  const int32_t* votee_index_ptr = post_nms_index_slice_ptr + class_index * bbox_num;
  const T* score_ptr = score_blob->dptr<T>(im_index);
  const T voting_thresh = op_conf().bbox_nms_and_limit_conf().bbox_vote().threshold();

  FOR_RANGE(int64_t, i, 0, votee_num) {
    const int32_t votee_index = votee_index_ptr[i];
    const BBox<T>* votee_bbox = BBox<T>::Cast(bbox_blob->dptr<T>()) + votee_index;
    auto ForEachNearBy = [&](const std::function<void(int32_t, float)>& Handler) {
      FOR_RANGE(int64_t, j, 0, voter_num) {
        const int32_t voter_index = voter_index_ptr[j];
        const BBox<T>* voter_bbox = BBox<T>::Cast(bbox_blob->dptr<T>()) + voter_index;
        float iou = voter_bbox->InterOverUnion(votee_bbox);
        if (iou >= voting_thresh) { Handler(voter_index, iou); }
      }
    };
    // stat new bbox
    T score_weighted_bbox_sum[4] = {0, 0, 0, 0};
    T score_sum = 0;
    ForEachNearBy([&](int32_t voter_index, float iou) {
      const T voter_score = score_ptr[voter_index];
      const BBox<T>* voter_bbox = BBox<T>::Cast(bbox_blob->dptr<T>()) + voter_index;
      FOR_RANGE(int32_t, k, 0, 4) { score_weighted_bbox_sum[k] += (*voter_bbox)[k] * voter_score; }
      score_sum += voter_score;
    });
    // new bbox
    BBox<T>* new_votee_bbox = BBox<T>::MutCast(bbox_blob->mut_dptr<T>()) + votee_index;
    FOR_RANGE(int32_t, k, 0, 4) { (*new_votee_bbox)[k] = score_weighted_bbox_sum[k] / score_sum; }
    // new score
    voting_score_blob->mut_dptr<T>()[votee_index] =
        scoring_method_->scoring(voting_score_blob->dptr<T>(), votee_index, ForEachNearBy);
  }
}

template<typename T>
int64_t BboxNmsAndLimitKernel<T>::IndexMemContinuous(const int64_t class_num, const int64_t box_num,
                                                     const int32_t* post_nms_keep_num_ptr,
                                                     int32_t* post_nms_index_slice_ptr) const {
  int64_t keep_index = 0;
  int64_t keep_num = 0;
  FOR_RANGE(int64_t, i, 1, class_num) {
    keep_num += post_nms_keep_num_ptr[i];
    FOR_RANGE(int32_t, j, 0, post_nms_keep_num_ptr[i]) {
      post_nms_index_slice_ptr[keep_index++] = post_nms_index_slice_ptr[i * box_num + j];
    }
  }
  return keep_num;
}

template<typename T>
int64_t BboxNmsAndLimitKernel<T>::Limit(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* voting_score_blob = BnInOp2Blob("voting_score");
  const int64_t boxes_num = voting_score_blob->shape().At(0);
  const int64_t class_num = voting_score_blob->shape().At(1);
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  const T* voting_score_ptr = voting_score_blob->dptr<T>();
  int32_t* post_nms_keep_num_ptr = BnInOp2Blob("post_nms_keep_num")->mut_dptr<int32_t>();
  int32_t* post_nms_index_slice_ptr = BnInOp2Blob("post_nms_index_slice")->mut_dptr<int32_t>();
  int32_t keep_num_per_im =
      IndexMemContinuous(class_num, boxes_num, post_nms_keep_num_ptr, post_nms_index_slice_ptr);
  if (conf.detections_per_im() > 0 && keep_num_per_im > conf.detections_per_im()) {
    std::sort(
        post_nms_index_slice_ptr, post_nms_index_slice_ptr + keep_num_per_im,
        [&](int32_t lhs, int32_t rhs) { return voting_score_ptr[lhs] > voting_score_ptr[rhs]; });
    keep_num_per_im = conf.detections_per_im();
  }
  return keep_num_per_im;
}

template<typename T>
void BboxNmsAndLimitKernel<T>::WriteOutputToOFRecord(
    const int64_t im_index, const int64_t limit_num,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const T* score_ptr = BnInOp2Blob("voting_score")->dptr<T>();
  const int32_t* post_nms_index_slice_ptr = BnInOp2Blob("post_nms_index_slice")->dptr<int32_t>();
  OFRecord* labeled_bbox_record = BnInOp2Blob("labeled_bbox")->mut_dptr<OFRecord>() + im_index;
  Feature& labeled_bbox_feature = (*labeled_bbox_record->mutable_feature())[kOFRecordMapDefaultKey];
  OFRecord* score_record = BnInOp2Blob("bbox_score")->mut_dptr<OFRecord>() + im_index;
  Feature& score_feature = (*score_record->mutable_feature())[kOFRecordMapDefaultKey];
  FOR_RANGE(int64_t, i, 0, limit_num) {
    int32_t index = post_nms_index_slice_ptr[i];
    const BBox<T>* bbox = BBox<T>::Cast(bbox_blob->dptr<T>()) + index;
    labeled_bbox_feature.mutable_int32_list()->add_value(bbox->x1());
    labeled_bbox_feature.mutable_int32_list()->add_value(bbox->y1());
    labeled_bbox_feature.mutable_int32_list()->add_value(bbox->x2());
    labeled_bbox_feature.mutable_int32_list()->add_value(bbox->y2());
    labeled_bbox_feature.mutable_int32_list()->add_value(index % bbox_blob->shape().At(0));
    score_feature.mutable_float_list()->add_value(score_ptr[index]);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
