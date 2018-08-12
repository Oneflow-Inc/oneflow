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
#define DEFINE_SCORING_METHOD(k, score_ptr, default_score, for_each_nearby)              \
class OF_PP_CAT(ScoreMethod, __LINE__) final : public ScoringMethodIf<T> {               \
  public:                                                                                \
   T scoring(const ScoredBBoxSlice<T>& slice, const T default_score,                     \
             const ForEachType& for_each_nearby) const override;                         \
};                                                                                       \
REGISTER_SCORING_METHOD(k, OF_PP_CAT(ScoreMethod, __LINE__));                            \
template<typename T>                                                                     \
T OF_PP_CAT(ScoreMethod, __LINE__)<T>::scoring(const ScoredBBoxSlice<T>& slice,          \
                                               const T default_score,                    \
                                               const ForEachType& for_each_nearby) const
// clang-format on

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kId, slice, default_score, ForEach) {
  return default_score;
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kAvg, slice, default_score, ForEach) {
  T score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t slice_index, float iou) {
    score_sum += slice.GetScore(slice_index);
    ++num;
  });
  return score_sum / num;
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kIouAvg, slice, default_score, ForEach) {
  T iou_weighted_score_sum = 0;
  T iou_sum = 0;
  ForEach([&](int32_t slice_index, float iou) {
    iou_weighted_score_sum += slice.GetScore(slice_index) * iou;
    iou_sum += iou;
  });
  return static_cast<T>(iou_weighted_score_sum / iou_sum);
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kGeneralizedAvg, slice, default_score, ForEach) {
  const float beta = this->conf().beta();
  T generalized_score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t slice_index, float iou) {
    generalized_score_sum += std::pow<T>(slice.GetScore(slice_index), beta);
    ++num;
  });
  return std::pow<T>(generalized_score_sum / num, 1.f / beta);
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kQuasiSum, slice, default_score, ForEach) {
  const float beta = this->conf().beta();
  T score_sum = 0;
  int32_t num = 0;
  ForEach([&](int32_t slice_index, float iou) {
    score_sum += slice.GetScore(slice_index);
    ++num;
  });
  return static_cast<T>(score_sum / std::pow<T>(num, beta));
}

template<typename T>
DEFINE_SCORING_METHOD(ScoringMethod::kTempAvg, slice, default_score, ForEach) {
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
void BboxNmsAndLimitKernel<T>::ForwardDataId(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BnInOp2Blob("labeled_bbox")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("rois"));
  BnInOp2Blob("bbox_score")->CopyDataIdFrom(ctx.device_ctx, BnInOp2Blob("rois"));
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* bbox_blob = BnInOp2Blob("bbox");
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  const int64_t image_num = BnInOp2Blob("rois")->shape().At(0);
  const int32_t class_num = bbox_blob->shape().At(1);
  FOR_RANGE(int64_t, i, 0, image_num) {
    BroadcastBboxTransform(i, BnInOp2Blob);
    ClipBox(bbox_blob);
    ScoredBBoxSlice<T> slice = NmsAndTryVote(i, BnInOp2Blob);
    Limit(conf.detections_per_im(), conf.threshold(), slice);
    WriteOutputToRecordBlob(i, class_num, slice, BnInOp2Blob("labeled_bbox"),
                            BnInOp2Blob("bbox_score"));
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::BroadcastBboxTransform(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const Blob* rois_blob = BnInOp2Blob("rois");
  const Blob* bbox_delta_blob = BnInOp2Blob("bbox_delta");
  Blob* bbox_blob = BnInOp2Blob("bbox");
  const int64_t rois_num = bbox_blob->shape().At(0);
  const int64_t class_num = bbox_blob->shape().At(1);
  CHECK_EQ(rois_blob->shape().At(1), rois_num);
  CHECK_EQ(bbox_delta_blob->shape().elem_cnt(), rois_blob->shape().elem_cnt() * class_num);
  CHECK_EQ(bbox_delta_blob->shape().At(0), rois_blob->shape().At(0) * rois_num);
  const BBoxRegressionWeights& bbox_reg_ws = op_conf().bbox_nms_and_limit_conf().bbox_reg_weights();
  // bbox broadcast
  FOR_RANGE(int64_t, i, 0, rois_num) {
    const BBox<T>* roi_bbox = BBox<T>::Cast(rois_blob->dptr<T>(im_index, i));
    const BBoxDelta<T>* class_bbox_delta =
        BBoxDelta<T>::Cast(bbox_delta_blob->dptr<T>(im_index * rois_num + i));
    FOR_RANGE(int64_t, j, 0, class_num) {
      BBox<T>::MutCast(bbox_blob->mut_dptr<T>(i, j))
          ->Transform(roi_bbox, class_bbox_delta + j, bbox_reg_ws);
    }
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ClipBox(Blob* bbox_blob) const {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  FasterRcnnUtil<T>::ClipBoxes(bbox_blob->shape().Count(0, 2), conf.image_height(),
                               conf.image_width(), bbox_blob->mut_dptr<T>());
}

template<typename T>
ScoredBBoxSlice<T> BboxNmsAndLimitKernel<T>::NmsAndTryVote(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  // input: scores (n * r, c) <=> (n, r * c)
  const Blob* scores_blob = BnInOp2Blob("scores");
  // datatmp: bbox (r, c, 4)
  Blob* bbox_blob = BnInOp2Blob("bbox");
  // datatmp: voting_score_blob (r, c)
  Blob* voting_score_blob = BnInOp2Blob("voting_score");
  const int64_t boxes_num = bbox_blob->shape().At(0);
  const int64_t class_num = scores_blob->shape().At(1);
  const T* bbox_ptr = bbox_blob->dptr<T>();
  const T* scores_ptr = scores_blob->dptr<T>(im_index * boxes_num);
  Blob* pre_nms_index_slice_blob = BnInOp2Blob("pre_nms_index_slice");
  Blob* post_nms_index_slice_blob = BnInOp2Blob("post_nms_index_slice");
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  ScoredBBoxSlice<T> all_class_slice(boxes_num * class_num, bbox_ptr, voting_score_blob->dptr<T>(),
                                     post_nms_index_slice_blob->mut_dptr<int32_t>());
  all_class_slice.Truncate(0);
  FOR_RANGE(int64_t, i, 1, class_num) {
    int32_t* cls_pre_nms_idx_ptr = pre_nms_index_slice_blob->mut_dptr<int32_t>(i);
    FOR_RANGE(int64_t, j, 0, boxes_num) { cls_pre_nms_idx_ptr[j] = i + j * class_num; }
    ScoredBBoxSlice<T> pre_nms_slice(boxes_num, bbox_ptr, scores_ptr, cls_pre_nms_idx_ptr);
    pre_nms_slice.DescSortByScore(false);
    pre_nms_slice.TruncateByThreshold(conf.score_threshold());

    int32_t* cls_post_nms_idx_ptr = post_nms_index_slice_blob->mut_dptr<int32_t>(i);
    ScoredBBoxSlice<T> post_nms_slice(boxes_num, bbox_ptr, scores_ptr, cls_post_nms_idx_ptr);
    post_nms_slice.NmsFrom(conf.nms_threshold(), pre_nms_slice);

    if (conf.bbox_vote_enabled()) {
      VoteBboxAndScore(pre_nms_slice, post_nms_slice, voting_score_blob, bbox_blob);
    }

    all_class_slice.Concat(post_nms_slice);
  }
  if (!conf.bbox_vote_enabled()) {
    std::memcpy(voting_score_blob->mut_dptr<T>(), scores_ptr, boxes_num * class_num * sizeof(T));
  }
  return all_class_slice;
}

template<typename T>
void BboxNmsAndLimitKernel<T>::VoteBboxAndScore(const ScoredBBoxSlice<T>& pre_nms_slice,
                                                const ScoredBBoxSlice<T>& post_nms_slice,
                                                Blob* voting_score_blob,
                                                Blob* voting_bbox_blob) const {
  CHECK_EQ(pre_nms_slice.score_ptr(), post_nms_slice.score_ptr());
  CHECK_EQ(pre_nms_slice.bbox_ptr(), post_nms_slice.bbox_ptr());
  const T voting_thresh = op_conf().bbox_nms_and_limit_conf().bbox_vote().threshold();
  BBox<T>* ret_voting_bbox_ptr = BBox<T>::MutCast(voting_bbox_blob->mut_dptr<T>());

  FOR_RANGE(int64_t, i, 0, post_nms_slice.available_len()) {
    const BBox<T>* votee_bbox = post_nms_slice.GetBBox(i);
    auto ForEachNearBy = [&](const std::function<void(int32_t, float)>& Handler) {
      FOR_RANGE(int64_t, j, 0, pre_nms_slice.available_len()) {
        const BBox<T>* voter_bbox = pre_nms_slice.GetBBox(j);
        float iou = voter_bbox->InterOverUnion(votee_bbox);
        if (iou >= voting_thresh) { Handler(j, iou); }
      }
    };
    // new bbox
    VoteBbox(pre_nms_slice, ForEachNearBy, ret_voting_bbox_ptr + post_nms_slice.GetSlice(i));
    // new score
    voting_score_blob->mut_dptr<T>()[post_nms_slice.GetSlice(i)] =
        scoring_method_->scoring(pre_nms_slice, post_nms_slice.GetScore(i), ForEachNearBy);
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::VoteBbox(
    const ScoredBBoxSlice<T>& pre_nms_slice,
    const std::function<void(const std::function<void(int32_t, float)>&)>& ForEachNearBy,
    BBox<T>* ret_bbox_ptr) const {
  std::array<T, 4> score_weighted_bbox = {0, 0, 0, 0};
  T score_sum = 0;
  ForEachNearBy([&](int32_t voter_slice_index, float iou) {
    const T voter_score = pre_nms_slice.GetScore(voter_slice_index);
    FOR_RANGE(int32_t, k, 0, 4) {
      score_weighted_bbox[k] += pre_nms_slice.GetBBox(voter_slice_index)->bbox()[k] * voter_score;
    }
    score_sum += voter_score;
  });
  FOR_RANGE(int32_t, k, 0, 4) { ret_bbox_ptr->mut_bbox()[k] = score_weighted_bbox[k] / score_sum; }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::Limit(const int32_t limit_num, const float thresh,
                                     ScoredBBoxSlice<T>& slice) const {
  slice.DescSortByScore(false);
  slice.Truncate(limit_num);
  slice.Filter([&](const T score, const BBox<T>* bbox) { return score < thresh; });
  slice.Sort([&](const T l_score, const T r_score, const BBox<T>& l_bbox, const BBox<T>& r_bbox) {
    return l_bbox.Area() > r_bbox.Area();
  });
}

template<typename T>
void BboxNmsAndLimitKernel<T>::WriteOutputToRecordBlob(const int64_t im_index,
                                                       const int64_t class_num,
                                                       const ScoredBBoxSlice<T>& slice,
                                                       Blob* labeled_bbox_blob,
                                                       Blob* bbox_score_blob) const {
  Int32List16* labeled_bbox_ptr = labeled_bbox_blob->mut_dptr<Int32List16>() + im_index;
  FloatList16* score_ptr = bbox_score_blob->mut_dptr<FloatList16>() + im_index;
  FOR_RANGE(int64_t, i, 0, slice.available_len()) {
    const BBox<T>* bbox = slice.GetBBox(i);
    labeled_bbox_ptr->mutable_value()->add_value(bbox->x1());
    labeled_bbox_ptr->mutable_value()->add_value(bbox->y1());
    labeled_bbox_ptr->mutable_value()->add_value(bbox->x2());
    labeled_bbox_ptr->mutable_value()->add_value(bbox->y2());
    labeled_bbox_ptr->mutable_value()->add_value(slice.GetSlice(i) % class_num);
    score_ptr->mutable_value()->add_value(slice.GetScore(i));
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
