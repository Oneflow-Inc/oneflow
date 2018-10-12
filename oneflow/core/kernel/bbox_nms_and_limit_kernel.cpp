#include "oneflow/core/kernel/bbox_nms_and_limit_kernel.h"
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
 public:                                                                                 \
  using ScoredBoxesIndices = typename ScoringMethodIf<T>::ScoredBoxesIndices;            \
  T scoring(const ScoredBoxesIndices& slice, const T default_score,                      \
             const ForEachType& for_each_nearby) const override;                         \
};                                                                                       \
REGISTER_SCORING_METHOD(k, OF_PP_CAT(ScoreMethod, __LINE__));                            \
template<typename T>                                                                     \
T OF_PP_CAT(ScoreMethod, __LINE__)<T>::scoring(const ScoredBoxesIndices& slice,          \
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

template<typename T> 
BboxNmsAndLimitKernel<T>::ScoredBoxesIndices
GenScoredBoxesIndices(size_t capacity, int32_t* index_buf, const T* bbox_buf,
                      const T* score_buf, bool init_index) {
  using BBoxT = typename BboxNmsAndLimitKernel<T>::BBox;
  BBoxIndices<IndexSequence, BBoxT> bbox_inds(IndexSequence(capacity, index_buf, init_index), bbox_buf);
  return ScoreIndices<BBoxIndices<IndexSequence, BBoxT>, T>(bbox_inds, score_buf);
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
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const Blob* bbox_pred_blob = BnInOp2Blob("bbox_pred");
  Blob* target_bbox_blob = BnInOp2Blob("target_bbox");
  const int64_t image_num = BnInOp2Blob("rois")->shape().At(0);
  const int32_t class_num = bbox_blob->shape().At(1);

  BroadcastBboxTransform(bbox_blob, bbox_pred_blob, target_bbox_blob);
  ClipBBox(target_bbox_blob);
  auto im_grouped_bbox_inds = GroupBBox(target_bbox_blob);
  for (auto& pair : im_grouped_bbox_inds) {
  
  
  }


  ///////////

  FOR_RANGE(int64_t, i, 0, image_num) {
    ClipBox(bbox_blob);
    auto slice = NmsAndTryVote(i, BnInOp2Blob);
    Limit(conf.detections_per_im(), conf.threshold(), slice);
    WriteOutputToRecordBlob(i, class_num, slice, BnInOp2Blob("labeled_bbox"),
                            BnInOp2Blob("bbox_score"));
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::BroadcastBboxTransform(
    const Blob* bbox_blob, const Blob* bbox_pred_blob, 
    Blob* target_bbox_blob) const {
  const BBoxRegressionWeights& bbox_reg_ws = op_conf().bbox_nms_and_limit_conf().bbox_reg_weights();
  int64_t num_boxes = bbox_blob->shape().At(0);
  int64_t num_classes = bbox_pred_blob->shape().At(1) / 4;
  CHECK_EQ(bbox_pred_blob->shape().At(0), num_boxes)
  FOR_RANGE(int64_t, i, 0, num_boxes) {
    const auto* bbox = BBox::Cast(bbox_blob->dptr<T>(i));
    const auto* delta = BBoxDelta<T>::Cast(bbox_pred_blob->dptr<T>(i));
    FOR_RANGE(int64_t, j, 0, num_classes) {
      BBox::MutCast(target_bbox_blob->mut_dptr<T>(i, j))->Transform(bbox, delta + j, bbox_reg_ws);
    }
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ClipBBox(Blob* target_bbox_blob) const {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  auto* bbox_ptr = BBox::MutCast(target_bbox_blob->mut_dptr<T>());
  FOR_RANGE(int64_t, i, 0, target_bbox_blob->shape().Count(0, 2)) { 
    bbox_ptr[i].Clip(conf.image_height(), conf.image_width()); 
  }
}

template<typename T>
typename BboxNmsAndLimitKernel<T>::Image2IndexVecMap
BboxNmsAndLimitKernel<T>::GroupBBox(Blob* target_bbox_blob) const {
  Image2IndexVecMap im_grouped_bbox_inds;
  FOR_RANGE(int32_t, i, 0, target_bbox_blob->shape().At(0)) {
    const auto* bbox = BBox::Cast(target_bbox_blob->dptr<T>(i, 0));
    int32_t im_idx = bbox->im_index<int32_t>();
    im_grouped_bbox_inds[im_idx].emplace_back(i);
  }
}

template<typename T>
typename BboxNmsAndLimitKernel<T>::ScoredBoxesIndices
BboxNmsAndLimitKernel<T>::ApplyNmsAndVoteByClass(
    const std::vector<int32_t>& bbox_row_ids, const Blob* bbox_prob_blob,
    Blob* bbox_score_blob, Blob* target_bbox_blob) const {
  int32_t num_classes = bbox_prob_blob->shape().At();
  std::vector<int32_t> all_cls_bbox_inds(bbox_row_ids.size() * num_classes);
  FOR_RANGE(int32_t, k, 1, num_classes) {
    std::vector<int32_t> cls_bbox_inds(bbox_row_ids.size());
    std::transform(bbox_row_ids.begin(), bbox_row_ids.end(), cls_bbox_inds.begin(), [&](int32_t idx) {
      return idx * num_classes + k;
    });
    std::sort(cls_bbox_inds.begin(), cls_bbox_inds.end(), [&](int32_t l_idx, int32_t h_idx) {
      return bbox_prob_ptr[l_idx] > bbox_prob_ptr[h_idx];
    });
    auto lt_thresh_it = std::find_if(cls_bbox_inds.begin(), cls_bbox_inds.end(), [&](int32_t idx) {
      return bbox_prob_ptr[idx] < conf.score_threshold();
    })
    cls_bbox_inds.erase(lt_thresh_it, cls_bbox_inds.end());
    // nms
    auto pre_nms_inds = GenScoredBoxesIndices(cls_bbox_inds.size(), cls_bbox_inds.data(), 
                                              target_bbox_blob->dptr<T>(), bbox_prob_ptr, false);
    std::vector<int32_t> post_nms_bbox_inds(cls_bbox_inds.size());
    auto post_nms_inds = GenScoredBoxesIndices(post_nms_bbox_inds.size(), post_nms_bbox_inds.data(), 
                                              target_bbox_blob->dptr<T>(), bbox_prob_ptr, false);
    BBoxUtil<BBox>::Nms(conf.nms_threshold(), pre_nms_inds, post_nms_inds);
    // voting
    if (conf.bbox_vote_enabled()) {
      //VoteBboxAndScore(pre_nms_slice, post_nms_slice, voting_score_blob, bbox_blob);
    }
    // concat all class
    all_bbox_inds.insert(all_bbox_inds.end(), post_nms_inds.index(), 
                          post_nms_inds.index() + post_nms_inds.size());
  }
}


template<typename T>
BboxNmsAndLimitKernel<T>::ScoredBoxesIndices BboxNmsAndLimitKernel<T>::NmsAndTryVote(
    const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  const Blob* scores_blob = BnInOp2Blob("scores");
  Blob* bbox_blob = BnInOp2Blob("bbox");
  Blob* voting_score_blob = BnInOp2Blob("voting_score");
  Blob* pre_nms_index_slice_blob = BnInOp2Blob("pre_nms_index_slice");
  Blob* post_nms_index_slice_blob = BnInOp2Blob("post_nms_index_slice");
  const int64_t boxes_num = bbox_blob->shape().At(0);
  const int64_t class_num = scores_blob->shape().At(1);
  const T* bbox_ptr = bbox_blob->dptr<T>();
  const T* scores_ptr = scores_blob->dptr<T>(im_index * boxes_num);

  ///////

  HashMap<int32_t, std::vector<int32_t>> im_grouped_bbox_inds;
  FOR_RANGE(int32_t, i, 0, num_boxes) {
    const auto* bbox = BBox::Cast(target_bbox_blob->dptr<T>(i, 0));
    int32_t im_idx = bbox->im_index<int32_t>();
    im_grouped_bbox_inds[im_idx].emplace_back(i);
  }

  for (auto& pair : im_grouped_bbox_inds) {
    std::vector<int32_t> all_bbox_inds(pair.second.size() * num_classes);
    FOR_RANGE(int32_t, k, 1, num_classes) {
      std::vector<int32_t> cls_bbox_inds(pair.second.size());
      std::transform(pair.second.begin(), pair.second.end(), cls_bbox_inds.begin(), [&](int32_t idx) {
        return idx * num_classes + k;
      });
      std::sort(cls_bbox_inds.begin(), cls_bbox_inds.end(), [&](int32_t l_idx, int32_t h_idx) {
        return bbox_prob_ptr[l_idx] > bbox_prob_ptr[h_idx];
      });
      auto lt_thresh_it = std::find_if(cls_bbox_inds.begin(), cls_bbox_inds.end(), [&](int32_t idx) {
        return bbox_prob_ptr[idx] < conf.score_threshold();
      })
      cls_bbox_inds.erase(lt_thresh_it, cls_bbox_inds.end());
      // nms
      auto pre_nms_inds = GenScoredBoxesIndices(cls_bbox_inds.size(), cls_bbox_inds.data(), 
                                                target_bbox_blob->dptr<T>(), bbox_prob_ptr, false);
      std::vector<int32_t> post_nms_bbox_inds(cls_bbox_inds.size());
      auto post_nms_inds = GenScoredBoxesIndices(post_nms_bbox_inds.size(), post_nms_bbox_inds.data(), 
                                                target_bbox_blob->dptr<T>(), bbox_prob_ptr, false);
      BBoxUtil<BBox>::Nms(conf.nms_threshold(), pre_nms_inds, post_nms_inds);
      // voting
      if (conf.bbox_vote_enabled()) {
        //VoteBboxAndScore(pre_nms_slice, post_nms_slice, voting_score_blob, bbox_blob);
      }
      // concat all class
      all_bbox_inds.insert(all_bbox_inds.end(), post_nms_inds.index(), 
                           post_nms_inds.index() + post_nms_inds.size());
    }
  }



  /////////


  IndexSequence bbox_inds(num_boxes, bbox_inds_blob->mut_dptr<int32_t>());
  bbox_inds.GroupBy()

  auto all_class_boxes =
      GenScoredBoxesIndex(boxes_num * class_num, post_nms_index_slice_blob->mut_dptr<int32_t>(),
                          bbox_ptr, voting_score_blob->dptr<T>(), false);
  all_class_boxes.Truncate(0);
  FOR_RANGE(int64_t, i, 1, class_num) {
    int32_t* cls_pre_nms_idx_ptr = pre_nms_index_slice_blob->mut_dptr<int32_t>(i);
    FOR_RANGE(int64_t, j, 0, boxes_num) { cls_pre_nms_idx_ptr[j] = i + j * class_num; }
    auto pre_nms_slice =
        GenScoredBoxesIndex(boxes_num, cls_pre_nms_idx_ptr, bbox_ptr, scores_ptr, false);
    pre_nms_slice.SortByScore([](T lhs_score, T rhs_score) { return lhs_score > rhs_score; });
    size_t score_top_n =
        pre_nms_slice.FindByScore([&](T score) { return score < conf.score_threshold(); });
    pre_nms_slice.Truncate(score_top_n);

    int32_t* cls_post_nms_idx_ptr = post_nms_index_slice_blob->mut_dptr<int32_t>(i);
    auto post_nms_slice =
        GenScoredBoxesIndex(boxes_num, cls_post_nms_idx_ptr, bbox_ptr, scores_ptr, false);
    FasterRcnnUtil<T>::Nms(conf.nms_threshold(), pre_nms_slice, post_nms_slice);
  
    if (conf.bbox_vote_enabled()) {
      VoteBboxAndScore(pre_nms_slice, post_nms_slice, voting_score_blob, bbox_blob);
    }

    all_class_boxes.Concat(post_nms_slice);
  }
  if (!conf.bbox_vote_enabled()) {
    std::memcpy(voting_score_blob->mut_dptr<T>(), scores_ptr, boxes_num * class_num * sizeof(T));
  }
  return all_class_boxes;
}

template<typename T>
void BboxNmsAndLimitKernel<T>::VoteBboxAndScore(const ScoredBoxesIndex<T>& pre_nms_slice,
                                                const ScoredBoxesIndex<T>& post_nms_slice,
                                                Blob* voting_score_blob,
                                                Blob* voting_bbox_blob) const {
  CHECK_EQ(pre_nms_slice.score_ptr(), post_nms_slice.score_ptr());
  CHECK_EQ(pre_nms_slice.bbox_ptr(), post_nms_slice.bbox_ptr());
  const T voting_thresh = op_conf().bbox_nms_and_limit_conf().bbox_vote().threshold();
  BBox<T>* ret_voting_bbox_ptr = BBox<T>::MutCast(voting_bbox_blob->mut_dptr<T>());

  FOR_RANGE(size_t, i, 0, post_nms_slice.size()) {
    const BBox<T>* votee_bbox = post_nms_slice.GetBBox(i);
    auto ForEachNearBy = [&](const std::function<void(int32_t, float)>& Handler) {
      FOR_RANGE(size_t, j, 0, pre_nms_slice.size()) {
        const BBox<T>* voter_bbox = pre_nms_slice.GetBBox(j);
        float iou = voter_bbox->InterOverUnion(votee_bbox);
        if (iou >= voting_thresh) { Handler(j, iou); }
      }
    };
    // new bbox
    VoteBbox(pre_nms_slice, ForEachNearBy, ret_voting_bbox_ptr + post_nms_slice.GetIndex(i));
    // new score
    voting_score_blob->mut_dptr<T>()[post_nms_slice.GetIndex(i)] =
        scoring_method_->scoring(pre_nms_slice, post_nms_slice.GetScore(i), ForEachNearBy);
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::VoteBbox(
    const ScoredBoxesIndex<T>& pre_nms_slice,
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
                                     ScoredBoxesIndex<T>& boxes) const {
  boxes.SortByScore([](T lhs_score, T rhs_score) { return lhs_score > rhs_score; });
  boxes.Truncate(limit_num);
  boxes.FilterByScore([&](size_t, int32_t, T score) { return score < thresh; });
}

template<typename T>
void BboxNmsAndLimitKernel<T>::WriteOutputToRecordBlob(const int64_t im_index,
                                                       const int64_t class_num,
                                                       const ScoredBoxesIndex<T>& slice,
                                                       Blob* labeled_bbox_blob,
                                                       Blob* bbox_score_blob) const {
  Int32List16* labeled_bbox_ptr = labeled_bbox_blob->mut_dptr<Int32List16>() + im_index;
  FloatList16* score_ptr = bbox_score_blob->mut_dptr<FloatList16>() + im_index;
  FOR_RANGE(int64_t, i, 0, slice.size()) {
    const BBox<T>* bbox = slice.GetBBox(i);
    labeled_bbox_ptr->mutable_value()->add_value(bbox->x1());
    labeled_bbox_ptr->mutable_value()->add_value(bbox->y1());
    labeled_bbox_ptr->mutable_value()->add_value(bbox->x2());
    labeled_bbox_ptr->mutable_value()->add_value(bbox->y2());
    labeled_bbox_ptr->mutable_value()->add_value(slice.GetIndex(i) % class_num);
    score_ptr->mutable_value()->add_value(slice.GetScore(i));
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
