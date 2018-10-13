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
  const Blob* bbox_prob_blob = BnInOp2Blob("bbox_prob");
  Blob* target_bbox_blob = BnInOp2Blob("target_bbox");
  Blob* bbox_score_blob = BnInOp2Blob("bbox_score");
  Blob* out_bbox_blob = BnInOp2Blob("out_bbox");
  Blob* out_bbox_score_blob = BnInOp2Blob("out_bbox_score");
  Blob* out_bbox_label_blob = BnInOp2Blob("out_bbox_label");

  BroadcastBboxTransform(bbox_blob, bbox_pred_blob, target_bbox_blob);
  ClipBBox(target_bbox_blob);
  std::vector<int32_t> all_im_bbox_inds;
  auto im_grouped_bbox_inds = GroupBBox(target_bbox_blob);
  for (auto& pair : im_grouped_bbox_inds) {
    auto im_detected_bbox_inds = ApplyNmsAndVoteByClass(pair.second, bbox_pred_blob, 
                                                        bbox_score_blob, target_bbox_blob);
    all_im_bbox_inds.insert(all_im_bbox_inds.end(), im_detected_bbox_inds.begin(), 
                                                    im_detected_bbox_inds.end());
  }
  Limit(bbox_score_blob, all_im_bbox_inds);
  OutputBBox(all_im_bbox_inds, target_bbox_blob, out_bbox_blob);
  OutputBBoxScore(all_im_bbox_inds, bbox_score_blob, out_bbox_score_blob);
  OutputBBoxLabel(all_im_bbox_inds, bbox_prob_blob->shape().At(1), out_bbox_label_blob);
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
std::vector<int32_t> BboxNmsAndLimitKernel<T>::ApplyNmsAndVoteByClass(
    const std::vector<int32_t>& bbox_row_ids, const Blob* bbox_prob_blob,
    Blob* bbox_score_blob, Blob* target_bbox_blob) const {
  const T* bbox_prob_ptr = bbox_prob_blob->dptr<T>();
  T* bbox_score_ptr = bbox_score_blob->mut_dptr<T>();
  int32_t num_classes = bbox_prob_blob->shape().At(1);
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
                                              target_bbox_blob->dptr<T>(), bbox_score_ptr, false);
    BBoxUtil<BBox>::Nms(conf.nms_threshold(), pre_nms_inds, post_nms_inds);
    // voting
    if (conf.bbox_vote_enabled()) {
      VoteBboxAndScore(pre_nms_inds, post_nms_inds);
    }
    // concat all class
    all_cls_bbox_inds.insert(all_cls_bbox_inds.end(), post_nms_inds.index(), 
                             post_nms_inds.index() + post_nms_inds.size());
  }
  if (!conf.bbox_vote_enabled()) {
    std::memcpy(bbox_score_ptr, bbox_prob_ptr, bbox_prob_blob.shape()->elem_cnt() * sizeof(T));
  }
  return all_cls_bbox_inds;
}

template<typename T>
void BboxNmsAndLimitKernel<T>::VoteBboxAndScore(const ScoredBoxesIndices& pre_nms_inds,
                                                const ScoredBoxesIndices& post_nms_inds) const {
  const T voting_thresh = op_conf().bbox_nms_and_limit_conf().bbox_vote().threshold();
  FOR_RANGE(size_t, i, 0, post_nms_inds.size()) {
    const auto* votee_bbox = post_nms_inds.GetBBox(i);
    auto ForEachNearBy = 
        [&pre_nms_inds, votee_bbox, voting_thresh](const std::function<void(int32_t, float)>& Handler) {
        FOR_RANGE(size_t, j, 0, pre_nms_inds.size()) {
          const auto* voter_bbox = pre_nms_inds.GetBBox(j);
          float iou = voter_bbox->InterOverUnion(votee_bbox);
          if (iou >= voting_thresh) { Handler(j, iou); }
        }
    };
    int32_t bbox_idx = post_nms_inds.GetIndex(i);
    T* score_ptr = const_cast<T*>(post_nms_inds.score());
    score_ptr[bbox_idx] =
        scoring_method_->scoring(pre_nms_inds, post_nms_inds.GetScore(i), ForEachNearBy);
    VoteBbox(pre_nms_inds, post_nms_inds.mut_bbox(bbox_idx), ForEachNearBy);
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::VoteBbox(
    const ScoredBoxesIndices& pre_nms_inds,
    BBox* votee_bbox,
    const std::function<void(const std::function<void(int32_t, float)>&)>& ForEachNearBy) const {
  std::array<T, 4> score_weighted_bbox = {0, 0, 0, 0};
  T score_sum = 0;
  ForEachNearBy([&](int32_t voter_idx, float iou) {
    const T voter_score = pre_nms_inds.GetScore(voter_idx);
    FOR_RANGE(int32_t, k, 0, 4) {
      score_weighted_bbox[k] += pre_nms_inds.GetBBox(voter_idx)->bbox_elem(k) * voter_score;
    }
    score_sum += voter_score;
  });
  FOR_RANGE(int32_t, k, 0, 4) { votee_bbox->set_bbox_elem(k, score_weighted_bbox[k] / score_sum); }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::Limit(const Blob* bbox_score_blob, 
                                     std::vector<int32_t>& bbox_inds) const {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  const T* bbox_score_ptr = bbox_score_blob->dptr<T>();
  std::sort(bbox_inds.begin(), bbox_inds.end(), [&](int32_t l_idx, int32_t r_idx) {
    return bbox_score_ptr[l_idx] > bbox_score_ptr[r_idx];
  });
  auot lt_threah_it = std::find_if(bbox_inds.begin(), bbox_inds.end(), [&](int32_t idx){
    return bbox_score_ptr[idx] < conf.threshold();
  });
  bbox_inds.erase(lt_threah_it, bbox_inds.end());
  if (bbox_inds.size() > conf.detections_per_im()) {
    bbox_inds.resize(conf.detections_per_im());
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::OutputBBox(const std::vector<int32_t> out_bbox_inds,
                                          const Blob* target_bbox_blob, 
                                          Blob* out_bbox_blob) const {
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    const auto* bbox = BBox::Cast(target_bbox_blob->dptr<T>()) + bbox_idx;
    auto* out_bbox = BBox::MutCast(out_bbox_blob->mut_dptr<T>()) + (out_cnt++);
    out_bbox->set_corner_coord(bbox->left(), bbox->top(), bbox->right(), bbox->bottom());
    out_bbox->set_im_index(bbox->im_index());
  }
  CHECK_LE(out_cnt, out_bbox_blob.shape().At(0));
  out_bbox_blob->set_dim0_valid_num(0, out_cnt);
}

template<typename T>
void BboxNmsAndLimitKernel<T>::OutputBBoxScore(const std::vector<int32_t> out_bbox_inds,
                                               const Blob* bbox_score_blob, 
                                               Blob* out_bbox_score_blob) const {
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    out_bbox_score_blob->mut_dptr<T>(out_cnt++) = bbox_score_blob->dptr<T>() + bbox_idx;
  }
  CHECK_LE(out_cnt, out_bbox_score_blob.shape().elem_cnt());
  out_bbox_score_blob->set_dim0_valid_num(0, out_cnt);
}

template<typename T>
void BboxNmsAndLimitKernel<T>::OutputBBoxLabel(const std::vector<int32_t> out_bbox_inds,
                                               const int32_t num_classes, 
                                               Blob* out_bbox_label_blob) const {
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    out_bbox_label_blob->mut_dptr<T>(out_cnt++) = bbox_idx % num_classes;
  }
  CHECK_LE(out_cnt, out_bbox_label_blob.shape().elem_cnt());
  out_bbox_label_blob->set_dim0_valid_num(0, out_cnt);
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
