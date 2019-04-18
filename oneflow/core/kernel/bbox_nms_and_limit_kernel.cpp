#include "oneflow/core/kernel/bbox_nms_and_limit_kernel.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/kernel/kernel_util.h"

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
typename BboxNmsAndLimitKernel<typename std::remove_const<T>::type>::ScoredBoxesIndices
GenScoredBoxesIndices(size_t capacity, int32_t* index_buf, T* bbox_buf, T* score_buf,
                      bool init_index) {
  using TT = typename std::remove_const<T>::type;
  using BBoxT = typename BboxNmsAndLimitKernel<TT>::BBox;
  BBoxIndices<IndexSequence, BBoxT> bbox_inds(IndexSequence(capacity, index_buf, init_index),
                                              const_cast<TT*>(bbox_buf));
  return ScoreIndices<BBoxIndices<IndexSequence, BBoxT>, TT>(bbox_inds, const_cast<TT*>(score_buf));
}

}  // namespace

template<typename T>
void BboxNmsAndLimitKernel<T>::VirtualKernelInit(const ParallelContext* parallel_ctx) {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  if (conf.has_bbox_vote()) {
    scoring_method_.reset(NewObj<ScoringMethodIf<T>>(conf.bbox_vote().scoring_method()));
    scoring_method_->Init(conf.bbox_vote());
  }
  num_images_ = Global<JobDesc>::Get()->DevicePieceSize4ParallelCtx(*parallel_ctx);
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ForwardDim0ValidNum(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ForwardRecordIdInDevicePiece(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  // do nothing
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  const Blob* image_size_blob = BnInOp2Blob("image_size");
  const Blob* bbox_blob = BnInOp2Blob("bbox");
  const Blob* bbox_pred_blob = BnInOp2Blob("bbox_pred");
  const Blob* bbox_prob_blob = BnInOp2Blob("bbox_prob");
  Blob* target_bbox_blob = BnInOp2Blob("target_bbox");
  Blob* vote_bbox_score_blob = BnInOp2Blob("bbox_score");
  Blob* out_bbox_blob = BnInOp2Blob("out_bbox");
  Blob* out_bbox_score_blob = BnInOp2Blob("out_bbox_score");
  Blob* out_bbox_label_blob = BnInOp2Blob("out_bbox_label");
  const Blob* bbox_score_blob = conf.bbox_vote_enabled() ? vote_bbox_score_blob : bbox_prob_blob;

  BroadcastBboxTransform(bbox_blob, bbox_pred_blob, target_bbox_blob);
  ClipBBox(image_size_blob, target_bbox_blob);
  std::vector<int32_t> all_im_bbox_inds;
  std::vector<std::vector<int32_t>> im_detected_bbox_inds_vec(num_images_);
  std::vector<std::vector<int32_t>> im_grouped_bbox_inds_vec = GroupBBox(target_bbox_blob);
  MultiThreadLoop(num_images_, [&](int64_t i) {
    im_detected_bbox_inds_vec[i] = ApplyNmsAndVoteByClass(
        im_grouped_bbox_inds_vec[i], bbox_prob_blob, vote_bbox_score_blob, target_bbox_blob);
    Limit(conf.detections_per_im(), bbox_score_blob, im_detected_bbox_inds_vec[i]);
  });
  for (const auto& im_detected_bbox_inds : im_detected_bbox_inds_vec) {
    all_im_bbox_inds.insert(all_im_bbox_inds.end(), im_detected_bbox_inds.begin(),
                            im_detected_bbox_inds.end());
  }
  OutputBBox(all_im_bbox_inds, target_bbox_blob, out_bbox_blob);
  OutputBBoxScore(all_im_bbox_inds, bbox_score_blob, out_bbox_score_blob);
  OutputBBoxLabel(all_im_bbox_inds, bbox_score_blob->shape().At(1), out_bbox_label_blob);
  if (bbox_blob->has_record_id_in_device_piece_field()) { FillRecordIdInDevicePiece(BnInOp2Blob); }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::BroadcastBboxTransform(const Blob* bbox_blob,
                                                      const Blob* bbox_pred_blob,
                                                      Blob* target_bbox_blob) const {
  const BBoxRegressionWeights& bbox_reg_ws = op_conf().bbox_nms_and_limit_conf().bbox_reg_weights();
  int64_t num_boxes = bbox_blob->shape().At(0);
  int64_t num_classes = bbox_pred_blob->shape().At(1) / 4;
  CHECK_EQ(bbox_pred_blob->shape().At(0), num_boxes);
  const auto* bbox_ptr = BBox::Cast(bbox_blob->dptr<T>());
  const auto* delta_ptr = BBoxDelta<T>::Cast(bbox_pred_blob->dptr<T>());
  BBox* target_bbox_ptr = BBox::Cast(target_bbox_blob->mut_dptr<T>());
  MultiThreadLoop(num_boxes * num_classes, [&](int64_t index) {
    int64_t i = index / num_classes;
    const auto* bbox = bbox_ptr + i;
    const auto* delta = delta_ptr + index;
    BBox* target_bbox = target_bbox_ptr + index;
    target_bbox->Transform(bbox, delta, bbox_reg_ws);
    target_bbox->set_index(bbox->index());
  });
}

template<typename T>
void BboxNmsAndLimitKernel<T>::ClipBBox(const Blob* image_size_blob, Blob* target_bbox_blob) const {
  const int32_t* image_size_ptr = image_size_blob->dptr<int32_t>();
  auto* bbox_ptr = BBox::Cast(target_bbox_blob->mut_dptr<T>());
  FOR_RANGE(size_t, i, 0, target_bbox_blob->shape().Count(0, 2)) {
    const int32_t im_index = bbox_ptr[i].index();
    const int32_t im_height = image_size_ptr[im_index * 2 + 0];
    const int32_t im_width = image_size_ptr[im_index * 2 + 1];
    bbox_ptr[i].Clip(im_height, im_width);
  }
}

template<typename T>
std::vector<std::vector<int32_t>> BboxNmsAndLimitKernel<T>::GroupBBox(
    Blob* target_bbox_blob) const {
  std::vector<std::vector<int32_t>> im_grouped_bbox_inds(num_images_);
  FOR_RANGE(int32_t, i, 0, target_bbox_blob->shape().At(0)) {
    const BBox* bbox = BBox::Cast(target_bbox_blob->dptr<T>(i, 0));
    im_grouped_bbox_inds[bbox->index()].emplace_back(i);
  }
  return im_grouped_bbox_inds;
}

template<typename T>
std::vector<int32_t> BboxNmsAndLimitKernel<T>::ApplyNmsAndVoteByClass(
    const std::vector<int32_t>& bbox_row_ids, const Blob* bbox_prob_blob, Blob* bbox_score_blob,
    Blob* target_bbox_blob) const {
  const BboxNmsAndLimitOpConf& conf = op_conf().bbox_nms_and_limit_conf();
  const T* bbox_prob_ptr = bbox_prob_blob->dptr<T>();
  T* bbox_score_ptr = bbox_score_blob->mut_dptr<T>();
  int32_t num_classes = bbox_prob_blob->shape().At(1);
  std::vector<int32_t> all_cls_bbox_inds;
  all_cls_bbox_inds.reserve(bbox_row_ids.size() * num_classes);
  FOR_RANGE(int32_t, k, 1, num_classes) {
    std::vector<int32_t> cls_bbox_inds(bbox_row_ids.size());
    std::transform(bbox_row_ids.begin(), bbox_row_ids.end(), cls_bbox_inds.begin(),
                   [&](int32_t idx) { return idx * num_classes + k; });
    std::sort(cls_bbox_inds.begin(), cls_bbox_inds.end(), [&](int32_t l_idx, int32_t h_idx) {
      return bbox_prob_ptr[l_idx] > bbox_prob_ptr[h_idx];
    });
    auto lt_thresh_it = std::find_if(cls_bbox_inds.begin(), cls_bbox_inds.end(), [&](int32_t idx) {
      return bbox_prob_ptr[idx] < conf.score_threshold();
    });
    cls_bbox_inds.erase(lt_thresh_it, cls_bbox_inds.end());
    // nms
    auto pre_nms_inds = GenScoredBoxesIndices(cls_bbox_inds.size(), cls_bbox_inds.data(),
                                              target_bbox_blob->dptr<T>(), bbox_prob_ptr, false);
    std::vector<int32_t> post_nms_bbox_inds(cls_bbox_inds.size());
    auto post_nms_inds =
        GenScoredBoxesIndices(post_nms_bbox_inds.size(), post_nms_bbox_inds.data(),
                              target_bbox_blob->mut_dptr<T>(), bbox_score_ptr, false);
    BBoxUtil<BBox>::Nms(conf.nms_threshold(), pre_nms_inds, post_nms_inds);
    // voting
    if (conf.bbox_vote_enabled()) { VoteBboxAndScore(pre_nms_inds, post_nms_inds); }
    // concat all class
    all_cls_bbox_inds.insert(all_cls_bbox_inds.end(), post_nms_inds.index(),
                             post_nms_inds.index() + post_nms_inds.size());
  }
  return all_cls_bbox_inds;
}

template<typename T>
void BboxNmsAndLimitKernel<T>::VoteBboxAndScore(const ScoredBoxesIndices& pre_nms_inds,
                                                ScoredBoxesIndices& post_nms_inds) const {
  const T voting_thresh = op_conf().bbox_nms_and_limit_conf().bbox_vote().threshold();
  FOR_RANGE(size_t, i, 0, post_nms_inds.size()) {
    const auto* votee_bbox = post_nms_inds.GetBBox(i);
    auto ForEachNearBy = [&pre_nms_inds, votee_bbox,
                          voting_thresh](const std::function<void(int32_t, float)>& Handler) {
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
    VoteBbox(pre_nms_inds, post_nms_inds.bbox(bbox_idx), ForEachNearBy);
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::VoteBbox(
    const ScoredBoxesIndices& pre_nms_inds, BBox* votee_bbox,
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
void BboxNmsAndLimitKernel<T>::Limit(size_t limit, const Blob* bbox_score_blob,
                                     std::vector<int32_t>& bbox_inds) const {
  const T* bbox_score_ptr = bbox_score_blob->dptr<T>();
  if (bbox_inds.size() > limit) {
    std::sort(bbox_inds.begin(), bbox_inds.end(), [&](int32_t l_idx, int32_t r_idx) {
      return bbox_score_ptr[l_idx] > bbox_score_ptr[r_idx];
    });
    bbox_inds.resize(limit);
  }
}

template<typename T>
void BboxNmsAndLimitKernel<T>::OutputBBox(const std::vector<int32_t> out_bbox_inds,
                                          const Blob* target_bbox_blob, Blob* out_bbox_blob) const {
  std::memset(out_bbox_blob->mut_dptr<T>(), 0,
              out_bbox_blob->static_shape().elem_cnt() * sizeof(T));
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    const auto* bbox = BBox::Cast(target_bbox_blob->dptr<T>()) + bbox_idx;
    auto* out_bbox = BBox::Cast(out_bbox_blob->mut_dptr<T>()) + out_cnt;
    out_bbox->set_ltrb(bbox->left(), bbox->top(), bbox->right(), bbox->bottom());
    out_bbox->set_index(bbox->index());
    ++out_cnt;
  }
  CHECK_LE(out_cnt, out_bbox_blob->static_shape().At(0));
  out_bbox_blob->set_dim0_valid_num(0, out_cnt);
}

template<typename T>
void BboxNmsAndLimitKernel<T>::OutputBBoxScore(const std::vector<int32_t> out_bbox_inds,
                                               const Blob* bbox_score_blob,
                                               Blob* out_bbox_score_blob) const {
  std::memset(out_bbox_score_blob->mut_dptr<T>(), 0,
              out_bbox_score_blob->static_shape().elem_cnt() * sizeof(T));
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    out_bbox_score_blob->mut_dptr<T>()[out_cnt] = bbox_score_blob->dptr<T>()[bbox_idx];
    ++out_cnt;
  }
  CHECK_LE(out_cnt, out_bbox_score_blob->static_shape().elem_cnt());
  out_bbox_score_blob->set_dim0_valid_num(0, out_cnt);
}

template<typename T>
void BboxNmsAndLimitKernel<T>::OutputBBoxLabel(const std::vector<int32_t> out_bbox_inds,
                                               const int32_t num_classes,
                                               Blob* out_bbox_label_blob) const {
  std::memset(out_bbox_label_blob->mut_dptr<int32_t>(), 0,
              out_bbox_label_blob->static_shape().elem_cnt() * sizeof(int32_t));
  int32_t out_cnt = 0;
  for (int32_t bbox_idx : out_bbox_inds) {
    out_bbox_label_blob->mut_dptr<int32_t>()[out_cnt] = bbox_idx % num_classes;
    ++out_cnt;
  }
  CHECK_LE(out_cnt, out_bbox_label_blob->static_shape().elem_cnt());
  out_bbox_label_blob->set_dim0_valid_num(0, out_cnt);
}

template<typename T>
void BboxNmsAndLimitKernel<T>::FillRecordIdInDevicePiece(
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  Blob* out_bbox_blob = BnInOp2Blob("out_bbox");
  Blob* out_bbox_score_blob = BnInOp2Blob("out_bbox_score");
  Blob* out_bbox_label_blob = BnInOp2Blob("out_bbox_label");
  FOR_RANGE(size_t, i, 0, out_bbox_blob->shape().At(0)) {
    int64_t im_index = BBox::Cast(out_bbox_blob->mut_dptr<T>(i))->index();
    out_bbox_blob->set_record_id_in_device_piece(i, im_index);
    out_bbox_score_blob->set_record_id_in_device_piece(i, im_index);
    out_bbox_label_blob->set_record_id_in_device_piece(i, im_index);
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kBboxNmsAndLimitConf, BboxNmsAndLimitKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
