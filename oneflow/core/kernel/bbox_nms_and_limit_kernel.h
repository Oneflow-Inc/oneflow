#ifndef ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_H_
#define ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/bbox_util.h"

namespace oneflow {

template<typename T>
class ScoringMethodIf {
 public:
  ScoringMethodIf() = default;
  virtual ~ScoringMethodIf() = default;

  using BBox = BBoxImpl<T, ImIndexedBBoxBase, BBoxCoord::kCorner>;
  using ScoredBoxesIndices = ScoreIndices<BBoxIndices<IndexSequence, BBox>, T>;

  void Init(const BboxVoteConf& vote_conf) { vote_conf_ = vote_conf; }
  const BboxVoteConf& conf() const { return vote_conf_; }
  virtual T scoring(
      const ScoredBoxesIndices&, const T default_score,
      const std::function<void(const std::function<void(int32_t, float)>&)>&) const = 0;

 private:
  BboxVoteConf vote_conf_;
};

template<typename T>
class BboxNmsAndLimitKernel final : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BboxNmsAndLimitKernel);
  BboxNmsAndLimitKernel() = default;
  ~BboxNmsAndLimitKernel() = default;

  using BBox = BBoxImpl<T, ImIndexedBBoxBase, BBoxCoord::kCorner>;
  using ScoredBoxesIndices = ScoreIndices<BBoxIndices<IndexSequence, BBox>, T>;
  using Image2IndexVecMap = HashMap<int32_t, std::vector<int32_t>>;

 private:
  void VirtualKernelInit(const ParallelContext* parallel_ctx) override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;

  void BroadcastBboxTransform(const Blob* bbox_blob, const Blob* bbox_pred_blob, 
                              Blob* target_bbox_blob) const;
  void ClipBBox(Blob* target_bbox_blob) const;


  void BroadcastBboxTransform(const int64_t im_index,
                              const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void ClipBox(Blob* bbox_blob) const;
  ScoredBoxesIndices NmsAndTryVote(
      const int64_t im_index, const std::function<Blob*(const std::string&)>& BnInOp2Blob) const;
  void VoteBboxAndScore(const ScoredBoxesIndices& pre_nms_slice,
                        const ScoredBoxesIndices& post_nms_slice, Blob* voting_score_blob,
                        Blob* bbox_blob) const;
  void VoteBbox(const ScoredBoxesIndices& pre_nms_slice,
                const std::function<void(const std::function<void(int32_t, float)>&)>&,
                BBox* ret_votee_bbox) const;
  void Limit(const int32_t limit_num, const float thresh, ScoredBoxesIndices& slice) const;
  void WriteOutputToRecordBlob(const int64_t im_index, const int64_t boxes_num,
                               const ScoredBoxesIndices& slice, Blob* labeled_bbox_blob,
                               Blob* bbox_score_blob) const;

  std::unique_ptr<ScoringMethodIf<T>> scoring_method_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BBOX_NMS_AND_LIMIT_KERNEL_H_
