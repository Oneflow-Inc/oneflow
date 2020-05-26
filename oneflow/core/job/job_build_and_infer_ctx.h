#ifndef ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
#define ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/blob_desc.h"

namespace oneflow {

class JobBuildAndInferCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobBuildAndInferCtx);
  JobBuildAndInferCtx(Job* job, int64_t job_id);
  virtual ~JobBuildAndInferCtx() = default;

  Maybe<OperatorConf> CheckAndCompleteUserOpConf(const OperatorConf& op_conf);
  Maybe<void> SetJobConf(const JobConfigProto& job_conf);
  Maybe<void> Complete();
  Maybe<void> AddAndInferOp(const OperatorConf& op_conf, const ParallelConf& parallel_conf);
  Maybe<void> AddAndInferConsistentOp(const OperatorConf& op_conf,
                                      const ParallelConf& parallel_conf);
  Maybe<void> AddAndInferMirroredOp(const OperatorConf& op_conf, const ParallelConf& parallel_conf);
  Maybe<void> AddLossLogicalBlobName(const std::string& lbn);

  bool HasJobConf() const;
  Maybe<Shape> GetStaticShape(const std::string& lbn) const;
  Maybe<DataType> GetDataType(const std::string& lbn) const;
  Maybe<bool> IsDynamic(const std::string& lbn) const;
  Maybe<bool> DisableBoxing(const std::string& lbn) const;
  Maybe<bool> IsTensorList(const std::string& lbn) const;
  Maybe<OptInt64> GetBatchAxis(const std::string& lbn) const;
  Maybe<OptInt64> GetSplitAxisFromProducerView(const std::string& lbn) const;
  Maybe<const ParallelDesc*> GetParallelDescFromProducerView(const std::string& lbn) const;

  bool IsMirroredBlob(const std::string& lbn) const;
  Maybe<int> MirroredBlobGetNumSubLbi(const std::string& lbn) const;
  Maybe<const LogicalBlobId*> MirroredBlobGetSubLbi(const std::string& lbn, int index) const;

  Maybe<Shape> MirroredBlobGetStaticShape(const std::string& lbn_with_hint) const;
  Maybe<DataType> MirroredBlobGetDataType(const std::string& lbn_with_hint) const;
  Maybe<bool> MirroredBlobIsDynamic(const std::string& lbn_with_hint) const;
  Maybe<bool> MirroredBlobIsTensorList(const std::string& lbn_with_hint) const;
  Maybe<bool> MirroredBlobDisableBoxing(const std::string& lbn_with_hint) const;
  Maybe<OptInt64> MirroredBlobGetBatchAxis(const std::string& lbn_with_hint) const;
  Maybe<OptInt64> MirroredBlobGetSplitAxisFromProducerView(const std::string& lbn_with_hint) const;
  Maybe<const ParallelDesc*> MirroredBlobGetParallelDescFromProducerView(
      const std::string& lbn_with_hint) const;

  const Job& job() const;
  Maybe<void> CheckJob() const;

 protected:
  virtual int64_t SizeOfSubConsistentOpList(int64_t parallel_num) const = 0;

 private:
  Maybe<ParallelConf> InferOpParallelConf(
      const Operator& op, const ParallelConf& origin_parallel_conf,
      const HashMap<std::string, bool>& ibn2disable_boxing) const;
  Maybe<void> AddOpNameParallelConf2Placement(const std::string& op_name,
                                              const ParallelConf& parallel_conf);
  void InitIbn2DisableBoxing(const Operator& op, HashMap<std::string, bool>* ibn2disable_boxing);
  void UpdateLbi2DisableBoxing(const Operator& op,
                               const HashMap<std::string, bool>& ibn2disable_boxing);
  Maybe<void> AddLbiParallelConf2BlobPlacement(
      const Operator* op, std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn);
  Maybe<OperatorConf> DecodeLbiHintAndReturnNewOpConf(
      const Operator& op, SbpSignature* sbp_sig_conf,
      HashMap<std::string, bool>* ibn2disable_boxing) const;
  void AddOp7AddSbpSigConf2Job(const OperatorConf& operator_conf,
                               const SbpSignature& sbp_signature) const;
  Maybe<void> InferOpOutSbpParallel(Operator*, const SbpSignature&, const ParallelDesc&,
                                    SbpSignature*);
  Maybe<void> GenOpProducedEmptyLogicalBlobDesc(Operator* op);
  Maybe<void> CheckOpBlobSplitability(Operator*, const SbpSignature&, int64_t parallel_num);
  Maybe<void> CheckPlacement() const;
  Maybe<void> CheckJobConf() const;
  Maybe<void> CheckLbnValidAndExist(const std::string& lbn) const;
  Maybe<LogicalBlobId> GetMirroredLbi(const std::string& lbn_with_hint) const;
  bool HasAnyMirroredBlobInput(const Operator& op) const;
  Maybe<void> CheckAllInputsConvertableToMirroredBlob(const Operator& op) const;
  Maybe<void> CheckAllInputsWithSameParallelNum(const Operator& op, int32_t parallel_num) const;
  Maybe<const SbpParallel*> SbpParallel4Lbi(const LogicalBlobId& lbi) const;
  Maybe<const ParallelDesc*> ParallelDesc4Lbi(const LogicalBlobId& lbi) const;
  Maybe<LogicalBlobId> FindOrCreateMirroredLbiFromCompatibleConsistentBlob(
      const LogicalBlobId& lbn);
  Maybe<void> AddLossConsistentBlobName(const std::string& lbn);
  Maybe<void> AddLossMirroredBlobName(const std::string& lbn);
  Maybe<const LogicalBlobId*> GetSubLbi(const LogicalBlobId& lbi, int32_t index);
  Maybe<bool> AllInputsBroadcastParallel(const Operator& op) const;
  bool IsVariableLbi(const LogicalBlobId& lbi) const;

  Job* job_;
  int64_t job_id_;
  HashMap<LogicalBlobId, OptInt64> lbi2batch_axis_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
  HashMap<LogicalBlobId, SbpParallel> lbi2sbp_parallel_from_producer_view_;
  HashMap<LogicalBlobId, ParallelDesc> lbi2parallel_desc_from_producer_view_;
  HashMap<LogicalBlobId, bool> lbi2disable_boxing_;
  HashMap<std::string, std::shared_ptr<Operator>> op_name2op_;
  HashMap<ParallelDesc, PlacementGroup*> parallel_desc2placement_group_;
  HashMap<ParallelDesc, BlobPlacementGroup*> parallel_desc2blob_placement_group_;
  HashMap<LogicalBlobId, LogicalBlobId> consistent_lbi2mirrored_lbi_;
  HashMap<LogicalBlobId, std::vector<LogicalBlobId>> mirrored_lbi2sub_lbis_;
  HashMap<LogicalBlobId, ParallelDesc> mirrored_lbi2parallel_desc_;
  HashMap<LogicalBlobId, SbpParallel> mirrored_lbi2sbp_parallel_;
  bool is_job_conf_frozen_;
  bool has_job_conf_;
};

class LazyJobBuildAndInferCtx : public JobBuildAndInferCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LazyJobBuildAndInferCtx);
  LazyJobBuildAndInferCtx(Job* job, int64_t job_id) : JobBuildAndInferCtx(job, job_id) {}
  virtual ~LazyJobBuildAndInferCtx() = default;

 private:
  int64_t SizeOfSubConsistentOpList(int64_t parallel_num) const override { return parallel_num; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
