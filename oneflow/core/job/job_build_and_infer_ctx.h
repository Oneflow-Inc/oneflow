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
  ~JobBuildAndInferCtx() = default;

  Maybe<void> SetJobConf(const JobConfigProto& job_conf);
  Maybe<void> AddAndInferOps(const OperatorConf& op_conf, const ParallelConf& parallel_conf);

  bool HasJobConf() const;
  Maybe<void> AddLossLogicalBlobName(const std::string& lbn);

  Maybe<Shape> GetStaticShape(const std::string& lbn) const;
  Maybe<DataType> GetDataType(const std::string& lbn) const;
  Maybe<bool> IsDynamic(const std::string& lbn) const;
  Maybe<bool> DisableBoxing(const std::string& lbn) const;
  Maybe<long long> GetNumOfLoDLevels(const std::string& lbn) const;
  Maybe<OptInt64> GetBatchAxis(const std::string& lbn) const;
  Maybe<OptInt64> GetSplitAxisFromProducerView(const std::string& lbn) const;
  Maybe<const ParallelDesc*> GetParallelDescFromProducerView(const std::string& lbn) const;

  bool IsSymmetricBlob(const std::string& lbn_or_symmetric_blob_name) const;
  Maybe<int> NumLbiInSymmetricBlob(const std::string& lbn_or_symmetric_blob_name) const;
  Maybe<const LogicalBlobId*> GetLbiInSymmetricBlob(const std::string& lbn_or_symmetric_blob_name,
                                                    int index) const;

  Maybe<Shape> SymmetricBlobGetStaticShape(const std::string& symmetric_blob_name_with_hint) const;
  Maybe<DataType> SymmetricBlobGetDataType(const std::string& symmetric_blob_name_with_hint) const;
  Maybe<bool> SymmetricBlobIsDynamic(const std::string& symmetric_blob_name_with_hint) const;
  Maybe<long long> SymmetricBlobGetNumOfLoDLevels(
      const std::string& symmetric_blob_name_with_hint) const;
  Maybe<bool> SymmetricBlobDisableBoxing(const std::string& symmetric_blob_name_with_hint) const;
  Maybe<OptInt64> SymmetricBlobGetBatchAxis(const std::string& symmetric_blob_name_with_hint) const;
  Maybe<OptInt64> SymmetricBlobGetSplitAxisFromProducerView(
      const std::string& symmetric_blob_name_with_hint) const;
  Maybe<const ParallelDesc*> SymmetricBlobGetParallelDescFromProducerView(
      const std::string& symmetric_blob_name_with_hint) const;

  const Job& job() const;
  Maybe<void> CheckJob() const;

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
  Maybe<std::string> SymmetricBlobNameStripHint(
      const std::string& symmetric_blob_name_with_hint) const;
  Maybe<void> AddAndInferOp(const OperatorConf& op_conf, const ParallelConf& parallel_conf);
  bool HasAnySymmetricBlobInput(const Operator& op) const;
  Maybe<void> CheckAllInputsConvertableToSymmetricBlob(const Operator& op) const;
  Maybe<void> CheckAllInputsWithSameParallelNum(const Operator& op, int32_t parallel_num) const;
  Maybe<const SbpParallel*> SbpParallel4Lbi(const LogicalBlobId& lbi) const;
  Maybe<const ParallelDesc*> ParallelDesc4Lbi(const LogicalBlobId& lbi) const;
  Maybe<const LogicalBlobId*> GetSymmetricBlobSubLbi(const std::string& lbn_or_symmetric_blob_name,
                                                     int32_t index);
  Maybe<std::string> FindOrCreateSymmetricBlobFromCompatibleConsistentBlob(const std::string& lbn);

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
  HashMap<std::string, std::vector<LogicalBlobId>> symmetric_blob_name2lbis_;
  HashMap<LogicalBlobId, std::string> consistent_lbi2symmetric_blob_name_;
  bool is_job_conf_frozen_;
  bool has_job_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
