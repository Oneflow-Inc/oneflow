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
  Maybe<void> AddAndInferOp(const OperatorConf& op_conf, const ParallelConf& parallel_conf);
  Maybe<void> AddLossLogicalBlobName(const std::string& lbn);

  bool HasJobConf() const;
  Maybe<Shape> GetStaticShape(const std::string& lbn) const;
  Maybe<DataType> GetDataType(const std::string& lbn) const;
  Maybe<bool> IsDynamic(const std::string& lbn) const;
  Maybe<long long> GetNumOfLoDLevels(const std::string& lbn) const;
  Maybe<OptInt64> GetBatchAxis(const std::string& lbn) const;
  Maybe<OptInt64> GetSplitAxisFromProducerView(const std::string& lbn) const;
  Maybe<const ParallelDesc*> GetParallelDescFromProducerView(const std::string& lbn) const;

  const Job& job() const;
  Maybe<void> CheckJob() const;

 private:
  Maybe<void> AddOpNameParallelConf2Placement(const std::string& op_name,
                                              const ParallelConf& parallel_conf);
  Maybe<void> AddLbiParallelConf2BlobPlacement(
      const Operator* op, std::function<ParallelDesc*(const std::string&)> ParallelDesc4Obn);
  Maybe<void> DecodeSplitHint7AddOp7AddSbpSigConf2Job(Operator*, SbpSignature*);
  Maybe<void> InferOpOutSbpParallel(Operator*, const SbpSignature&, const ParallelDesc&,
                                    SbpSignature*);
  Maybe<void> GenOpProducedEmptyLogicalBlobDesc(Operator* op);
  Maybe<void> CheckOpBlobSplitability(Operator*, const SbpSignature&, int64_t parallel_num);
  Maybe<void> CheckPlacement() const;
  Maybe<void> CheckJobConf() const;
  Maybe<void> CheckLbnValidAndExist(const std::string& lbn) const;

  Job* job_;
  int64_t job_id_;
  HashMap<LogicalBlobId, OptInt64> lbi2batch_axis_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
  HashMap<LogicalBlobId, SbpParallel> lbi2sbp_parallel_from_producer_view_;
  HashMap<LogicalBlobId, ParallelDesc> lbi2parallel_desc_from_producer_view_;
  HashMap<std::string, std::shared_ptr<Operator>> op_name2op_;
  HashMap<ParallelDesc, PlacementGroup*> parallel_desc2placement_group_;
  HashMap<ParallelDesc, BlobPlacementGroup*> parallel_desc2blob_placement_group_;
  bool is_job_conf_frozen_;
  bool has_job_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
