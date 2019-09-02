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

std::shared_ptr<ErrorProto> GenJobBuildAndInferError(JobBuildAndInferError err_code,
                                                     std::string msg);

class JobBuildAndInferCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobBuildAndInferCtx);
  JobBuildAndInferCtx(Job* job, int64_t job_id);
  ~JobBuildAndInferCtx() = default;

  Maybe<void> SetJobConf(const JobConfigProto& job_conf);
  Maybe<void> AddAndInferOp(const OperatorConf& op_conf, const ParallelConf& parallel_conf);
  Maybe<void> AddLossLogicalBlobName(const std::string& lbn);
  Maybe<void> AddPlacementGroup(const PlacementGroup& placement_group);

  Maybe<void> CheckJob() const;

  bool HasJobConf() const;
  Maybe<Shape> GetStaticShape(const std::string& lbn) const;
  Maybe<DataType> GetDataType(const std::string& lbn) const;
  Maybe<OptInt64> GetBatchAxis(const std::string& lbn) const;
  Maybe<bool> GetHasSplitAxisFromProducerView(const std::string& lbn) const;
  Maybe<int64_t> GetSplitAxisFromProducerView(const std::string& lbn) const;
  Maybe<ParallelDesc> GetParallelDescFromProducerView(const std::string& lbn) const;

  const Job& job() const;

 private:
  Maybe<void> GenOpProducedEmptyLogicalBlobDesc(Operator* op);
  Maybe<void> CheckPlacement() const;
  Maybe<void> CheckJobConf() const;

  Job* job_;
  int64_t job_id_;
  HashMap<LogicalBlobId, OptInt64> lbi2batch_axis_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
  HashMap<std::string, std::shared_ptr<Operator>> op_name2op_;
  bool is_job_conf_frozen_;
  bool has_job_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
