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

Error GenJobBuildAndInferError(JobBuildAndInferError err_code, std::string msg);

class JobBuildAndInferCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobBuildAndInferCtx);
  JobBuildAndInferCtx(const std::string& job_name);
  ~JobBuildAndInferCtx() = default;

  Maybe<void> SetJobConf(const JobConfigProto& job_conf);
  Maybe<void> AddAndInferInputOp(const OperatorConf& op_conf);
  Maybe<void> AddAndInferNonInputOp(const OperatorConf& op_conf);
  Maybe<void> AddLossLogicalBlobName(const std::string& lbn);

  bool HasJobConf() const;
  Maybe<Shape> GetStaticShape(const std::string& lbn) const;
  Maybe<DataType> GetDataType(const std::string& lbn) const;
  Maybe<bool> GetHasBatchDim(const std::string& lbn) const;
  Maybe<bool> GetHasSplitDim(const std::string& lbn) const;
  Maybe<int64_t> GetSplitDim(const std::string& lbn) const;
  Maybe<ParallelDesc> GetParallelDesc(const std::string& lbn) const;

  const Job& job() const;

 private:
  Maybe<void> GenOpProducedEmptyLogicalBlobDesc(Operator* op);

  Job job_;
  HashMap<LogicalBlobId, bool> lbi2has_batch_dim_;
  HashMap<LogicalBlobId, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
  HashMap<std::string, std::shared_ptr<Operator>> op_name2op_;
  bool is_job_conf_frozen_;
  bool has_job_conf_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
