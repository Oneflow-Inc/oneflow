#ifndef ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
#define ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

class JobBuildAndInferCtx {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobBuildAndInferCtx);
  JobBuildAndInferCtx(const std::string& job_name);
  ~JobBuildAndInferCtx() = default;

  Maybe<void> SetJobConf(const JobConfigProto& job_conf);
  Maybe<void> AddAndInferInputOp(const OperatorConf& op_conf, int64_t parallel_num);
  Maybe<void> AddAndInferNonInputOp(const OperatorConf& op_conf, int64_t parallel_num);
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
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_JOB_BUILD_AND_INFER_CTX_H_
