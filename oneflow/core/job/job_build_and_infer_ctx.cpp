#include "oneflow/core/job/job_build_and_infer_ctx.h"

namespace oneflow {

JobBuildAndInferCtx::JobBuildAndInferCtx(const std::string& job_name) { TODO(); }

Maybe<void> JobBuildAndInferCtx::SetJobConf(const JobConfigProto& job_conf) { TODO(); }

Maybe<void> JobBuildAndInferCtx::AddAndInferInputOp(const OperatorConf& op_conf,
                                                    int64_t parallel_num) {
  TODO();
}

Maybe<void> JobBuildAndInferCtx::AddAndInferNonInputOp(const OperatorConf& op_conf,
                                                       int64_t parallel_num) {
  TODO();
}

Maybe<void> JobBuildAndInferCtx::AddLossLogicalBlobName(const std::string& lbn) { TODO(); }

bool JobBuildAndInferCtx::HasJobConf() const { TODO(); }

Maybe<Shape> JobBuildAndInferCtx::GetStaticShape(const std::string& lbn) const { TODO(); }

Maybe<DataType> JobBuildAndInferCtx::GetDataType(const std::string& lbn) const { TODO(); }

Maybe<bool> JobBuildAndInferCtx::GetHasBatchDim(const std::string& lbn) const { TODO(); }

Maybe<bool> JobBuildAndInferCtx::GetHasSplitDim(const std::string& lbn) const { TODO(); }

Maybe<int64_t> JobBuildAndInferCtx::GetSplitDim(const std::string& lbn) const { TODO(); }

Maybe<ParallelDesc> JobBuildAndInferCtx::GetParallelDesc(const std::string& lbn) const { TODO(); }

const Job& JobBuildAndInferCtx::job() const { TODO(); }

}  // namespace oneflow
