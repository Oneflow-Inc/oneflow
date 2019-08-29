#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"

std::string JobBuildAndInferCtx_NewAndEnter(const std::string& job_name) { TODO(); }

std::pair<std::string, std::string> JobBuildAndInferCtx_GetCurrentJobName() { TODO(); }

void JobBuildAndInferCtx_Leave() { TODO(); }

std::string CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf) { TODO(); }

std::string CurJobBuildAndInferCtx_AddAndInferInputOp(const std::string& serialized_op_conf) {
  TODO();
}

std::string CurJobBuildAndInferCtx_AddAndInferNonInputOp(const std::string& serialized_op_conf) {
  TODO();
}

std::string CurJobBuildAndInferCtx_AddLossLogicalBlobName(const std::string& lbn) { TODO(); }

bool CurJobBuildAndInferCtx_HasJobConf() { TODO(); }

std::pair<std::string, std::string> JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(
    const std::string& job_name, const std::string& lbn) {
  TODO();
}

std::pair<long long, std::string> JobBuildAndInferCtx_GetDataType(const std::string& job_name,
                                                                  const std::string& lbn) {
  TODO();
}

std::pair<bool, std::string> JobBuildAndInferCtx_GetHasBatchDim(const std::string& job_name,
                                                                const std::string& lbn) {
  TODO();
}

std::pair<bool, std::string> JobBuildAndInferCtx_GetHasSplitDimFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  TODO();
}

std::pair<long long, std::string> JobBuildAndInferCtx_GetSplitDimFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  TODO();
}

std::pair<std::string, std::string> JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  TODO();
}
