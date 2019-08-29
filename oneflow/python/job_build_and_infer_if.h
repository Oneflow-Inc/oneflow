#include <google/protobuf/text_format.h>
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"

std::string JobBuildAndInferCtx_NewAndEnter(const std::string& job_name) { TODO(); }

std::pair<std::string, std::string> JobBuildAndInferCtx_GetCurrentJobName(
    const std::string& job_name) {
  TODO();
}

std::string JobBuildAndInferCtx_Leave() { TODO(); }

std::string JobBuildAndInferCtx_SetJobConf(const std::string& job_name,
                                           const std::string& serialized_job_conf) {
  TODO();
}

std::string JobBuildAndInferCtx_AddAndInferInputOp(const std::string& job_name,
                                                   const std::string& serialized_op_conf) {
  TODO();
}

std::string JobBuildAndInferCtx_AddAndInferNonInputOp(const std::string& job_name,
                                                      const std::string& serialized_op_conf) {
  TODO();
}

std::string JobBuildAndInferCtx_AddLossLogicalBlobName(const std::string& job_name,
                                                       const std::string& lbn) {
  TODO();
}

bool JobBuildAndInferCtx_HasJobConf(const std::string& job_name) { TODO(); }

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
