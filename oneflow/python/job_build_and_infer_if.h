#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/python/job_build_and_infer_helper.h"

void JobBuildAndInferCtx_Open(const std::string& job_name, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_Open(job_name).GetDataAndSerializedErrorProto(error_str);
}

std::string JobBuildAndInferCtx_GetCurrentJobName(std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_GetCurrentJobName().GetDataAndSerializedErrorProto(error_str,
                                                                                         "");
}

void JobBuildAndInferCtx_Close(std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_Close().GetDataAndSerializedErrorProto(error_str);
}

void CurJobBuildAndInferCtx_CheckJob(std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_CheckJob().GetDataAndSerializedErrorProto(error_str);
}

void CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf,
                                       std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_SetJobConf(serialized_job_conf)
      .GetDataAndSerializedErrorProto(error_str);
}

bool CurJobBuildAndInferCtx_HasJobConf(std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_HasJobConf().GetDataAndSerializedErrorProto(error_str,
                                                                                     false);
}

std::string CurJobBuildAndInferCtx_CheckAndCompleteUserOpConf(const std::string& serialized_op_conf,
                                                              std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_CheckAndCompleteUserOpConf(serialized_op_conf)
      .GetDataAndSerializedErrorProto(error_str, "");
}

void CurJobBuildAndInferCtx_AddAndInferOp(const std::string& serialized_op_conf,
                                          const std::string& serialized_parallel_conf,
                                          std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferOp(serialized_op_conf, serialized_parallel_conf)
      .GetDataAndSerializedErrorProto(error_str);
}

void CurJobBuildAndInferCtx_AddAndInferMirroredOp(const std::string& serialized_op_conf,
                                                  const std::string& serialized_parallel_conf,
                                                  std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferMirroredOp(serialized_op_conf,
                                                               serialized_parallel_conf)
      .GetDataAndSerializedErrorProto(error_str);
}

void CurJobBuildAndInferCtx_AddAndInferConsistentOp(const std::string& serialized_op_conf,
                                                    const std::string& serialized_parallel_conf,
                                                    std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferConsistentOp(serialized_op_conf,
                                                                 serialized_parallel_conf)
      .GetDataAndSerializedErrorProto(error_str);
}

void CurJobBuildAndInferCtx_AddLossLogicalBlobName(const std::string& lbn, std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_AddLossLogicalBlobName(lbn).GetDataAndSerializedErrorProto(
      error_str);
}

void CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(const std::string& lbi_uuid_pair,
                                                         std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(lbi_uuid_pair)
      .GetDataAndSerializedErrorProto(error_str);
}

std::string JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(const std::string& job_name,
                                                                 const std::string& lbn,
                                                                 std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

long long JobBuildAndInferCtx_GetDataType(const std::string& job_name, const std::string& lbn,
                                          std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_GetDataType(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, 0LL);
}

bool JobBuildAndInferCtx_IsDynamic(const std::string& job_name, const std::string& lbn,
                                   std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_IsDynamic(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, false);
}

bool JobBuildAndInferCtx_DisableBoxing(const std::string& job_name, const std::string& lbn,
                                       std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_DisableBoxing(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, false);
}

long long JobBuildAndInferCtx_IsTensorList(const std::string& job_name, const std::string& lbn,
                                           std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_IsTensorList(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, 0LL);
}

std::string JobBuildAndInferCtx_GetBatchAxis(const std::string& job_name, const std::string& lbn,
                                             std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_GetBatchAxis(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string JobBuildAndInferCtx_GetSplitAxisFromProducerView(const std::string& job_name,
                                                             const std::string& lbn,
                                                             std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_GetSplitAxisFromProducerView(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

bool JobBuildAndInferCtx_IsMirroredBlob(const std::string& job_name, const std::string& lbn,
                                        std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_IsMirroredBlob(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, false);
}

int JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(const std::string& job_name,
                                                 const std::string& lbn, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, 0);
}

std::string JobBuildAndInferCtx_MirroredBlobGetSerializedSubLbi(const std::string& job_name,
                                                                const std::string& lbn, int index,
                                                                std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSubLbi(job_name, lbn, index)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string JobBuildAndInferCtx_MirroredBlobGetSerializedIdListAsStaticShape(
    const std::string& job_name, const std::string& lbn, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSerializedIdListAsStaticShape(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

long long JobBuildAndInferCtx_MirroredBlobGetDataType(const std::string& job_name,
                                                      const std::string& lbn,
                                                      std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetDataType(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, 0LL);
}

bool JobBuildAndInferCtx_MirroredBlobIsDynamic(const std::string& job_name, const std::string& lbn,
                                               std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobIsDynamic(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, false);
}

bool JobBuildAndInferCtx_MirroredBlobIsTensorList(const std::string& job_name,
                                                  const std::string& lbn, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobIsTensorList(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, 0LL);
}

std::string JobBuildAndInferCtx_MirroredBlobGetBatchAxis(const std::string& job_name,
                                                         const std::string& lbn,
                                                         std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetBatchAxis(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(
    const std::string& job_name, const std::string& lbn, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
             job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}
