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

void CurJobBuildAndInferCtx_AddAndInferOp(const std::string& serialized_op_conf,
                                          const std::string& serialized_parallel_conf,
                                          std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferOp(serialized_op_conf, serialized_parallel_conf)
      .GetDataAndSerializedErrorProto(error_str);
}

void CurJobBuildAndInferCtx_AddAndInferMirrorOp(const std::string& serialized_op_conf,
                                                const std::string& serialized_parallel_conf,
                                                std::string* error_str) {
  return oneflow::CurJobBuildAndInferCtx_AddAndInferMirrorOp(serialized_op_conf,
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

long long JobBuildAndInferCtx_GetNumOfLoDLevels(const std::string& job_name, const std::string& lbn,
                                                std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_GetNumOfLoDLevels(job_name, lbn)
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

bool JobBuildAndInferCtx_IsMirrorBlob(const std::string& job_name, const std::string& lbn,
                                      std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_IsMirrorBlob(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, false);
}

int JobBuildAndInferCtx_MirrorBlobGetNumSubLbi(const std::string& job_name, const std::string& lbn,
                                               std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobGetNumSubLbi(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, 0);
}

std::string JobBuildAndInferCtx_MirrorBlobGetSerializedSubLbi(const std::string& job_name,
                                                              const std::string& lbn, int index,
                                                              std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobGetSubLbi(job_name, lbn, index)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string JobBuildAndInferCtx_MirrorBlobGetSerializedIdListAsStaticShape(
    const std::string& job_name, const std::string& lbn, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobGetSerializedIdListAsStaticShape(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

long long JobBuildAndInferCtx_MirrorBlobGetDataType(const std::string& job_name,
                                                    const std::string& lbn,
                                                    std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobGetDataType(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, 0LL);
}

bool JobBuildAndInferCtx_MirrorBlobIsDynamic(const std::string& job_name, const std::string& lbn,
                                             std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobIsDynamic(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, false);
}

long long JobBuildAndInferCtx_MirrorBlobGetNumOfLoDLevels(const std::string& job_name,
                                                          const std::string& lbn,
                                                          std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobGetNumOfLoDLevels(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, 0LL);
}

std::string JobBuildAndInferCtx_MirrorBlobGetBatchAxis(const std::string& job_name,
                                                       const std::string& lbn,
                                                       std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobGetBatchAxis(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string JobBuildAndInferCtx_MirrorBlobGetSplitAxisFromProducerView(const std::string& job_name,
                                                                       const std::string& lbn,
                                                                       std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobGetSplitAxisFromProducerView(job_name, lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}

std::string JobBuildAndInferCtx_MirrorBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn, std::string* error_str) {
  return oneflow::JobBuildAndInferCtx_MirrorBlobGetSerializedParallelConfFromProducerView(job_name,
                                                                                          lbn)
      .GetDataAndSerializedErrorProto(error_str, "");
}
