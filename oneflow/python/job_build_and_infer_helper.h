#ifndef ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_
#define ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

namespace {

Maybe<JobBuildAndInferCtx*> GetCurInferCtx() {
  const auto& job_name = *JUST(Global<JobBuildAndInferCtxMgr>::Get()->GetCurrentJobName());
  return Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name);
}

}  // namespace

Maybe<void> JobBuildAndInferCtx_Open(const std::string& job_name) {
  return Global<JobBuildAndInferCtxMgr>::Get()->OpenJobBuildAndInferCtx(job_name);
}

Maybe<std::string> JobBuildAndInferCtx_GetCurrentJobName() {
  return Global<JobBuildAndInferCtxMgr>::Get()->GetCurrentJobName();
}

Maybe<void> JobBuildAndInferCtx_Close() {
  Global<JobBuildAndInferCtxMgr>::Get()->CloseCurrentJobBuildAndInferCtx();
  return Maybe<void>::Ok();
}

Maybe<void> CurJobBuildAndInferCtx_CheckJob() { return JUST(GetCurInferCtx())->CheckJob(); }

Maybe<void> CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf) {
  // parse
  JobConfigProto job_conf;
  OF_CHECK(TxtString2PbMessage(serialized_job_conf, &job_conf)) << "job conf parse failed";
  return JUST(GetCurInferCtx())->SetJobConf(job_conf);
}

Maybe<bool> CurJobBuildAndInferCtx_HasJobConf() { return JUST(GetCurInferCtx())->HasJobConf(); }

Maybe<void> CurJobBuildAndInferCtx_AddAndInferOp(const std::string& op_conf_str,
                                                 const std::string& parallel_conf_str) {
  OperatorConf op_conf;
  OF_CHECK(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  ParallelConf parallel_conf;
  OF_CHECK(TxtString2PbMessage(parallel_conf_str, &parallel_conf)) << "parallel conf parse failed";
  return JUST(GetCurInferCtx())->AddAndInferOps(op_conf, parallel_conf);
}

Maybe<void> CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(
    const std::string& lbi_uuid_pair_str) {
  LbiAndDiffWatcherUuidPair lbi_uuid_pair;
  OF_CHECK(TxtString2PbMessage(lbi_uuid_pair_str, &lbi_uuid_pair))
      << "LbiAndDiffWatcherUuidPair parse failed";
  return Global<JobBuildAndInferCtxMgr>::Get()->AddLbiAndDiffWatcherUuidPair(lbi_uuid_pair);
}

Maybe<std::string> JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(const std::string& job_name,
                                                                        const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  const auto& shape = JUST(ctx->GetStaticShape(lbn));
  Int64List id_list;
  *id_list.mutable_value() = {shape->dim_vec().begin(), shape->dim_vec().end()};
  return PbMessage2TxtString(id_list);
}

Maybe<long long> JobBuildAndInferCtx_GetDataType(const std::string& job_name,
                                                 const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return *JUST(ctx->GetDataType(lbn));
}

Maybe<bool> JobBuildAndInferCtx_IsDynamic(const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->IsDynamic(lbn);
}

Maybe<long long> JobBuildAndInferCtx_GetNumOfLoDLevels(const std::string& job_name,
                                                       const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->GetNumOfLoDLevels(lbn);
}

Maybe<std::string> JobBuildAndInferCtx_GetBatchAxis(const std::string& job_name,
                                                    const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->GetBatchAxis(lbn)));
}

Maybe<std::string> JobBuildAndInferCtx_GetSplitAxisFromProducerView(const std::string& job_name,
                                                                    const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->GetSplitAxisFromProducerView(lbn)));
}

Maybe<std::string> JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(JUST(ctx->GetParallelDescFromProducerView(lbn))->parallel_conf());
}

Maybe<void> CurJobBuildAndInferCtx_AddLossLogicalBlobName(const std::string& lbn) {
  return JUST(GetCurInferCtx())->AddLossLogicalBlobName(lbn);
}

Maybe<bool> JobBuildAndInferCtx_IsMirrorBlob(const std::string& job_name,
                                             const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->IsMirrorBlob(mirror_blob_name);
}

Maybe<int> JobBuildAndInferCtx_NumLbiInMirrorBlob(const std::string& job_name,
                                                  const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->NumLbiInMirrorBlob(mirror_blob_name);
}

Maybe<std::string> JobBuildAndInferCtx_GetLbiInMirrorBlob(const std::string& job_name,
                                                          const std::string& mirror_blob_name,
                                                          int index) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->GetLbiInMirrorBlob(mirror_blob_name, index)));
}

Maybe<std::string> JobBuildAndInferCtx_MirrorBlobGetSerializedIdListAsStaticShape(
    const std::string& job_name, const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  const auto& shape = JUST(ctx->MirrorBlobGetStaticShape(mirror_blob_name));
  Int64List id_list;
  *id_list.mutable_value() = {shape->dim_vec().begin(), shape->dim_vec().end()};
  return PbMessage2TxtString(id_list);
}

Maybe<long long> JobBuildAndInferCtx_MirrorBlobGetDataType(const std::string& job_name,
                                                           const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return *JUST(ctx->MirrorBlobGetDataType(mirror_blob_name));
}

Maybe<bool> JobBuildAndInferCtx_MirrorBlobIsDynamic(const std::string& job_name,
                                                    const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->MirrorBlobIsDynamic(mirror_blob_name);
}

Maybe<long long> JobBuildAndInferCtx_MirrorBlobGetNumOfLoDLevels(
    const std::string& job_name, const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->MirrorBlobGetNumOfLoDLevels(mirror_blob_name);
}

Maybe<std::string> JobBuildAndInferCtx_MirrorBlobGetBatchAxis(const std::string& job_name,
                                                              const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->MirrorBlobGetBatchAxis(mirror_blob_name)));
}

Maybe<std::string> JobBuildAndInferCtx_MirrorBlobGetSplitAxisFromProducerView(
    const std::string& job_name, const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->MirrorBlobGetSplitAxisFromProducerView(mirror_blob_name)));
}

Maybe<std::string> JobBuildAndInferCtx_MirrorBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& mirror_blob_name) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(
      JUST(ctx->MirrorBlobGetParallelDescFromProducerView(mirror_blob_name))->parallel_conf());
}

}  // namespace oneflow

#endif  // ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_
