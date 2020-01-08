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

Maybe<void> CurJobBuildAndInferCtx_Complete() { return JUST(GetCurInferCtx())->Complete(); }

Maybe<bool> CurJobBuildAndInferCtx_HasJobConf() { return JUST(GetCurInferCtx())->HasJobConf(); }

Maybe<std::string> CurJobBuildAndInferCtx_CheckAndCompleteUserOpConf(
    const std::string& op_conf_str) {
  OperatorConf op_conf;
  OF_CHECK(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  return PbMessage2TxtString(*JUST(JUST(GetCurInferCtx())->CheckAndCompleteUserOpConf(op_conf)));
}

Maybe<void> CurJobBuildAndInferCtx_AddAndInferOp(const std::string& op_conf_str,
                                                 const std::string& parallel_conf_str) {
  OperatorConf op_conf;
  OF_CHECK(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  ParallelConf parallel_conf;
  OF_CHECK(TxtString2PbMessage(parallel_conf_str, &parallel_conf)) << "parallel conf parse failed";
  return JUST(GetCurInferCtx())->AddAndInferOp(op_conf, parallel_conf);
}

Maybe<void> CurJobBuildAndInferCtx_AddAndInferMirroredOp(const std::string& op_conf_str,
                                                         const std::string& parallel_conf_str) {
  OperatorConf op_conf;
  OF_CHECK(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  ParallelConf parallel_conf;
  OF_CHECK(TxtString2PbMessage(parallel_conf_str, &parallel_conf)) << "parallel conf parse failed";
  return JUST(GetCurInferCtx())->AddAndInferMirroredOp(op_conf, parallel_conf);
}

Maybe<void> CurJobBuildAndInferCtx_AddAndInferConsistentOp(const std::string& op_conf_str,
                                                           const std::string& parallel_conf_str) {
  OperatorConf op_conf;
  OF_CHECK(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  ParallelConf parallel_conf;
  OF_CHECK(TxtString2PbMessage(parallel_conf_str, &parallel_conf)) << "parallel conf parse failed";
  return JUST(GetCurInferCtx())->AddAndInferConsistentOp(op_conf, parallel_conf);
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

Maybe<bool> JobBuildAndInferCtx_DisableBoxing(const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->DisableBoxing(lbn);
}

Maybe<bool> JobBuildAndInferCtx_IsTensorList(const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->IsTensorList(lbn);
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

Maybe<bool> JobBuildAndInferCtx_IsMirroredBlob(const std::string& job_name,
                                               const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->IsMirroredBlob(lbn);
}

Maybe<int> JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(const std::string& job_name,
                                                        const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->MirroredBlobGetNumSubLbi(lbn);
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSubLbi(const std::string& job_name,
                                                             const std::string& lbn, int index) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->MirroredBlobGetSubLbi(lbn, index)));
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSerializedIdListAsStaticShape(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  const auto& shape = JUST(ctx->MirroredBlobGetStaticShape(lbn));
  Int64List id_list;
  *id_list.mutable_value() = {shape->dim_vec().begin(), shape->dim_vec().end()};
  return PbMessage2TxtString(id_list);
}

Maybe<long long> JobBuildAndInferCtx_MirroredBlobGetDataType(const std::string& job_name,
                                                             const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return *JUST(ctx->MirroredBlobGetDataType(lbn));
}

Maybe<bool> JobBuildAndInferCtx_MirroredBlobIsDynamic(const std::string& job_name,
                                                      const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->MirroredBlobIsDynamic(lbn);
}

Maybe<bool> JobBuildAndInferCtx_MirroredBlobIsTensorList(const std::string& job_name,
                                                         const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return ctx->MirroredBlobIsTensorList(lbn);
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetBatchAxis(const std::string& job_name,
                                                                const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->MirroredBlobGetBatchAxis(lbn)));
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->MirroredBlobGetSplitAxisFromProducerView(lbn)));
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(
      JUST(ctx->MirroredBlobGetParallelDescFromProducerView(lbn))->parallel_conf());
}

}  // namespace oneflow

#endif  // ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_
