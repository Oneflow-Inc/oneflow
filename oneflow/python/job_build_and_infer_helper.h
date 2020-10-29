/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_
#define ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_

#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

bool EagerExecutionEnabled() { return *Global<bool, EagerExecution>::Get(); }

namespace {

Maybe<JobBuildAndInferCtxMgr*> GlobalJobBuildAndInferCtxMgr() {
  if (EagerExecutionEnabled()) {
    return JUST(GlobalMaybe<EagerJobBuildAndInferCtxMgr>());
  } else {
    return JUST(GlobalMaybe<LazyJobBuildAndInferCtxMgr>());
  }
}

Maybe<JobBuildAndInferCtx*> GetJobBuildAndInferCtx(const std::string& job_name) {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->FindJobBuildAndInferCtx(job_name);
}

Maybe<JobBuildAndInferCtx*> GetCurInferCtx() {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->FindJobBuildAndInferCtx(*JUST(mgr->GetCurrentJobName()));
}

}  // namespace

Maybe<void> JobBuildAndInferCtx_Open(const std::string& job_name) {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->OpenJobBuildAndInferCtx(job_name);
}

Maybe<std::string> JobBuildAndInferCtx_GetCurrentJobName() {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->GetCurrentJobName();
}

Maybe<void> JobBuildAndInferCtx_Close() {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  JUST(mgr->CloseCurrentJobBuildAndInferCtx());
  return Maybe<void>::Ok();
}

Maybe<void> CurJobBuildAndInferCtx_CheckJob() { return JUST(GetCurInferCtx())->CheckJob(); }

Maybe<void> CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf) {
  // parse
  JobConfigProto job_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(serialized_job_conf, &job_conf)) << "job conf parse failed";
  return JUST(GetCurInferCtx())->SetJobConf(job_conf);
}

Maybe<void> CurJobBuildAndInferCtx_SetTrainConf(const std::string& train_conf_str) {
  TrainConf train_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(train_conf_str, &train_conf)) << "train conf parse failed";
  return JUST(GetCurInferCtx())->SetTrainConf(train_conf);
}

Maybe<void> CurJobBuildAndInferCtx_Complete() { return JUST(GetCurInferCtx())->Complete(); }

Maybe<bool> CurJobBuildAndInferCtx_HasJobConf() { return JUST(GetCurInferCtx())->HasJobConf(); }

Maybe<std::string> CurJobBuildAndInferCtx_AddAndInferMirroredOp(const std::string& op_conf_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  auto* ctx = JUST(GetCurInferCtx());
  const auto& op_attribute = JUST(ctx->AddAndInferMirroredOp(op_conf));
  return PbMessage2TxtString(*op_attribute);
}

Maybe<std::string> CurJobBuildAndInferCtx_AddAndInferConsistentOp(const std::string& op_conf_str) {
  OperatorConf op_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(op_conf_str, &op_conf)) << "operator conf parse failed";
  auto* ctx = JUST(GetCurInferCtx());
  const auto& op_attribute = JUST(ctx->AddAndInferConsistentOp(op_conf));
  return PbMessage2TxtString(*op_attribute);
}

Maybe<void> CurJobBuildAndInferCtx_AddLbiAndDiffWatcherUuidPair(
    const std::string& lbi_uuid_pair_str) {
  auto* ctx = JUST(GetCurInferCtx());
  LbiAndDiffWatcherUuidPair lbi_uuid_pair;
  CHECK_OR_RETURN(TxtString2PbMessage(lbi_uuid_pair_str, &lbi_uuid_pair))
      << "LbiAndDiffWatcherUuidPair parse failed";
  return ctx->AddLbiAndDiffWatcherUuidPair(lbi_uuid_pair);
}

Maybe<std::string> JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(const std::string& job_name,
                                                                        const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  const auto& shape = JUST(ctx->GetStaticShape(lbn));
  Int64List id_list;
  *id_list.mutable_value() = {shape->dim_vec().begin(), shape->dim_vec().end()};
  return PbMessage2TxtString(id_list);
}

Maybe<long long> JobBuildAndInferCtx_GetDataType(const std::string& job_name,
                                                 const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return JUST(ctx->GetDataType(lbn));
}

Maybe<bool> JobBuildAndInferCtx_IsDynamic(const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return ctx->IsDynamic(lbn);
}

Maybe<bool> JobBuildAndInferCtx_DisableBoxing(const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return ctx->DisableBoxing(lbn);
}

Maybe<bool> JobBuildAndInferCtx_IsTensorList(const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return ctx->IsTensorList(lbn);
}

Maybe<std::string> JobBuildAndInferCtx_GetBatchAxis(const std::string& job_name,
                                                    const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->GetBatchAxis(lbn)));
}

Maybe<std::string> JobBuildAndInferCtx_GetSplitAxisFromProducerView(const std::string& job_name,
                                                                    const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->GetSplitAxisFromProducerView(lbn)));
}

Maybe<std::string> JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(JUST(ctx->GetParallelDescFromProducerView(lbn))->parallel_conf());
}

Maybe<void> CurJobBuildAndInferCtx_AddLossLogicalBlobName(const std::string& lbn) {
  return JUST(GetCurInferCtx())->AddLossLogicalBlobName(lbn);
}

Maybe<bool> JobBuildAndInferCtx_IsMirroredBlob(const std::string& job_name,
                                               const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return ctx->IsMirroredBlob(lbn);
}

Maybe<int> JobBuildAndInferCtx_MirroredBlobGetNumSubLbi(const std::string& job_name,
                                                        const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return ctx->MirroredBlobGetNumSubLbi(lbn);
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSubLbi(const std::string& job_name,
                                                             const std::string& lbn, int index) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->MirroredBlobGetSubLbi(lbn, index)));
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSerializedIdListAsStaticShape(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  const auto& shape = JUST(ctx->MirroredBlobGetStaticShape(lbn));
  Int64List id_list;
  *id_list.mutable_value() = {shape->dim_vec().begin(), shape->dim_vec().end()};
  return PbMessage2TxtString(id_list);
}

Maybe<long long> JobBuildAndInferCtx_MirroredBlobGetDataType(const std::string& job_name,
                                                             const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return JUST(ctx->MirroredBlobGetDataType(lbn));
}

Maybe<bool> JobBuildAndInferCtx_MirroredBlobIsDynamic(const std::string& job_name,
                                                      const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return ctx->MirroredBlobIsDynamic(lbn);
}

Maybe<bool> JobBuildAndInferCtx_MirroredBlobIsTensorList(const std::string& job_name,
                                                         const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return ctx->MirroredBlobIsTensorList(lbn);
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetBatchAxis(const std::string& job_name,
                                                                const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->MirroredBlobGetBatchAxis(lbn)));
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSplitAxisFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(*JUST(ctx->MirroredBlobGetSplitAxisFromProducerView(lbn)));
}

Maybe<std::string> JobBuildAndInferCtx_MirroredBlobGetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  auto* ctx = JUST(GetJobBuildAndInferCtx(job_name));
  return PbMessage2TxtString(
      JUST(ctx->MirroredBlobGetParallelDescFromProducerView(lbn))->parallel_conf());
}

}  // namespace oneflow

#endif  // ONEFLOW_PYTHON_JOB_BUILD_AND_INFER_HELPER_H_
