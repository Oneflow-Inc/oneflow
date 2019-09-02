#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/python/job_build_and_infer_helper.h"

std::string JobBuildAndInferCtx_Open(const std::string& job_name) {
  using namespace oneflow;
  auto maybe_ok = TRY(Global<JobBuildAndInferCtxMgr>::Get()->OpenJobBuildAndInferCtx(job_name));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return Error::Ok();
}

std::string JobBuildAndInferCtx_GetCurrentJobName(std::string* error_str) {
  using namespace oneflow;
  auto maybe_ok = TRY(Global<JobBuildAndInferCtxMgr>::Get()->GetCurrentJobName());
  if (maybe_ok.IsOk()) {
    *error_str = Error::Ok();
    return *maybe_ok.data();
  } else {
    PbMessage2TxtString(*maybe_ok.error(), error_str);
    return "";
  }
}

void JobBuildAndInferCtx_Close() {
  using namespace oneflow;
  Global<JobBuildAndInferCtxMgr>::Get()->CloseCurrentJobBuildAndInferCtx();
}

std::string CurJobBuildAndInferCtx_CheckJob() {
  using namespace oneflow;
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return error_str; }
  auto maybe_ok = TRY(ctx->CheckJob());
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return Error::Ok();
}

std::string CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf) {
  using namespace oneflow;
  // parse
  JobConfigProto job_conf;
  if (TxtString2PbMessage(serialized_job_conf, &job_conf) == false) {
    return Error::ProtoParseFailedError() << "job conf parse failed";
  }
  // get current JobBuildandInferCtx
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return error_str; }
  // set job_conf
  auto maybe_ok = TRY(ctx->SetJobConf(job_conf));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return Error::Ok();
}

bool CurJobBuildAndInferCtx_HasJobConf(std::string* error_str) {
  using namespace oneflow;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(error_str);
  if (ctx == nullptr) { return false; }
  *error_str = Error::Ok();
  return ctx->HasJobConf();
}

std::string CurJobBuildAndInferCtx_AddAndInferOp(const std::string& serialized_op_conf,
                                                 const std::string& serialized_parallel_conf) {
  using namespace oneflow;
  // parse
  OperatorConf op_conf;
  if (TxtString2PbMessage(serialized_op_conf, &op_conf) == false) {
    return Error::ProtoParseFailedError() << "operator conf parse failed";
  }
  ParallelConf parallel_conf;
  if (TxtString2PbMessage(serialized_parallel_conf, &parallel_conf) == false) {
    return Error::ProtoParseFailedError() << "parallel conf parse failed";
  }
  // get current JobBuildandInferCtx
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return error_str; }
  // add and infer input_op
  auto maybe_ok = TRY(ctx->AddAndInferOp(op_conf, parallel_conf));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return Error::Ok();
}

std::string CurJobBuildAndInferCtx_AddLossLogicalBlobName(const std::string& lbn) {
  using namespace oneflow;
  // get current JobBuildandInferCtx
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return error_str; }
  // add loss_lbn
  auto maybe_ok = TRY(ctx->AddLossLogicalBlobName(lbn));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return Error::Ok();
}

std::string CurJobBuildAndInferCtx_AddPlacementGroup(const std::string& serialized_placement_grp) {
  using namespace oneflow;
  // parse
  PlacementGroup placement_group;
  if (TxtString2PbMessage(serialized_placement_grp, &placement_group) == false) {
    return Error::ProtoParseFailedError() << "placement group parse failed";
  }
  // get current JobBuildandInferCtx
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return error_str; }
  // add and infer input_op
  auto maybe_ok = TRY(ctx->AddPlacementGroup(placement_group));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return Error::Ok();
}

std::string JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(const std::string& job_name,
                                                                 const std::string& lbn,
                                                                 std::string* error_str) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  Int64List id_list;
  if (maybe_ctx.IsOk() == false) {
    PbMessage2TxtString(*maybe_ctx.error(), error_str);
    return PbMessage2TxtString(id_list);
  }
  auto maybe_shape = TRY(maybe_ctx.data()->GetStaticShape(lbn));
  if (maybe_shape.IsOk() == false) {
    PbMessage2TxtString(*maybe_shape.error(), error_str);
    return PbMessage2TxtString(id_list);
  }
  *error_str = Error::Ok();
  const auto& shape = *maybe_shape.data();
  *id_list.mutable_value() = {shape.dim_vec().begin(), shape.dim_vec().end()};
  return PbMessage2TxtString(id_list);
}

long long JobBuildAndInferCtx_GetDataType(const std::string& job_name, const std::string& lbn,
                                          std::string* error_str) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    PbMessage2TxtString(*maybe_ctx.error(), error_str);
    return 0LL;
  }
  auto maybe_data_type = TRY(maybe_ctx.data()->GetDataType(lbn));
  if (maybe_data_type.IsOk() == false) {
    PbMessage2TxtString(*maybe_data_type.error(), error_str);
    return 0LL;
  }
  *error_str = Error::Ok();
  return *maybe_data_type.data();
}

std::string JobBuildAndInferCtx_GetBatchAxis(const std::string& job_name, const std::string& lbn,
                                             std::string* error_str) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    PbMessage2TxtString(*maybe_ctx.error(), error_str);
    return PbMessage2TxtString(OptInt64());
  }
  auto maybe_has_batch_dim = TRY(maybe_ctx.data()->GetBatchAxis(lbn));
  if (maybe_has_batch_dim.IsOk() == false) {
    PbMessage2TxtString(*maybe_has_batch_dim.error(), error_str);
    return PbMessage2TxtString(OptInt64());
  }
  *error_str = Error::Ok();
  return PbMessage2TxtString(*maybe_has_batch_dim.data());
}

bool JobBuildAndInferCtx_GetHasSplitAxisFromProducerView(const std::string& job_name,
                                                         const std::string& lbn,
                                                         std::string* error_str) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    PbMessage2TxtString(*maybe_ctx.error(), error_str);
    return false;
  }
  auto maybe_has_split_axis = TRY(maybe_ctx.data()->GetHasSplitAxisFromProducerView(lbn));
  if (maybe_has_split_axis.IsOk() == false) {
    PbMessage2TxtString(*maybe_has_split_axis.error(), error_str);
    return false;
  }
  *error_str = Error::Ok();
  return *maybe_has_split_axis.data();
}

long long JobBuildAndInferCtx_GetSplitAxisFromProducerView(const std::string& job_name,
                                                           const std::string& lbn,
                                                           std::string* error_str) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    PbMessage2TxtString(*maybe_ctx.error(), error_str);
    return 0LL;
  }
  auto maybe_split_axis = TRY(maybe_ctx.data()->GetSplitAxisFromProducerView(lbn));
  if (maybe_split_axis.IsOk() == false) {
    PbMessage2TxtString(*maybe_split_axis.error(), error_str);
    return 0LL;
  }
  *error_str = Error::Ok();
  return *maybe_split_axis.data();
}

std::string JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn, std::string* error_str) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    PbMessage2TxtString(*maybe_ctx.error(), error_str);
    return PbMessage2TxtString(ParallelConf());
  }
  auto maybe_parallel_conf = TRY(maybe_ctx.data()->GetParallelDescFromProducerView(lbn));
  if (maybe_parallel_conf.IsOk() == false) {
    PbMessage2TxtString(*maybe_parallel_conf.error(), error_str);
    return PbMessage2TxtString(ParallelConf());
  }
  *error_str = Error::Ok();
  return PbMessage2TxtString(maybe_parallel_conf.data()->parallel_conf());
}
