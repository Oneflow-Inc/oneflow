#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/error_util.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/python/job_build_and_infer_helper.h"

std::string JobBuildAndInferCtx_NewAndEnter(const std::string& job_name) {
  using namespace oneflow;
  auto maybe_ok = TRY(Global<JobBuildAndInferCtxMgr>::Get()->EnterJobBuildAndInferCtx(job_name));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return PbMessage2TxtString(ErrorUtil::Ok());
}

std::pair<std::string, std::string> JobBuildAndInferCtx_GetCurrentJobName() {
  using namespace oneflow;
  auto maybe_ok = TRY(Global<JobBuildAndInferCtxMgr>::Get()->GetCurrentJobName());
  if (maybe_ok.IsOk()) {
    return std::make_pair(*maybe_ok.data(), PbMessage2TxtString(ErrorUtil::Ok()));
  } else {
    return std::make_pair(std::string(""), PbMessage2TxtString(*maybe_ok.error()));
  }
}

void JobBuildAndInferCtx_Leave() {
  using namespace oneflow;
  Global<JobBuildAndInferCtxMgr>::Get()->LeaveCurrentJobBuildAndInferCtx();
}

std::string CurJobBuildAndInferCtx_SetJobConf(const std::string& serialized_job_conf) {
  using namespace oneflow;
  // parse
  JobConfigProto job_conf;
  if (TxtString2PbMessage(serialized_job_conf, &job_conf) == false) {
    return PbMessage2TxtString(ErrorUtil::ProtoParseFailedError("job conf parse failed"));
  }
  // get current JobBuildandInferCtx
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return error_str; }
  // set job_conf
  auto maybe_ok = TRY(ctx->SetJobConf(job_conf));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return PbMessage2TxtString(ErrorUtil::Ok());
}

std::string CurJobBuildAndInferCtx_AddAndInferInputOp(const std::string& serialized_op_conf) {
  using namespace oneflow;
  // parse
  OperatorConf op_conf;
  if (TxtString2PbMessage(serialized_op_conf, &op_conf) == false) {
    return PbMessage2TxtString(ErrorUtil::ProtoParseFailedError("operator conf parse failed"));
  }
  // get current JobBuildandInferCtx
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return error_str; }
  // add and infer input_op
  auto maybe_ok = TRY(ctx->AddAndInferInputOp(op_conf));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return PbMessage2TxtString(ErrorUtil::Ok());
}

std::string CurJobBuildAndInferCtx_AddAndInferNonInputOp(const std::string& serialized_op_conf) {
  using namespace oneflow;
  // parse
  OperatorConf op_conf;
  if (TxtString2PbMessage(serialized_op_conf, &op_conf) == false) {
    return PbMessage2TxtString(ErrorUtil::ProtoParseFailedError("operator conf parse failed"));
  }
  // get current JobBuildandInferCtx
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return error_str; }
  // add and infer input_op
  auto maybe_ok = TRY(ctx->AddAndInferNonInputOp(op_conf));
  if (maybe_ok.IsOk() == false) { return PbMessage2TxtString(*maybe_ok.error()); }
  return PbMessage2TxtString(ErrorUtil::Ok());
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
  return PbMessage2TxtString(ErrorUtil::Ok());
}

std::pair<bool, std::string> CurJobBuildAndInferCtx_HasJobConf() {
  using namespace oneflow;
  std::string error_str;
  JobBuildAndInferCtx* ctx = JobBuildAndInferHelper::GetCurInferCtx(&error_str);
  if (ctx == nullptr) { return std::make_pair(false, error_str); }
  return std::make_pair(ctx->HasJobConf(), PbMessage2TxtString(ErrorUtil::Ok()));
}

std::pair<std::string, std::string> JobBuildAndInferCtx_GetSerializedIdListAsStaticShape(
    const std::string& job_name, const std::string& lbn) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  Int64List id_list;
  if (maybe_ctx.IsOk() == false) {
    return std::make_pair(PbMessage2TxtString(id_list), PbMessage2TxtString(*maybe_ctx.error()));
  }
  auto maybe_shape = TRY(maybe_ctx.data()->GetStaticShape(lbn));
  if (maybe_shape.IsOk() == false) {
    return std::make_pair(PbMessage2TxtString(id_list), PbMessage2TxtString(*maybe_shape.error()));
  }
  const auto& shape = *maybe_shape.data();
  *id_list.mutable_value() = {shape.dim_vec().begin(), shape.dim_vec().end()};
  return std::make_pair(PbMessage2TxtString(id_list), PbMessage2TxtString(ErrorUtil::Ok()));
}

std::pair<long long, std::string> JobBuildAndInferCtx_GetDataType(const std::string& job_name,
                                                                  const std::string& lbn) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    return std::make_pair(0LL, PbMessage2TxtString(*maybe_ctx.error()));
  }
  auto maybe_data_type = TRY(maybe_ctx.data()->GetDataType(lbn));
  if (maybe_data_type.IsOk() == false) {
    return std::make_pair(0LL, PbMessage2TxtString(*maybe_data_type.error()));
  }
  long long dtype = *maybe_data_type.data();
  return std::make_pair(dtype, PbMessage2TxtString(ErrorUtil::Ok()));
}

std::pair<bool, std::string> JobBuildAndInferCtx_GetHasBatchDim(const std::string& job_name,
                                                                const std::string& lbn) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    return std::make_pair(0LL, PbMessage2TxtString(*maybe_ctx.error()));
  }
  auto maybe_has_batch_dim = TRY(maybe_ctx.data()->GetHasBatchDim(lbn));
  if (maybe_has_batch_dim.IsOk() == false) {
    return std::make_pair(false, PbMessage2TxtString(*maybe_has_batch_dim.error()));
  }
  return std::make_pair(*maybe_has_batch_dim.data(), PbMessage2TxtString(ErrorUtil::Ok()));
}

std::pair<bool, std::string> JobBuildAndInferCtx_GetHasSplitDimFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    return std::make_pair(0LL, PbMessage2TxtString(*maybe_ctx.error()));
  }
  auto maybe_has_split_dim = TRY(maybe_ctx.data()->GetHasSplitDimFromProducerView(lbn));
  if (maybe_has_split_dim.IsOk() == false) {
    return std::make_pair(false, PbMessage2TxtString(*maybe_has_split_dim.error()));
  }
  return std::make_pair(*maybe_has_split_dim.data(), PbMessage2TxtString(ErrorUtil::Ok()));
}

std::pair<long long, std::string> JobBuildAndInferCtx_GetSplitDimFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    return std::make_pair(0LL, PbMessage2TxtString(*maybe_ctx.error()));
  }
  auto maybe_split_dim = TRY(maybe_ctx.data()->GetSplitDimFromProducerView(lbn));
  if (maybe_split_dim.IsOk() == false) {
    return std::make_pair(0LL, PbMessage2TxtString(*maybe_split_dim.error()));
  }
  long long split_dim = *maybe_split_dim.data();
  return std::make_pair(split_dim, PbMessage2TxtString(ErrorUtil::Ok()));
}

std::pair<std::string, std::string> JobBuildAndInferCtx_GetSerializedParallelConfFromProducerView(
    const std::string& job_name, const std::string& lbn) {
  using namespace oneflow;
  auto maybe_ctx = TRY(Global<JobBuildAndInferCtxMgr>::Get()->FindJobBuildAndInferCtx(job_name));
  if (maybe_ctx.IsOk() == false) {
    return std::make_pair(PbMessage2TxtString(ParallelConf()),
                          PbMessage2TxtString(*maybe_ctx.error()));
  }
  auto maybe_parallel_conf = TRY(maybe_ctx.data()->GetParallelDescFromProducerView(lbn));
  if (maybe_parallel_conf.IsOk() == false) {
    return std::make_pair(PbMessage2TxtString(ParallelConf()),
                          PbMessage2TxtString(*maybe_parallel_conf.error()));
  }
  const auto& parallel_conf = maybe_parallel_conf.data()->parallel_conf();
  return std::make_pair(PbMessage2TxtString(parallel_conf), PbMessage2TxtString(ErrorUtil::Ok()));
}
