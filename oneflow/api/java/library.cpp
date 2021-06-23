#include <bits/stdint-intn.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "oneflow/api/java/library.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session_api.h"
#include "oneflow/core/common/cfg.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/api/java/util/string_util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/vm/init_symbol_instruction_type.h"
#include "oneflow/core/framework/symbol_id_cache.h"
#include "oneflow/core/vm/stream_type.h"

JNIEXPORT 
void JNICALL Java_org_oneflow_Library_initDefaultSession(JNIEnv* env, jobject obj) {
  int64_t session_id = oneflow::NewSessionId();
  oneflow::RegsiterSession(session_id);
}

JNIEXPORT
jboolean JNICALL Java_org_oneflow_Library_isEnvInited(JNIEnv* env, jobject obj) {
  return oneflow::IsEnvInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_initEnv(JNIEnv* env, jobject obj, jstring env_proto_str) {
  oneflow::InitEnv(convert_jstring_to_string(env, env_proto_str)).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_initScopeStack(JNIEnv* env, jobject obj, jstring jstr) {
  std::shared_ptr<oneflow::cfg::JobConfigProto> job_conf = std::make_shared<oneflow::cfg::JobConfigProto>();
  job_conf->mutable_predict_conf();
  job_conf->set_job_name("");

  std::shared_ptr<oneflow::Scope> scope;
  auto BuildInitialScope = [&scope, &job_conf](oneflow::InstructionsBuilder* builder) mutable -> void {
    int session_id = oneflow::GetDefaultSessionId().GetOrThrow();
    const std::vector<std::string> machine_device_ids({"0:0"});
    std::shared_ptr<oneflow::Scope> initialScope = builder->BuildInitialScope(session_id, job_conf, "cpu", machine_device_ids, nullptr, false).GetPtrOrThrow();
    scope = initialScope;
  };
  oneflow::LogicalRun(BuildInitialScope);
  oneflow::InitThreadLocalScopeStack(scope);
}

JNIEXPORT
jboolean JNICALL Java_org_oneflow_Library_isSessionInited(JNIEnv* env, jobject obj) {
  return oneflow::IsSessionInited().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_initSession(JNIEnv* env, jobject obj) {
  // default configuration
  std::shared_ptr<oneflow::ConfigProto> config_proto = std::make_shared<oneflow::ConfigProto>();
  config_proto->mutable_resource()->set_machine_num(1);
  config_proto->mutable_resource()->set_gpu_device_num(1);
  config_proto->set_session_id(oneflow::GetDefaultSessionId().GetOrThrow());
  config_proto->mutable_io_conf()->mutable_data_fs_conf()->mutable_localfs_conf();
  config_proto->mutable_io_conf()->mutable_snapshot_fs_conf()->mutable_localfs_conf();
  config_proto->mutable_resource()->set_gpu_device_num(1);
  config_proto->mutable_io_conf()->set_enable_legacy_model_io(true);

  oneflow::InitLazyGlobalSession(config_proto->DebugString());
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_openJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring job_name) {
  oneflow::JobBuildAndInferCtx_Open(convert_jstring_to_string(env, job_name)).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_setJobConfForCurJobBuildAndInferCtx(JNIEnv* env, jobject obj, jstring job_conf_proto) {
  oneflow::CurJobBuildAndInferCtx_SetJobConf(convert_jstring_to_string(env, job_conf_proto)).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_setScopeForCurJob(JNIEnv* env, jobject obj) {
  std::shared_ptr<oneflow::cfg::JobConfigProto> job_conf = std::make_shared<oneflow::cfg::JobConfigProto>();
  job_conf->mutable_predict_conf();
  job_conf->set_job_name("mlp_inference");

  std::shared_ptr<oneflow::Scope> scope;
  auto BuildInitialScope = [&scope, &job_conf](oneflow::InstructionsBuilder* builder) mutable -> void {
    int session_id = oneflow::GetDefaultSessionId().GetOrThrow();
    const std::vector<std::string> machine_device_ids({"0:0"});
    std::shared_ptr<oneflow::Scope> initialScope = builder->BuildInitialScope(session_id, job_conf, "gpu", machine_device_ids, nullptr, false).GetPtrOrThrow();
    scope = initialScope;
  };
  oneflow::LogicalRun(BuildInitialScope);
  oneflow::ThreadLocalScopeStackPush(scope).GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_curJobAddOp(JNIEnv* env, jobject obj, jstring op_conf_proto) {
  std::string op_conf_proto_str = convert_jstring_to_string(env, op_conf_proto);
  op_conf_proto_str = subreplace(op_conf_proto_str, "user_input", "input");
  op_conf_proto_str = subreplace(op_conf_proto_str, "user_output", "output");

  oneflow::OperatorConf op_conf;
  oneflow::TxtString2PbMessage(op_conf_proto_str, &op_conf);

  auto scope = oneflow::GetCurrentScope().GetPtrOrThrow();
  op_conf.set_scope_symbol_id(scope->symbol_id().GetOrThrow());
  op_conf.set_device_tag(scope->device_parallel_desc_symbol()->device_tag());

  std::cout << op_conf.DebugString() << std::endl;

  oneflow::CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf.DebugString());
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_completeCurJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  oneflow::CurJobBuildAndInferCtx_Complete().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_rebuildCurJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  oneflow::CurJobBuildAndInferCtx_Rebuild().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_unsetScopeForCurJob(JNIEnv* env, jobject obj) {
  oneflow::ThreadLocalScopeStackPop().GetOrThrow();
}

JNIEXPORT
void JNICALL Java_org_oneflow_Library_closeJobBuildAndInferCtx(JNIEnv* env, jobject obj) {
  oneflow::JobBuildAndInferCtx_Close().GetOrThrow();
}
