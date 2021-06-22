#include <bits/stdint-intn.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "oneflow/api/java/library.h"
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
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/session.h"
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
  return oneflow::InitEnv(convert_jstring_to_string(env, env_proto_str)).GetOrThrow();
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
