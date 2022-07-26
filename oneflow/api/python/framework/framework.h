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
#ifndef ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_H_
#define ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_H_

#include <string>
#include <google/protobuf/text_format.h>
#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/job/foreign_watcher.h"
#include "oneflow/core/job/job_instance.h"
#include "oneflow/core/job/oneflow.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/framework/config_def.h"
#include "oneflow/core/framework/load_library.h"
#include "oneflow/core/serving/saved_model.pb.h"

namespace oneflow {

inline Maybe<void> RegisterGlobalForeignCallback(const std::shared_ptr<ForeignCallback>& callback) {
  CHECK_ISNULL_OR_RETURN(Singleton<std::shared_ptr<ForeignCallback>>::Get())
      << "foreign callback registered";
  // Singleton<T>::SetAllocated is preferred since Singleton<T>::New will output logs but
  // glog is not constructed yet.
  Singleton<std::shared_ptr<ForeignCallback>>::SetAllocated(
      new std::shared_ptr<ForeignCallback>(callback));
  return Maybe<void>::Ok();
}

inline Maybe<void> DestroyGlobalForeignCallback() {
  if (Singleton<std::shared_ptr<ForeignCallback>>::Get()) {
    Singleton<std::shared_ptr<ForeignCallback>>::Delete();
  }
  return Maybe<void>::Ok();
}

inline Maybe<void> RegisterGlobalWatcher(const std::shared_ptr<ForeignWatcher>& watcher) {
  CHECK_ISNULL_OR_RETURN(Singleton<std::shared_ptr<ForeignWatcher>>::Get())
      << "foreign watcher registered";
  // Singleton<T>::SetAllocated is preferred since Singleton<T>::New will output logs but
  // glog is not constructed yet.
  Singleton<std::shared_ptr<ForeignWatcher>>::SetAllocated(
      new std::shared_ptr<ForeignWatcher>(watcher));
  return Maybe<void>::Ok();
}

inline Maybe<void> LaunchJob(const std::shared_ptr<oneflow::JobInstance>& cb) {
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());
  CHECK_NOTNULL_OR_RETURN(Singleton<Oneflow>::Get());
  const auto& job_name = cb->job_name();
  auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<JobInstance>>>::Get();
  int64_t job_id = Singleton<JobName2JobId>::Get()->at(job_name);
  if (IsPullJob(job_name, *Singleton<InterUserJobInfo>::Get())) {
    buffer_mgr->Get(GetForeignOutputBufferName(job_name))->Push(cb);
  }
  if (IsPushJob(job_name, *Singleton<InterUserJobInfo>::Get())) {
    buffer_mgr->Get(GetForeignInputBufferName(job_name))->Push(cb);
  }
  buffer_mgr->Get(GetCallbackNotifierBufferName(job_name))->Push(cb);
  Singleton<BufferMgr<int64_t>>::Get()->Get(kBufferNameGlobalWaitJobId)->Push(job_id);
  return Maybe<void>::Ok();
}

inline Maybe<std::string> GetSerializedStructureGraph() {
  const auto* job_ctx_mgr = Singleton<LazyJobBuildAndInferCtxMgr>::Get();
  CHECK_NOTNULL_OR_RETURN(job_ctx_mgr);
  return job_ctx_mgr->structure_graph();
}

inline Maybe<std::string> GetSerializedInterUserJobInfo() {
  CHECK_OR_RETURN(GlobalProcessCtx::IsThisProcessMaster());
  CHECK_NOTNULL_OR_RETURN(Singleton<Oneflow>::Get());
  CHECK_NOTNULL_OR_RETURN(Singleton<InterUserJobInfo>::Get());
  return Singleton<InterUserJobInfo>::Get()->SerializeAsString();
}

inline Maybe<const JobSet&> GetJobSet() {
  auto* job_ctx_mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  CHECK_NOTNULL_OR_RETURN(job_ctx_mgr);
  return job_ctx_mgr->job_set();
}

inline Maybe<std::string> GetSerializedJobSet() { return JUST(GetJobSet()).SerializeAsString(); }

inline Maybe<std::string> GetSerializedCurrentJob() {
  auto* job_ctx_mgr = Singleton<LazyJobBuildAndInferCtxMgr>::Get();
  CHECK_NOTNULL_OR_RETURN(job_ctx_mgr);
  auto* job_ctx =
      JUST(job_ctx_mgr->FindJobBuildAndInferCtx(*JUST(job_ctx_mgr->GetCurrentJobName())));
  CHECK_NOTNULL_OR_RETURN(job_ctx);
  return job_ctx->job().SerializeAsString();
}

inline Maybe<std::string> GetFunctionConfigDef() {
  std::string ret;
  google::protobuf::TextFormat::PrintToString(GlobalFunctionConfigDef(), &ret);
  return ret;
}

inline Maybe<std::string> GetScopeConfigDef() {
  std::string ret;
  google::protobuf::TextFormat::PrintToString(GlobalScopeConfigDef(), &ret);
  return ret;
}

inline Maybe<std::string> GetSerializedMachineId2DeviceIdListOFRecord(
    const std::string& parallel_conf_str) {
  ParallelConf parallel_conf;
  CHECK_OR_RETURN(TxtString2PbMessage(parallel_conf_str, &parallel_conf))
      << "parallel conf parse failed";
  return PbMessage2TxtString(*JUST(ParseMachineAndDeviceIdList(parallel_conf)));
}

inline Maybe<std::string> LoadSavedModel(const std::string& saved_model_meta_file,
                                         bool is_prototxt_file) {
  SavedModel saved_model_proto;
  if (is_prototxt_file) {
    CHECK_OR_RETURN(TryParseProtoFromTextFile(saved_model_meta_file, &saved_model_proto));
  } else {
    CHECK_OR_RETURN(TryParseProtoFromPbFile(saved_model_meta_file, &saved_model_proto));
  }
  return saved_model_proto.SerializeAsString();
}

inline Maybe<void> LoadLibraryNow(const std::string& lib_path) { return LoadLibrary(lib_path); }

}  // namespace oneflow

#endif  // ONEFLOW_API_PYTHON_FRAMEWORK_FRAMEWORK_H_
