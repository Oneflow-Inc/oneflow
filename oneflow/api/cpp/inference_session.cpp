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
#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/api/python/env/env_api.h"
#include "oneflow/api/python/session/session_api.h"
#include "oneflow/api/python/job_build/job_build_and_infer_api.h"
#include "oneflow/api/cpp/job_instance.h"
#include "oneflow/api/cpp/env/env_util.h"
#include "oneflow/api/cpp/session/session_util.h"
#include "oneflow/api/cpp/inference_session.h"

namespace oneflow {

inline int FindModelLatestVersion(std::string saved_model_dir) {
    std::vector<int> versions;
    std::vector<std::string> subfiles = LocalFS()->ListDir(saved_model_dir);
    for(auto& f : subfiles) {
      try {
        versions.push_back(std::stoi(f));
      } catch (...) {}
    }
    return std::max(std::begin(versions), std::end(versions));
}

void SignatureProtoToCfg(const JobSignatureDef& signature_proto, 
                         cfg::JobSignatureDef& mut_signature_cfg) {}
  for (input_name, input_def) in signature_proto.inputs.items():
      input_def_cfg = job_conf_proto_cfg.JobInputDef()
      input_def_cfg.mutable_lbi().set_op_name(input_def.lbi.op_name)
      input_def_cfg.mutable_lbi().set_blob_name(input_def.lbi.blob_name)
      _inferface_blob_conf_proto_to_cfg(
          input_def.blob_conf, input_def_cfg.mutable_blob_conf()
      )
      mut_signature_cfg.mutable_inputs()[input_name].CopyFrom(input_def_cfg)
  for (output_name, output_def) in signature_proto.outputs.items():
      output_def_cfg = job_conf_proto_cfg.JobOutputDef()
      output_def_cfg.mutable_lbi().set_op_name(output_def.lbi.op_name)
      output_def_cfg.mutable_lbi().set_blob_name(output_def.lbi.blob_name)
        mut_signature_cfg.mutable_outputs()[output_name].CopyFrom(output_def_cfg)
}

bool NeedCheckDeviceTag(OpConf& op_conf) {
  if (op_conf.has_return_conf()) {
    return false;
  }
  return op_conf.has_device_tag();
}

InferenceSession::InferenceSession(const SessionOption& option)
  : option_(option), is_mirrored_(option.is_mirrored_view)
    checkpoint_path_(""), cur_job_name_("") {
  InitEventLoop();
  Init();
}

InferenceSession::~InferenceSession() {
  if(this->status_ != SessionStatus::CLOSED) {
    Close();
  }
}

Maybe<void> InferenceSession::Init() {
  // env init
  if(!IsEnvInited()) {
    EnvProto env_proto;
    // TODO: set env - machine id, ctrl port, data port
    std::string env_proto_str = PbMessage2TxtString(env_proto);
    // FIXME: multi-client should be true or false?
    InitEnv(env_proto_str, false);

    // scope init
    InitScopeStack();
  }

  if (!IsEnvInited()) {
    LOG(ERROR) << "Env is not inited correctly";
  }

  // session init
  if(!IsSessionInited()) {
    this->MakeConfigProto();
    TryCompleteConfigProto(this->config_proto_);
    InitLazyGlobalSession(this->config_proto_);
  }

  if (!IsSessionInited()) {
    LOG(ERROR) << "Session is not inited correctly";
  }

  this->status_ = SessionStatus::OPEN;
}

void InferenceSession::Close() {
  this->event_loop_.run_until_complete(this->wait_for_all_jobs_finished())
  this->event_loop_.close()

  if(this->status_ == SessionStatus::RUNNING) {
    StopLazyGlobalSession();
    DestroyLazyGlobalSession();
  } else if(this->status_ == SessionStatus::OPEN) {
    DestroyLazyGlobalSession();
  }

  this->status_ = SessionStatus::CLOSED;
}

void InferenceSession::OpenCtx(std::string job_name, JobSignatureDef signature, int batch_size = 0) {
  this->CheckStatus(SessionStatus::OPEN);
  JobBuildAndInferCtx_Open(job_name);

  if (!signature.empty()) {
    this->SetJobSignature(job_name, signature);
  }

  if (batch_size != 0) {
    this->SetJobBatchSize(job_name, batch_size);
  }

  cfg::JobConfigProto& job_conf = this->GetJobConf(job_name);
  CurJobBuildAndInferCtx_SetJobConf(job_conf);

  std::string device_tag;
  std::vector<std::string> device_ids;
  if (this->config_proto_.resource().has_gpu_device_num() 
        && this->config_proto_.resource().gpu_device_num() > 0) {
    device_tag = "gpu";
    for(int i = 0; i < this->config_proto_.resource().machine_num(); i++) {
      device_ids.push_back(std::to_string(i) + ":0-" 
          + std::to_string(this->config_proto_.resource().gpu_device_num()-1));
    }
  } else if (this->config_proto_.resource().has_cpu_device_num()) {
    device_tag = "cpu";
    for(int i = 0; i < this->config_proto_.resource().machine_num(); i++) {
      device_ids.push_back(std::to_string(i) + ":0-" 
          + std::to_string(this->config_proto_.resource().cpu_device_num()-1));
    }
  } else {
    CHECK_OR_RETURN(false) << Error::Unimplemented();
  }

  std::shared_ptr<Scope> scope = MakeInitialScope(job_conf, device_tag, 
                                    device_ids, nullptr, this->is_mirrored_);

  ThreadLocalScopeStackPush(scope).GetOrThrow();
  this->cur_job_name_ = job_name;
}

void InferenceSession::CloseCtx() {
  this->cur_job_name_.clear();
  ThreadLocalScopeStackPop().GetOrThrow();
  JobBuildAndInferCtx_Close();
}

void InferenceSession::Compile(std::vector<OperatorConf>& op_list) {
  this->CheckStatus(SessionStatus::OPEN);

  std::shared_ptr<Scope> scope = GetCurrentScope().GetPtrOrThrow();
  for (auto& op_conf : op_list) {
      op_conf.set_scope_symbol_id(scope->symbol_id().GetOrThrow());
      if(!op_conf.has_device_tag()) {
        op_conf.set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
      }
      std::string op_conf_str = PbMessage2TxtString(op_conf);
      CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf_str);
    }
  }

  CurJobBuildAndInferCtx_Complete();
  CurJobBuildAndInferCtx_Rebuild();
}

void InferenceSession::Launch() {
  this->CheckStatus(SessionStatus::OPEN);
  StartLazyGlobalSession();
  std::string inter_user_job_info_str = GetSerializedInterUserJobInfo();
  this->inter_user_job_info_.ParseFromString(inter_user_job_info_str);
  this->RunLoadCheckpointJob();
  this->status_ = SessionStatus::RUNNING;
}

Maybe<void> InferenceSession::LoadModel(std::string saved_model_dir,
                                             ModelVersionPolicy model_version_policy,
                                             std::string saved_model_meta_file_basename,
                                             std::string graph_name = "",
                                             std::string signature_name = "") {

  CHECK_OR_RETURN(LocalFS()->IsDirectory(saved_model_dir))
    << Error::ValueError(saved_model_dir + std::string(" is not a valid directory"));

  int model_version;
  if(model_version_policy.latest) {
    model_version = FindModelLatestVersion(saved_model_dir);
  } else {
    model_version = model_version_policy.version;
  }
  
  std::string saved_model_path = JoinPath(saved_model_dir, std::to_string(model_version));

  CHECK_OR_RETURN(LocalFS()->IsDirectory(saved_model_path))
      << Error::ValueError(std::string("version of saved model in dir do not exist"));

  std::vector<std::string> subfiles = LocalFS()->ListDir(saved_model_path);
  std::string saved_model_meta_pb_filename = saved_model_meta_file_basename + ".pb";
  std::string saved_model_meta_prototxt_filename = saved_model_meta_file_basename + ".prototxt";
  
  std::shared_ptr<cfg::SavedModel> saved_model_proto;
  if (std:::find(std::begin(subfiles), std::end(subfiles), 
      saved_model_meta_pb_filename) != std::end(subfiles)) {
      saved_model_proto = LoadSavedModel(saved_model_meta_pb_filename, false);
  } else if (std:::find(std::begin(subfiles), std::end(subfiles), 
      saved_model_meta_prototxt_filename) != std::end(subfiles)) {
      saved_model_proto = LoadSavedModel(saved_model_meta_prototxt_filename, true);
  } else {
      CHECK_OR_RETURN(false) << Error::ValueError("saved model meta file" + 
        saved_model_meta_file_basename + " do not exist in " + saved_model_path);
  }

  // set checkpoint
  this->SetCheckpointPath(JoinPath(saved_model_path, saved_model_proto.checkpoint_dir));

  // get signature
  JobSignatureDef signature;
  if (graph_name.empty()) {
    graph_name = saved_model_proto->default_graph_name();
  } else {
    CHECK_OR_RETURN(saved_model_proto->graphs().count(graph_name))
      << ValueError("graph " + graph_name + "do not exist");
  }

  GraphDef graph_def = saved_model_proto->mut_graphs()->at(graph_name);
  if (signature_name.empty() && graph_def.has_default_signature_name()) {
      signature_name = graph_def.default_signature_name();
  }

  if (!signature_name.empty()) {
    CHECK_OR_RETURN(graph_def.signatures().count(signature_name))
      << ValueError("signature " + signature_name + "do not exist");
    signature = graph_def.mut_signatures()->at(signature_name);
  }

  // compile job
  this->OpenCtx(graph_name, signature);
  this->Compile(graph_def.op_list);
  this->CloseCtx();

  return Maybe<void>::Ok();
}

void InferenceSession::Run(std::string job_name) {
  this->CheckStatus(SessionStatus::RUNNING);
  return this->event_loop_.run_until_complete(this->AsyncRun(job_name, **kwargs));
}

void InferenceSession::AsyncRun(std::string job_name) {
  this->CheckStatus(SessionStatus::RUNNING);
  this->RunPushJobs(**kwargs);
  auto job_inst = MakeUserJobInstance(job_name);
  this->RunJob(job_inst);
  std::vector<std::future> output_futures;
  auto future_map = this->RunPullJobs(job_name);
  for(auto& pair : future_map) {
    output_futures.push_back(pair.second);
  }
  return await asyncio.gather(*output_futures);
}

void InferenceSession::WaitForAllJobsFinished() {
  await asyncio.gather(*this->job_futures_);
  this->job_futures_.clear();
}

void InferenceSession::SetCheckpointPath(std::string checkpoint_path) {
  this->CheckStatus(SessionStatus::OPEN);
  this->checkpoint_path_ = checkpoint_path;
}

void InferenceSession::SetJobSignature(std::string job_name, JobSignatureDef signature) {
  std::shared_ptr<cfg::JobConfigProto> job_conf = this->GetJobConf(job_name);
  SignatureProtoToCfg(signature, *job_conf->mutable_signature());
}

void InferenceSession::SetJobBatchSize(std::string job_name, int batch_size) {
  std::vector<SessionStatus> status = { SessionStatus::OPEN };
  this->CheckStatus(status);
  std::shared_ptr<cfg::JobConfigProto> job_conf = this->GetJobConf(job_name);
  for (auto& pair : job_conf->mutable_signature()->mutable_inputs()) {
    ShapeProto* mut_shape = pair.second.mutable_blob_conf()->mutable_shape();
    mut_shape->set_dim(batch_size);
  }
}

void InferenceSession::PrintJobSet() {
  this->CheckStatus(SessionStatus::OPEN, SessionStatus::RUNNING);
  const JobSet& job_set = JUST(GetJobSet());
  for (const auto& job : job_set.job()) {
      LOG(INFO) << "job_name:", job.job_conf().job_name();
      for (const auto& op_conf : job.net().op()) {
          LOG(INFO) << "\top_name:" << op_conf.name();
      }
  }
}

std::vector<std::string> InferenceSession::ListJobs() {
  this->CheckStatus(SessionStatus::RUNNING);
  std::vector<std::string> job_names;
  for(auto& pair : this->job_name2job_conf_) {
    job_names.push_back(pair.first);
  }
  return job_names;
}

std::vector<std::string> InferenceSession::ListInputs() {
  this->CheckStatus(SessionStatus::RUNNING);
  std::vector<std::string> input_names;
  for (auto& pair : this->inter_user_job_info_.input_or_var_op_name2push_job_name) {
    input_names.push_back(pair.first);
  }
  return input_names;
}

std::vector<std::string> InferenceSession::ListOutputs() {
  this->CheckStatus(SessionStatus::RUNNING);
  std::vector<std::string> output_names;
  for (auto& pair : this->inter_user_job_info_.output_or_var_op_name2pull_job_name) {
    output_names.push_back(pair.first);
  }
  return output_names;
}

Maybe<void> InferenceSession::InitEventLoop() {
  this->event_loop_ = asyncio.get_event_loop()
  if this->event_loop_.is_closed():
      asyncio.set_event_loop(asyncio.new_event_loop())
      this->event_loop_ = asyncio.get_event_loop()
}

void InferenceSession::CheckStatus(SessionStatus status) {
  bool check_success = (status == this->status_);
  CHECK_OR_RETURN(check_success) 
    << Error::ValueError("The calling to " + caller_func_name + " is not allowed in current status");
}

void InferenceSession::CheckStatus(const std::vector<SessionStatus>& status) {
  bool check_success = false;
  for(auto stat : status) {
    if(stat == this->status_) {
      check_success = true;
      break;
    }
  }
  CHECK_OR_RETURN(check_success) 
    << Error::ValueError("The calling to " + caller_func_name + " is not allowed in current status");
}

void InferenceSession::MakeConfigProto() {
  this->config_proto_ = GetDefaultConfigProto();

  if(this->option_.device_tag == "gpu") {
    this->config_proto_.mut_resource()->set_gpu_device_num(this->option_.device_num);
  } else if(this->option_.device_tag == "cpu") {
    this->config_proto_.mut_resource()->set_cpu_device_num(this->option_.device_num);
    this->config_proto_.mut_resource()->set_gpu_device_num(0);
  } else {
    CHECK_OR_RETURN(false) << Error::Unimplemented()
        << "not supported device tag " << this->option_.device_tag;
  }

  this->config_proto_.mut_resource()->set_enable_legacy_model_io(true);
}

std::shared_ptr<cfg::JobConfigProto>& InferenceSession::GetJobConf(std::string job_name) {
  if (std::find(std::begin(this->job_name2job_conf_), 
                std::end(this->job_name2job_conf_), 
                job_name) != std::end(this->job_name2job_conf_) {
    return this->job_name2job_conf_[job_name];
  } else {
    std::shared_ptr<cfg::JobConfigProto> job_conf = std::make_shared<cfg::JobConfigProto>();
    job_conf->set_job_name(job_name);
    job_conf->mutable_predict_conf();
    this->job_name2job_conf_[job_name] = job_conf;
    return this->job_name2job_conf_[job_name];
  }
}

void InferenceSession::RunJob(std::shared_ptr<JobInstance> job_inst) {
  std::promise<void> job_promise;
  auto job_finish_cb = [job_promise](){ job_promise.set_value(); };
  job_inst->AddPostFinishCallback(job_finish_cb);
  LaunchJob(job_inst);
  this->job_futures_.append(job_promise.get_future());
}

void InferenceSession::RunPushJobs() {
  for (auto& pair : this->inter_user_job_info_.input_or_var_op_name2push_job_name) {
      std::string input_name = pair.first;
      std::string push_job_name = pair.second;
      if input_name not in kwargs:
          raise ValueError('input "{}" is absent'.format(input_name))

      input_numpy = kwargs[input_name]
      if not isinstance(input_numpy, np.ndarray):
          raise ValueError('input "{}" requires numpy.ndarray'.format(input_name))

      push_fn = input_blob_util._MakePushNdarrayCallback(input_numpy)
      auto push_job_inst = MakePushJobInstance(push_job_name, input_name, push_fn);
      this->RunJob(push_job_inst);
  }
}

std::map<std::string, std::future> InferenceSession::RunPullJobs() {
  std::map<std::string, std::future> output_futures;
  for (auto& pair : this->inter_user_job_info_.output_or_var_op_name2pull_job_name) {
      std::string output_name = pair.first;
      std::string pull_job_name = pair.second;
      std::promise<void> pull_job_promise;
      auto pull_fn = this->MakePullJobCb(output_name, user_job_name, pull_job_promise);
      auto pull_job_inst = this->MakePullJobInstance(pull_job_name, output_name, pull_fn);
      this->RunJob(pull_job_inst);
      output_futures[output_name] = pull_job_promise.get_future();
  }
  return output_futures;
}

std::function<void(OfBlob*)> InferenceSession::MakePullJobCb(std::string output_name,
    std::string user_job_name, std::promise<void> pull_job_promise) {
  std::string output_lbn = JobBuildAndInferCtx_GetOpBlobLbn(user_job_name, output_name, "out");

  // std::string split_axis_str = JobBuildAndInferCtx_GetSplitAxisFromProducerView(user_job_name, output_lbn);
  // cfg::OptInt64 split_axis;
  // CHECK_OR_RETURN(TxtString2PbMessage(split_axis_str, &split_axis)) << "OptInt64 parse failed";
  // if (split_axis.has_value()) {
  //   int64_t split_axis_val = split_axis.value();
  // }

  auto pull_fn = [](OfBlob* ofblob) {
      ndarray = ofblob.CopyToNdarray();
      this->event_loop_.call_soon_threadsafe(future.set_result, ndarray)
  };

  return pull_fn;
}

void InferenceSession::RunLoadCheckpointJob() {
  CHECK_OR_RETURN(!this->checkpoint_path_.empty()) 
    << Error::ValueError(std::string("checkpoint path not set")); 

  auto copy_model_load_path = [](OfBlob* ofblob) {
      ofblob.CopyFromNdarray(
          np.frombuffer(this->checkpoint_path_.encode("ascii"), dtype=np.int8)
      )
  };

  auto load_checkpoint_job_inst = MakeUserInstance(
    this->inter_user_job_info_.global_model_load_job_name);
  load_checkpoint_job_inst->SetPushCb(copy_model_load_path);
  this->RunJob(load_checkpoint_job_inst);
}

}  // namespace oneflow
