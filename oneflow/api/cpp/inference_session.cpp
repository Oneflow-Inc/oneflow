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
#include <map>
#include <algorithm>

#include "oneflow/core/persistence/posix/posix_file_system.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/register/ofblob.h"
#include "oneflow/core/serving/saved_model.cfg.h"
#include "oneflow/core/serving/saved_model.pb.h"

#include "oneflow/api/python/env/env.h"
#include "oneflow/api/python/session/session.h"
#include "oneflow/api/python/framework/framework.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"

#include "oneflow/api/cpp/env/env_util.h"
#include "oneflow/api/cpp/session/session_util.h"
#include "oneflow/api/cpp/framework/framework_util.h"
#include "oneflow/api/cpp/job_build/job_build_and_infer_util.h"
#include "oneflow/api/cpp/tensor/tensor.h"
#include "oneflow/api/cpp/job_instance.h"
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
    return *(std::max(std::begin(versions), std::end(versions)));
}

inline bool NeedCheckDeviceTag(OperatorConf& op_conf) {
  if (op_conf.has_return_conf()) {
    return false;
  }
  return op_conf.has_device_tag();
}

InferenceSession::InferenceSession(const SessionOption& option)
  : option_(option), is_mirrored_(option.is_mirrored_view),
    checkpoint_path_(""), cur_job_name_("") {
  Init().GetOrThrow();
}

InferenceSession::~InferenceSession() {
  if(this->status_ != SessionStatus::CLOSED) {
    Close().GetOrThrow();
  }
}

Maybe<void> InferenceSession::Init() {
  // env init
  if(!IsEnvInited().GetOrThrow()) {
    // TODO: set env - machine id, ctrl port, data port
    EnvProto env_proto;
    // FIXME: multi-client should be true or false?
    JUST(InitEnv(env_proto, false));

    // scope init
    JUST(InitScopeStack());
  }

  if (!IsEnvInited().GetOrThrow()) {
    LOG(ERROR) << "Env is not inited correctly";
  }

  // session init
  if(!IsSessionInited().GetOrThrow()) {
    JUST(this->MakeConfigProto());
    JUST(InitLazyGlobalSession(this->config_proto_));
  }

  if (!IsSessionInited().GetOrThrow()) {
    LOG(ERROR) << "Session is not inited correctly";
  }

  this->status_ = SessionStatus::OPEN;
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::Close() {
  WaitForAllJobsFinished();

  if(this->status_ == SessionStatus::RUNNING) {
    JUST(StopLazyGlobalSession());
    JUST(DestroyLazyGlobalSession());
  } else if(this->status_ == SessionStatus::OPEN) {
    JUST(DestroyLazyGlobalSession());
  }

  this->status_ = SessionStatus::CLOSED;
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::OpenCtx(std::string job_name, JobSignatureDef* signature, int batch_size = 0) {
  JUST(this->CheckStatus(SessionStatus::OPEN));
  JUST(JobBuildAndInferCtx_Open(job_name));

  if (!signature) {
    this->SetJobSignature(job_name, signature);
  }

  if (batch_size != 0) {
    this->SetJobBatchSize(job_name, batch_size);
  }

  std::shared_ptr<JobConfigProto> job_conf = this->GetJobConf(job_name);
  JUST(CurJobBuildAndInferCtx_SetJobConf(*job_conf));

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

  std::shared_ptr<cfg::JobConfigProto> job_cfg_conf = std::make_shared<cfg::JobConfigProto>(*job_conf);
  std::shared_ptr<Scope> scope = MakeInitialScope(job_cfg_conf, device_tag, 
                                    device_ids, nullptr, this->is_mirrored_);

  JUST(ThreadLocalScopeStackPush(scope));
  this->cur_job_name_ = job_name;
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::CloseCtx() {
  this->cur_job_name_.clear();
  JUST(ThreadLocalScopeStackPop());
  JUST(JobBuildAndInferCtx_Close());
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::Compile(GraphDef& graph_def) {
  JUST(this->CheckStatus(SessionStatus::OPEN));

  std::shared_ptr<Scope> scope = GetCurrentScope().GetPtrOrThrow();
  for (auto& op_conf : *(graph_def.mutable_op_list())) {
    op_conf.set_scope_symbol_id(scope->symbol_id().GetOrThrow());
    if(!op_conf.has_device_tag()) {
      op_conf.set_device_tag(scope->device_parallel_desc_symbol()->device_tag());
    }
    JUST(CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf));
  }

  JUST(CurJobBuildAndInferCtx_Complete());
  JUST(CurJobBuildAndInferCtx_Rebuild());
  return Maybe<void>::Ok();
}

void InferenceSession::Launch() {
  this->CheckStatus(SessionStatus::OPEN).GetOrThrow();
  StartLazyGlobalSession().GetOrThrow();
  this->inter_user_job_info_ = GetInterUserJobInfo().GetOrThrow();
  this->RunLoadCheckpointJob().GetOrThrow();
  this->status_ = SessionStatus::RUNNING;
}

void InferenceSession::LoadModel(std::string saved_model_dir,
                                 ModelVersionPolicy model_version_policy,
                                 std::string saved_model_meta_file_basename,
                                 std::string graph_name,
                                 std::string signature_name) {
  return this->LoadModel_(saved_model_dir,
                          model_version_policy, 
                          saved_model_meta_file_basename,
                          graph_name,
                          signature_name).GetOrThrow();
}

Maybe<void> InferenceSession::LoadModel_(std::string saved_model_dir,
                                         ModelVersionPolicy model_version_policy,
                                         std::string saved_model_meta_file_basename,
                                         std::string graph_name,
                                         std::string signature_name) {
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
  
  SavedModel saved_model_proto;
  if (std::find(std::begin(subfiles), std::end(subfiles), 
      saved_model_meta_pb_filename) != std::end(subfiles)) {
      CHECK_OR_RETURN(TryParseProtoFromPbFile(saved_model_meta_pb_filename, &saved_model_proto));
  } else if (std::find(std::begin(subfiles), std::end(subfiles), 
      saved_model_meta_prototxt_filename) != std::end(subfiles)) {
      CHECK_OR_RETURN(TryParseProtoFromTextFile(saved_model_meta_prototxt_filename, &saved_model_proto));
  } else {
      CHECK_OR_RETURN(false) << Error::ValueError("saved model meta file" + 
        saved_model_meta_file_basename + " do not exist in " + saved_model_path);
  }

  // set checkpoint
  this->SetCheckpointPath(JoinPath(saved_model_path, saved_model_proto.checkpoint_dir()));

  // get signature
  JobSignatureDef* signature_ptr;
  if (graph_name.empty()) {
    graph_name = saved_model_proto.default_graph_name();
  } else {
    CHECK_OR_RETURN(saved_model_proto.graphs().count(graph_name))
      << Error::ValueError("graph " + graph_name + "do not exist");
  }

  GraphDef graph_def = saved_model_proto.mutable_graphs()->at(graph_name);
  if (signature_name.empty() && graph_def.has_default_signature_name()) {
      signature_name = graph_def.default_signature_name();
  }

  if (!signature_name.empty()) {
    CHECK_OR_RETURN(graph_def.signatures().count(signature_name))
      << Error::ValueError("signature " + signature_name + "do not exist");
    signature_ptr = &(graph_def.mutable_signatures()->at(signature_name));
  }

  // compile job
  JUST(this->OpenCtx(graph_name, signature_ptr));
  JUST(this->Compile(graph_def));
  JUST(this->CloseCtx());

  return Maybe<void>::Ok();
}

std::map<std::string, std::shared_ptr<Tensor>> InferenceSession::Run(std::string job_name,
                                                    std::map<std::string, std::shared_ptr<Tensor>>& input_tensors) {
  this->CheckStatus(SessionStatus::RUNNING).GetOrThrow();
  this->RunPushJobs(input_tensors).GetOrThrow();
  auto job_inst = MakeUserJobInstance(job_name);
  this->RunJob(job_inst).GetOrThrow();
  std::map<std::string, std::shared_ptr<Tensor>> output_tensors;
  this->RunPullJobs(output_tensors).GetOrThrow();
  return output_tensors;
}

void InferenceSession::WaitForAllJobsFinished() {
  for(auto promise : this->job_promises_) {
    promise->get_future().wait();
  }
  this->job_promises_.clear();
}

void InferenceSession::SetCheckpointPath(std::string checkpoint_path) {
  this->CheckStatus(SessionStatus::OPEN).GetOrThrow();
  this->checkpoint_path_ = checkpoint_path;
}

void InferenceSession::SetJobSignature(std::string job_name, JobSignatureDef* signature) {
  std::shared_ptr<JobConfigProto> job_conf = this->GetJobConf(job_name);
  // TODO
  job_conf->set_allocated_signature(signature);
}

void InferenceSession::SetJobBatchSize(std::string job_name, int batch_size) {
  this->CheckStatus(SessionStatus::OPEN).GetOrThrow();
  std::shared_ptr<JobConfigProto> job_conf = this->GetJobConf(job_name);
  for (auto& pair : *(job_conf->mutable_signature()->mutable_inputs())) {
    pair.second.mutable_blob_conf()->mutable_shape()->mutable_dim()->at(0) = batch_size;
  }
}

Maybe<void> InferenceSession::CheckStatus(SessionStatus status) {
  bool check_success = (status == this->status_);
  // TODO
  std::string caller_func_name = "";
  CHECK_OR_RETURN(check_success) 
    << Error::ValueError("The calling to " + caller_func_name + " is not allowed in current status");
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::CheckStatus(const std::vector<SessionStatus>& status) {
  bool check_success = false;
  for(auto stat : status) {
    if(stat == this->status_) {
      check_success = true;
      break;
    }
  }
  // TODO
  std::string caller_func_name = "";
  CHECK_OR_RETURN(check_success) 
    << Error::ValueError("The calling to " + caller_func_name + " is not allowed in current status");
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::MakeConfigProto() {
  this->config_proto_ = GetDefaultConfigProto();

  if(this->option_.device_tag == "gpu") {
    this->config_proto_.mutable_resource()->set_gpu_device_num(this->option_.device_num);
  } else if(this->option_.device_tag == "cpu") {
    this->config_proto_.mutable_resource()->set_cpu_device_num(this->option_.device_num);
    this->config_proto_.mutable_resource()->set_gpu_device_num(0);
  } else {
    CHECK_OR_RETURN(false) << Error::Unimplemented()
        << "not supported device tag " << this->option_.device_tag;
  }

  this->config_proto_.mutable_resource()->set_enable_legacy_model_io(true);
  if (this->config_proto_.resource().machine_num() == 0) {
    this->config_proto_.mutable_resource()->set_machine_num(GetNodeSize().GetOrThrow());
  }
  return Maybe<void>::Ok();
}

std::shared_ptr<JobConfigProto> InferenceSession::GetJobConf(std::string job_name) {
  if (this->job_name2job_conf_.count(job_name)) {
    return this->job_name2job_conf_[job_name];
  } else {
    std::shared_ptr<JobConfigProto> job_conf = std::make_shared<JobConfigProto>();
    job_conf->set_job_name(job_name);
    job_conf->mutable_predict_conf();
    this->job_name2job_conf_[job_name] = job_conf;
    return this->job_name2job_conf_[job_name];
  }
}

Maybe<void> InferenceSession::RunJob(std::shared_ptr<CPPJobInstance> job_inst) {
  std::shared_ptr<std::promise<void>> job_promise = std::make_shared<std::promise<void>>();
  auto job_finish_cb = [&job_promise](JobInstance*){ job_promise->set_value(); };
  job_inst->AddPostFinishCallback(job_finish_cb);
  JUST(LaunchJob(job_inst));
  this->job_promises_.push_back(job_promise);
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::RunPushJobs(std::map<std::string, std::shared_ptr<Tensor>>& input_tensors) {
  for (auto& pair : this->inter_user_job_info_->input_or_var_op_name2push_job_name()) {
    std::string input_name = pair.first;
    std::string push_job_name = pair.second;
    if(!input_tensors.count(input_name)) {
      CHECK_OR_RETURN(false) 
        << Error::ValueError("input \"" + input_name + "\" is absent");
    }

    std::shared_ptr<Tensor> input_tensor = input_tensors[input_name];

    auto push_fn = [&input_tensor](OfBlob* ofblob){
      ofblob->CopyShapeFrom(input_tensor->shape().dim_vec().data(), input_tensor->num_axes());
      int64_t num_elems = input_tensor->num_elems();
      DataType dtype = input_tensor->dtype();

      // support type traits
      if (dtype == kFloat) {
        ofblob->AutoMemCopyFrom((const float*) input_tensor->data(), num_elems);
      }
      else if (dtype == kInt32) {
        ofblob->AutoMemCopyFrom((const int*) input_tensor->data(), num_elems);
      }
    };
    auto push_job_inst = MakePushJobInstance(push_job_name, input_name, push_fn);
    JUST(this->RunJob(push_job_inst));
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::RunPullJobs(std::map<std::string, std::shared_ptr<Tensor>>& output_tensors) {
  for (auto& pair : this->inter_user_job_info_->output_or_var_op_name2pull_job_name()) {
      std::string output_name = pair.first;
      std::string pull_job_name = pair.second;
      std::promise<std::shared_ptr<Tensor>> pull_job_promise;
      auto pull_fn = [&pull_job_promise](OfBlob* ofblob) {
        DataType dtype = ofblob->blob().data_type();
        Shape shape = ofblob->blob().static_shape();
        int64_t num_elems = shape.elem_cnt();

        std::shared_ptr<Tensor> tensor = std::make_shared<Tensor>(shape, dtype);

        // support type traits
        if (dtype == kFloat) {
          ofblob->AutoMemCopyTo((float*)tensor->mutable_data(), num_elems);
        }
        else if (dtype == kInt32) {
          ofblob->AutoMemCopyTo((int*)tensor->mutable_data(), num_elems);
        }
        pull_job_promise.set_value(tensor);
      };
      auto pull_job_inst = MakePullJobInstance(pull_job_name, output_name, pull_fn);
      JUST(this->RunJob(pull_job_inst));
      output_tensors[output_name] = pull_job_promise.get_future().get();
  }
  return Maybe<void>::Ok();
}

Maybe<void> InferenceSession::RunLoadCheckpointJob() {
  CHECK_OR_RETURN(!this->checkpoint_path_.empty()) 
    << Error::ValueError(std::string("checkpoint path not set")); 

  auto copy_model_load_path = [this](OfBlob* ofblob) -> void {
    int64_t* shape = new int64_t[1]{this->checkpoint_path_.size()};
    ofblob->CopyShapeFrom(shape, 1);
    ofblob->AutoMemCopyFrom(this->checkpoint_path_.data(), this->checkpoint_path_.size());
    delete[] shape;
  };

  std::string load_job_name = this->inter_user_job_info_->global_model_load_job_name();
  auto load_checkpoint_job_inst = MakePushJobInstance(load_job_name, "", copy_model_load_path);
  JUST(this->RunJob(load_checkpoint_job_inst));
  return Maybe<void>::Ok();
}

void InferenceSession::PrintJobSet() {
  this->CheckStatus({SessionStatus::OPEN, SessionStatus::RUNNING}).GetOrThrow();
  const JobSet& job_set = GetJobSet().GetOrThrow();
  for (const auto& job : job_set.job()) {
      LOG(INFO) << "job_name:", job.job_conf().job_name();
      for (const auto& op_conf : job.net().op()) {
          LOG(INFO) << "\top_name:" << op_conf.name();
      }
  }
}

std::vector<std::string> InferenceSession::ListJobs() {
  this->CheckStatus(SessionStatus::RUNNING).GetOrThrow();
  std::vector<std::string> job_names;
  for(auto& pair : this->job_name2job_conf_) {
    job_names.push_back(pair.first);
  }
  return job_names;
}

std::vector<std::string> InferenceSession::ListInputs() {
  this->CheckStatus(SessionStatus::RUNNING).GetOrThrow();
  std::vector<std::string> input_names;
  for (auto& pair : this->inter_user_job_info_->input_or_var_op_name2push_job_name()) {
    input_names.push_back(pair.first);
  }
  return input_names;
}

std::vector<std::string> InferenceSession::ListOutputs() {
  this->CheckStatus(SessionStatus::RUNNING).GetOrThrow();
  std::vector<std::string> output_names;
  for (auto& pair : this->inter_user_job_info_->output_or_var_op_name2pull_job_name()) {
    output_names.push_back(pair.first);
  }
  return output_names;
}

}  // namespace oneflow
