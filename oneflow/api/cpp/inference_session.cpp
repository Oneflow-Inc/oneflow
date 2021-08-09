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
#include "oneflow/core/register/ofblob.h"
#include "oneflow/api/python/session/session_api.h"
#include "oneflow/api/python/job_build/job_build_and_infer_api.h"
#include "oneflow/api/cpp/job_instance.h"
#include "oneflow/api/cpp/inference_session.h"

namespace oneflow {

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
    // TODO
    flow.env.init();
  }

  // session init
  if(!IsSessionInited()) {
    MakeConfigProto();
    TryCompleteConfigProto(this->config_proto_);
    InitLazyGlobalSession(this->config_proto_);
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
  CheckStatus(SessionStatus::OPEN);
  JobBuildAndInferCtx_Open(job_name);

  if (!signature.empty()) {
    this->set_job_signature(job_name, signature);
  }

  if (batch_size != 0) {
     this->set_job_batch_size(job_name, batch_size);
  }

  JobConfigProto job_conf = this->_get_job_conf(job_name);
  CurJobBuildAndInferCtx_SetJobConf(job_conf);

  tag_and_dev_ids = placement_util.GetDefaultMachineDeviceIds(
      this->config_proto_.resource
  );
  scope = scope_util.MakeInitialScope(
      job_conf, *tag_and_dev_ids, None, this->is_mirrored_
  );

  with runtime_mode.ModeScope(runtime_mode.GLOBAL_MODE):
      with scope_util.ScopeContext(scope):
  
  this->cur_job_name_ = job_name;
}

void InferenceSession::CloseCtx() {
  this->cur_job_name_.clear();
  JobBuildAndInferCtx_Close();
}

void InferenceSession::Compile(std::vector<OperatorConf> op_list) {
  this->CheckStatus(SessionStatus::OPEN);

  scope = flow.current_scope()
  device_tag = scope.device_parallel_desc_symbol.device_tag
  for (auto& op_conf : op_list) {
      if _need_check_device_tag(op_conf) and op_conf.device_tag != device_tag:
          print(
              "WARNING: the device_tag of op {} is not equal to the device_tag of seesion's current scope"
              " ({} vs. {})"
              ", which may cause the op graph to be incompatible".format(
                  op_conf.name, op_conf.device_tag, device_tag
              )
          )

    if distribute_ctx.IsMirroredStrategyEnabled(){
      assert not hob.consistent_view_enabled(None)
      scope_symbol = oneflow.current_scope()
      op_conf.scope_symbol_id = scope_symbol.symbol_id
      if not op_conf.HasField("device_tag"):
          device_tag = scope_symbol.device_parallel_desc_symbol.device_tag
          op_conf.device_tag = device_tag
      op_attr = c_api_util.CurJobBuildAndInferCtx_AddAndInferMirroredOp(op_conf)
      if c_api_util.IsInterfaceOpConf(op_conf):
          sess = session_ctx.GetDefaultSession()
          sess.AddInfo4InterfaceOpName(op_conf.name, op_attr)
    } else {
      scope_symbol = oneflow.current_scope()
      op_conf.scope_symbol_id = scope_symbol.symbol_id
      if not op_conf.HasField("device_tag"):
          device_tag = scope_symbol.device_parallel_desc_symbol.device_tag
          op_conf.device_tag = device_tag
      op_attr = c_api_util.CurJobBuildAndInferCtx_AddAndInferConsistentOp(op_conf)
      if c_api_util.IsInterfaceOpConf(op_conf):
          sess = session_ctx.GetDefaultSession()
          sess.AddInfo4InterfaceOpName(op_conf.name, op_attr)
    }
  }

  CurJobBuildAndInferCtx_Complete();
  CurJobBuildAndInferCtx_Rebuild();
}

void InferenceSession::Launch() {
  this->CheckStatus(SessionStatus::OPEN);
  StartLazyGlobalSession();
  std::string inter_user_job_info_str = GetSerializedInterUserJobInfo();
  this->inter_user_job_info_ = InterUserJobInfo.parseFromString(inter_user_job_info_str);
  this->RunLoadCheckpointJob();
  this->status_ = SessionStatus::RUNNING;
}

Maybe<void> InferenceSession::LoadSavedModel(std::string saved_model_dir,
                    int model_version,
                    std::string saved_model_meta_file_basename,
                    std::string graph_name = "",
                    std::string signature_name = "") {

  CHECK_OR_RETURN(LocalFS()->IsDirectory(saved_model_dir))
    << Error::ValueError(saved_model_dir + std::string(" is not a valid directory"));

  if(model_version == ModelVersionPolicy::LATEST) {
    model_version = _find_model_latest_version(saved_model_dir);
  }
  
  std::string saved_model_path = JoinPath(saved_model_dir, std::to_string(model_version));

  CHECK_OR_RETURN(LocalFS()->IsDirectory(saved_model_path))
      << Error::ValueError(std::string("version of saved model in dir do not exist"));

  std::vector<std::string> subfiles = LocalFS()->ListDir(saved_model_path);
  std::string saved_model_meta_pb_filename = saved_model_meta_file_basename + ".pb";
  std::string saved_model_meta_prototxt_filename = saved_model_meta_file_basename + ".prototxt";
  
  SavedModel saved_model_proto = saved_model_pb.SavedModel();
  if (std:::find(std::begin(subfiles), std::end(subfiles), 
      saved_model_meta_pb_filename) != std::end(subfiles)) {
      std::string saved_model_meta_file_path = JoinPath(
          saved_model_path, saved_model_meta_pb_filename
      );
      with open(saved_model_meta_file_path, "rb") as f:
          saved_model_proto.ParseFromString(f.read())
  } else if (std:::find(std::begin(subfiles), std::end(subfiles), 
      saved_model_meta_prototxt_filename) != std::end(subfiles)) {
      std::string saved_model_meta_file_path = JoinPath(
          saved_model_path, saved_model_meta_prototxt_filename
      );
      with open(saved_model_meta_file_path, "rt") as f:
          text_format.Merge(f.read(), saved_model_proto)
  } else {
      raise ValueError(
          "saved model meta file {} do not exist in {}".format(
              saved_model_meta_file_basename, saved_model_path
          )
      )
  }

  // set checkpoint
  this->SetCheckpointPath(JoinPath(saved_model_path, saved_model_proto.checkpoint_dir));

  // get signature
  std::string signature = "";
  if (graph_name.empty()) {
      graph_name = saved_model_proto.default_graph_name
  } else {
      if graph_name not in saved_model_proto.graphs:
          raise ValueError("graph {} do not exist".format(graph_name))
  }

  GraphDef graph_def = saved_model_proto.graphs[graph_name];
  if (signature_name == "" && graph_def.HasField("default_signature_name")) {
      signature_name = graph_def.default_signature_name;
  }

  if (signature_name != "") {
      if (signature_name not in graph_def.signatures) {
          raise ValueError("signature {} do not exist".format(signature_name))
      } else {
          signature = graph_def.signatures[signature_name];
      }
  }

  // compile job
  this->open(graph_name, signature);
  this->compile(graph_def.op_list);
  this->close();

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
  JobConfigProto& job_conf = this->GetJobConf(job_name);
  signature_proto_to_cfg(signature, job_conf.mutable_signature());
}

void InferenceSession::SetJobBatchSize(std::string job_name, int batch_size) {
  std::vector<SessionStatus> status = { SessionStatus::OPEN };
  CheckStatus(status);
  // TODO
  JobConfigProto& job_conf = this->GetJobConf(job_name);
  for (auto& pair : job_conf.mutable_signature()->mutable_inputs()) {
    ShapeProto* mut_shape = pair.second.mutable_blob_conf()->mutable_shape();
    (*mut_shape.mutable_dim()) = batch_size;
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

InferenceSession::input_info(std::string input_name, std::string job_name = "") {
  return this->GetOpBlobInfo(job_name, input_name, "out");
}

InferenceSession::output_info(std::string input_name, std::string job_name = "") {
  return this->GetOpBlobInfo(job_name, output_name, "in");
}

Maybe<void> InferenceSession::InitEventLoop() {
  this->event_loop_ = asyncio.get_event_loop()
  if this->event_loop_.is_closed():
      asyncio.set_event_loop(asyncio.new_event_loop())
      this->event_loop_ = asyncio.get_event_loop()
}

bool InferenceSession::CheckStatus(SessionStatus status) {
  bool check_success = (status == this->status_);

  if(!check_success) {
    // TODO
    caller_func_name = inspect.stack()[1].function
    allowed_status = ",".join(status)
    raise ValueError(
        "The calling to {} is only allowed when status is {}, current status is {}".format(
            caller_func_name, allowed_status, this->status_
        )
    )
  }
}

bool InferenceSession::CheckStatus(const std::vector<SessionStatus>& status) {
  bool check_success = false;
  for(auto stat : status) {
    if(stat == this->status_) {
      check_success = true;
      break;
    }
  }

  if(!check_success) {
    // TODO
    caller_func_name = inspect.stack()[1].function
    allowed_status = ",".join(status)
    raise ValueError(
        "The calling to {} is only allowed when status is {}, current status is {}".format(
            caller_func_name, allowed_status, this->status_
        )
    )
  }
}

void InferenceSession::MakeConfigProto() {
  this->config_proto_ = this->GetDefaultConfigProto();

  if(this->option_.device_tag == "gpu") {
    (*this->config_proto_.mut_resource()->mut_gpu_device_num()) = this->option_.device_num;
  } else if(this->option_.device_tag == "cpu") {
    (*this->config_proto_.mut_resource()->mut_cpu_device_num()) = this->option_.device_num;
    (*this->config_proto_.mut_resource()->mut_gpu_device_num()) = 0;
  } else {
    CHECK_OR_RETURN(false) << Error::Unimplemented()
        << "not supported device tag " << this->option_.device_tag;
  }

  this->config_proto_.io_conf.enable_legacy_model_io = true;
}

JobConfigProto& InferenceSession::GetJobConf(std::string job_name) {
  // TODO
  if (std::find(std::begin(this->job_name2job_conf_), 
                std::end(this->job_name2job_conf_), 
                job_name) != std::end(this->job_name2job_conf_) {
    return this->job_name2job_conf_[job_name];
  } else {
    JobConfigProto job_conf;
    job_conf.set_job_name(job_name);
    job_conf.mutable_predict_conf();
    this->job_name2job_conf_[job_name] = job_conf;
    return this->job_name2job_conf_[job_name];
  }
}

InferenceSession::GetOpBlobInfo(std::string job_name, 
                                std::string op_name, 
                                std::string blob_name) {
  std::vector<SessionStatus> status = {SessionStatus::OPEN, SessionStatus::RUNNING};
  this->CheckStatus(status);
  if (std::find(std::begin(this->job_name2job_conf_), 
                std::end(this->job_name2job_conf_), 
                job_name) != std::end(this->job_name2job_conf_) {
    return this->job_name2job_conf_[job_name];
  } 

  if (std::find(std::begin(this->inferface_name2info_), 
                std::end(this->inferface_name2info_), 
                op_name) != std::end(this->inferface_name2info_) {
    return this->inferface_name2info_[op_name];
  }

  if(job_name.empty()) job_name = this->cur_job_name_;
  CHECK_OR_RETURN(!job_name.empty()) << Error::ValueError(std::string("please specify job_name")); 

  std::string lbn = JobBuildAndInferCtx_GetOpBlobLbn(job_name, op_name, blob_name);
  std::vector<int> shape = c_api_util.JobBuildAndInferCtx_GetStaticShape(job_name, lbn);
  dtype = c_api_util.JobBuildAndInferCtx_GetDataType(job_name, lbn);
  dtype = dtype_util.convert_proto_dtype_to_oneflow_dtype(dtype);
  auto info = std::pair<std::vector<int>, dtype>(shape, dtype);
  this->inferface_name2info_[op_name] = info;
  return info;
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

  std::string split_axis_str = JobBuildAndInferCtx_GetSplitAxisFromProducerView(user_job_name, output_lbn);
  split_axis = text_format.Parse(split_axis_str, dtype_util.OptInt64());
  if split_axis.HasField("value") {
    split_axis_val = split_axis.value;
  }

  def pull_fn(ofblob):
      ndarray = ofblob.CopyToNdarray()
      this->event_loop_.call_soon_threadsafe(future.set_result, ndarray)

  return pull_fn;
}

void InferenceSession::RunLoadCheckpointJob() {
  CHECK_OR_RETURN(!this->checkpoint_path_.empty()) << Error::ValueError(std::string("checkpoint path not set")); 

  def copy_model_load_path(ofblob):
      ofblob.CopyFromNdarray(
          np.frombuffer(this->checkpoint_path_.encode("ascii"), dtype=np.int8)
      )

  auto load_checkpoint_job_inst = MakeUserInstance(
    this->inter_user_job_info_.global_model_load_job_name);
  load_checkpoint_job_inst->SetPushCb(copy_model_load_path);
  this->RunJob(load_checkpoint_job_inst);
}

int InferenceSession::find_model_latest_version(std::string saved_model_dir) {


}

bool InferenceSession::need_check_device_tag(OperatorConf op_conf) {

}
void InferenceSession::signature_proto_to_cfg(JobSignatureDef signature_proto, mut_signature_cfg) {


}

}  // namespace oneflow
