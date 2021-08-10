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
#ifndef ONEFLOW_API_CPP_INFERENCE_SESSION_H_
#define ONEFLOW_API_CPP_INFERENCE_SESSION_H_

#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/api/python/job_build/job_build_and_infer_api.h"

namespace oneflow {

struct ModelVersionPolicy { 
  bool latest = true;
  int version = 1;
};

class SessionOption {
 public:
  SessionOption() : device_tag("gpu"), device_num(1), is_mirrored_view(false) {}

 private:
  std::string device_tag;
  int device_num;
  bool is_mirrored_view;
};

class InferenceSession {
 public:
  explicit InferenceSession(const SessionOption& option);
  InferenceSession(const InferenceSession&) = delete;
  InferenceSession(InferenceSession&&) = delete;
  ~InferenceSession();

  Maybe<void> Init();
  void Close();
  void OpenCtx(std::string job_name, JobSignatureDef signature, int batch_size);
  void CloseCtx();
  void Compile(std::vector<OperatorConf> op_list);
  void Launch();
  void LoadModel(std::string saved_model_dir,
                      int model_version,
                      std::string saved_model_meta_file_basename,
                      std::string graph_name = "",
                      std::string signature_name = "");
  void Run(std::string job_name);
  void AsyncRun(std::string job_name);
  void WaitForAllJobsFinished();

  void SetCheckpointPath(std::string checkpoint_path);
  void SetJobSignature(std::string job_name, JobSignatureDef)
  void SetJobBatchSize(std::string job_name, int batch_size);

  void PrintJobSet();
  std::vector<JobConfigProto> ListJobs();
  std::vector<std::string> ListInputs();
  std::vector<std::string> ListOutputs();

 private:
  enum class SessionStatus { OPEN = 1, RUNNING = 2, CLOSED = 3 };

  Maybe<void> InitEventLoop();
  bool CheckStatus(const std::vector<SessionStatus>& status);
  void MakeConfigProto();

  std::shared_ptr<cfg::JobConfProto> GetJobConf(std::string job_name);
  
  void RunJob();
  void RunPushJobs();
  void RunPullJobs();
  void MakePullJobCb();
  void RunLoadCheckpointJob();

  SessionOption option_;
  bool is_mirrored_;
  std::string checkpoint_path_;
  std::map<std::string, std::shared_ptr<cfg::JobConfigProto>> job_name2job_conf_;
  InterUserJobInfo inter_user_job_info_;
  std::string cur_job_name_;
  std::map<std::string, OpBlobInfo> inferface_name2info_;
  std::map<std::string, std::future> output_name2future_;
  std::vector<std::future> job_futures_;
  SessionStatus status_;
 
  ConfigProto config_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_INFERENCE_SESSION_H_
