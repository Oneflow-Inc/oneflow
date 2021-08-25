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

#include <future>
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/api/python/job_build/job_build_and_infer_api.h"

namespace oneflow {

struct ModelVersionPolicy { 
  bool latest = true;
  int version = 1;
};

struct SessionOption {
  SessionOption() : device_tag("gpu"), device_num(1), 
                    is_mirrored_view(false) {}

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

  void Launch();
  Maybe<void> Close();
  void LoadModel(std::string saved_model_dir,
                 ModelVersionPolicy model_version_policy,
                 std::string saved_model_meta_file_basename,
                 std::string graph_name = "",
                 std::string signature_name = "");

  std::map<std::string, std::shared_ptr<Tensor>> Run(std::string job_name, 
                                    std::map<std::string, std::shared_ptr<Tensor>>& input_tensors);

 private:
  enum class SessionStatus { OPEN = 1, RUNNING = 2, CLOSED = 3 };

  Maybe<void> Init();
  Maybe<void> OpenCtx(std::string job_name, JobSignatureDef* signature, int batch_size);
  Maybe<void> CloseCtx();
  Maybe<void> Compile(std::vector<OperatorConf>& op_list);

  Maybe<void> LoadModel_(std::string saved_model_dir,
                         ModelVersionPolicy model_version_policy,
                         std::string saved_model_meta_file_basename,
                         std::string graph_name = "",
                         std::string signature_name = "");

  std::shared_ptr<JobConfigProto> GetJobConf(std::string job_name);
  
  Maybe<void> RunJob(std::shared_ptr<CPPJobInstance> job_inst);
  Maybe<void> RunPushJobs(std::map<std::string, std::shared_ptr<Tensor>>& input_tensors);
  Maybe<void> RunPullJobs(std::map<std::string, std::shared_ptr<Tensor>>& output_tensors);
  Maybe<void> RunLoadCheckpointJob();

  void WaitForAllJobsFinished();

  void SetCheckpointPath(std::string checkpoint_path);
  void SetJobSignature(std::string job_name, JobSignatureDef*);
  void SetJobBatchSize(std::string job_name, int batch_size);

  Maybe<void> CheckStatus(const std::vector<SessionStatus>& status);
  Maybe<void> CheckStatus(SessionStatus status);
  Maybe<void> MakeConfigProto();

  void PrintJobSet();
  std::vector<std::string> ListJobs();
  std::vector<std::string> ListInputs();
  std::vector<std::string> ListOutputs();

  SessionOption option_;
  bool is_mirrored_;
  std::string checkpoint_path_;
  std::map<std::string, std::shared_ptr<JobConfigProto>> job_name2job_conf_;
  InterUserJobInfo* inter_user_job_info_;
  std::string cur_job_name_;
  std::vector<std::shared_ptr<std::promise<void>>> job_promises_;
  SessionStatus status_;
 
  ConfigProto config_proto_;
};

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_INFERENCE_SESSION_H_
