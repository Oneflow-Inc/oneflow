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
#ifndef ONEFLOW_API_CPP_JOB_INSTANCE_H_
#define ONEFLOW_API_CPP_JOB_INSTANCE_H_

#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/api/python/job_build/job_build_and_infer_api.h"
#include "oneflow/core/job/job_instance.h"

namespace oneflow {

class CPPJobInstance : public JobInstance {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CPPJobInstance);
  explicit CPPJobInstance(std::string job_name,
                          std::string sole_input_op_name_in_user_job,
                          std::string sole_output_op_name_in_user_job,
                          std::function<void(OfBlob*)> push_cb,
                          std::function<void(OfBlob*)> pull_cb,
                          std::function<void()> finish_cb);
  ~CPPJobInstance();

  std::string job_name() const override;
  std::string sole_input_op_name_in_user_job() const override;
  std::string sole_output_op_name_in_user_job() const override;
  void PushBlob(uint64_t ofblob_ptr) const override;
  void PullBlob(uint64_t ofblob_ptr) const override;
  void Finish() const override;
  void AddPostFinishCallback(std::function<void(JobInstance*)> cb);

 private:
  int thisown;
  std::string job_name_;
  std::string sole_input_op_name_in_user_job_;
  std::string sole_output_op_name_in_user_job_;
  std::function<void(OfBlob*)> push_cb_;
  std::function<void(OfBlob*)> pull_cb_;
  std::function<void()> finish_cb_;
  std::vector<std::function<void(CPPJobInstance*)>> post_finish_cbs_;
};

std::shared_ptr<CPPJobInstance> MakeUserJobInstance(
  std::string job_name, 
  std::function<void()> finish_cb = std::function<void()>()
);

std::shared_ptr<CPPJobInstance> MakePullJobInstance(
  std::string job_name, 
  std::string op_name,
  std::function<void(OfBlob*)> pull_cb,
  std::function<void()> finish_cb = std::function<void()>()
);

std::shared_ptr<CPPJobInstance> MakePushJobInstance(
  std::string job_name, 
  std::string op_name,
  std::function<void(OfBlob*)> push_cb,
  std::function<void()> finish_cb = std::function<void()>()
);

std::shared_ptr<CPPJobInstance> MakeArgPassJobInstance(
  std::string job_name,
  std::string src_op_name,
  std::string dst_op_name,
  std::function<void()> finish_cb = std::function<void()>()
);

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_JOB_INSTANCE_H_
