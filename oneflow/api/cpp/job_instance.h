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
#include "oneflow/core/job/foreign_job_instance.h"

namespace oneflow {

class JobInstance : public ForeignJobInstance {
 public:
  OF_DISALLOW_COPY_AND_MOVE(JobInstance);
  explicit JobInstance(std::string job_name,
                       std::string sole_input_op_name_in_user_job = "",
                       std::string sole_output_op_name_in_user_job = "",
                       push_cb = []{},
                       pull_cb = []{},
                       finish_cb = []{});
  ~JobInstance();

  std::string job_name() override const;
  std::string sole_input_op_name_in_user_job() override const;
  std::string sole_output_op_name_in_user_job() override const;
  void PushBlob(uint64_t ofblob_ptr) override const;
  void PullBlob(uint64_t ofblob_ptr) override const;
  void Finish() const;
  void AddPostFinishCallback();

 private:
  int thisown;
  std::string job_name_;
  std::string sole_input_op_name_in_user_job_;
  std::string sole_output_op_name_in_user_job_;
  std::function<void(OfBlob*)> push_cb_;
  std::function<void(OfBlob*)> pull_cb_;
  std::function<void()> finish_cb_;
  std::vector<std::function<void(JobInstance*)>> post_finish_cbs_;
};

std::shared_ptr<JobInstance> MakeUserJobInstance(
  std::string job_name, 
  std::function<void()> finish_cb
);

std::shared_ptr<JobInstance> MakePullJobInstance(
  std::string job_name, 
  std::string op_name,
  std::function<void(OfBlob*)> pull_cb,
  std::function<void()> finish_cb
);

std::shared_ptr<JobInstance> MakePushJobInstance(
  std::string job_name, 
  std::string op_name,
  std::function<void(OfBlob*)> push_cb,
  std::function<void()> finish_cb
);

std::shared_ptr<JobInstance> MakeArgPassJobInstance(
  std::string job_name,
  std::string src_op_name;
  std::string dst_op_name;
  std::function<void()> finish_cb
);

}  // namespace oneflow

#endif  // ONEFLOW_API_CPP_JOB_INSTANCE_H_
