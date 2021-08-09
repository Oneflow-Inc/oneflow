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

#include "oneflow/api/cpp/job_instance.h"

namespace oneflow {

explicit JobInstance::JobInstance(std::string job_name,
    std::string sole_input_op_name_in_user_job = "",
    std::string sole_output_op_name_in_user_job = "",
    push_cb = []{},
    pull_cb = []{},
    finish_cb = []{}) {
  
}

~JobInstance::JobInstance(){}

std::string JobInstance::job_name() override const { return this->job_name_; }

std::string JobInstance::sole_input_op_name_in_user_job() override const {
    return this->sole_input_op_name_in_user_job_;
}

std::string JobInstance::sole_output_op_name_in_user_job() override const {
    return this->sole_output_op_name_in_user_job_;
}

void JobInstance::PushBlob(uint64_t ofblob_ptr) override const {
    this->push_cb_(reinterpret_cast<OfBlob*>(of_blob_ptr));
}

void JobInstance::PullBlob(uint64_t ofblob_ptr) override const {
    this->pull_cb_(reinterpret_cast<OfBlob*>(of_blob_ptr));
}

void JobInstance::Finish() const {
    this->finish_cb_();

    for (auto& post_finish_cb : this->post_finish_cbs_)
        post_finish_cb(this);
}

void JobInstance::AddPostFinishCallback(std::function<void(JobInstance*)> cb) {
    this->post_finish_cbs_.push_back(cb);
}

std::shared_ptr<JobInstance> MakeUserJobInstance(
  std::string job_name, 
  std::function<void()> finish_cb) {
  return std::make_shared<JobInstance>()
}

std::shared_ptr<JobInstance> MakePullJobInstance(
  std::string job_name, 
  std::string op_name,
  std::function<void(OfBlob*)> pull_cb,
  std::function<void()> finish_cb {

}

std::shared_ptr<JobInstance> MakePushJobInstance(
  std::string job_name, 
  std::string op_name,
  std::function<void(OfBlob*)> push_cb,
  std::function<void()> finish_cb {

}

std::shared_ptr<JobInstance> MakeArgPassJobInstance(
  std::string job_name,
  std::string src_op_name;
  std::string dst_op_name;
  std::function<void()> finish_cb {

}

}  // namespace oneflow