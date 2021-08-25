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

CPPJobInstance::CPPJobInstance(std::string job_name,
                                std::string sole_input_op_name_in_user_job,
                                std::string sole_output_op_name_in_user_job,
                                std::function<void(OfBlob*)> push_cb,
                                std::function<void(OfBlob*)> pull_cb,
                                std::function<void()> finish_cb) 
    : job_name_(job_name), 
      sole_input_op_name_in_user_job_(sole_input_op_name_in_user_job),
      sole_output_op_name_in_user_job_(sole_output_op_name_in_user_job),
      push_cb_(push_cb),
      pull_cb_(pull_cb),
      finish_cb_(finish_cb) {}

~CPPJobInstance::CPPJobInstance(){}

std::string CPPJobInstance::job_name() const override { return this->job_name_; }

std::string CPPJobInstance::sole_input_op_name_in_user_job() const override {
  return this->sole_input_op_name_in_user_job_;
}

std::string CPPJobInstance::sole_output_op_name_in_user_job() const override {
  return this->sole_output_op_name_in_user_job_;
}

void CPPJobInstance::PushBlob(uint64_t ofblob_ptr) const override {
  this->push_cb_(reinterpret_cast<OfBlob*>(of_blob_ptr));
}

void CPPJobInstance::PullBlob(uint64_t ofblob_ptr) const override {
  this->pull_cb_(reinterpret_cast<OfBlob*>(of_blob_ptr));
}

void CPPJobInstance::Finish() const override {
  this->finish_cb_();

  for (auto& post_finish_cb : this->post_finish_cbs_)
      post_finish_cb(this);
}

void CPPJobInstance::AddPostFinishCallback(std::function<void(JobInstance*)> cb) {
  this->post_finish_cbs_.push_back(cb);
}

std::shared_ptr<CPPJobInstance> MakeUserJobInstance(
  std::string job_name, 
  std::function<void()> finish_cb = std::function<void()>()) {
  return std::make_shared<CPPJobInstance>(job_name, "", "",
    std::function<void(OfBlob*)>(), std::function<void(OfBlob*)>(),
    finish_cb);
}

std::shared_ptr<CPPJobInstance> MakePullJobInstance(
  std::string job_name, 
  std::string op_name,
  std::function<void(OfBlob*)> pull_cb,
  std::function<void()> finish_cb = std::function<void()>()) {
  return std::make_shared<CPPJobInstance>(
      job_name, op_name, "", std::function<void(OfBlob*)>(), pull_cb, finish_cb);
}

std::shared_ptr<CPPJobInstance> MakePushJobInstance(
  std::string job_name, 
  std::string op_name,
  std::function<void(OfBlob*)> push_cb,
  std::function<void()> finish_cb = std::function<void()>()) {
  return std::make_shared<CPPJobInstance>(
      job_name, "", op_name, push_cb, std::function<void(OfBlob*)>(), finish_cb);
}

std::shared_ptr<CPPJobInstance> MakeArgPassJobInstance(
  std::string job_name,
  std::string src_op_name;
  std::string dst_op_name;
  std::function<void()> finish_cb = std::function<void()>()) {
  return std::make_shared<CPPJobInstance>(job_name, src_op_name, dst_op_name,
    std::function<void(OfBlob*)>(), std::function<void(OfBlob*)>(),
    finish_cb);
}

}  // namespace oneflow