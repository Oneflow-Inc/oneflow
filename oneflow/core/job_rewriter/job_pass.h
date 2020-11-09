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
#ifndef ONEFLOW_CORE_JOB_REWRITER_JOB_PASS_H_
#define ONEFLOW_CORE_JOB_REWRITER_JOB_PASS_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class JobPassCtx;

class JobPass {
 public:
  JobPass() = default;
  virtual ~JobPass() = default;

  Maybe<void> operator()(Job* job, JobPassCtx* ctx) const { return Apply(job, ctx); }
  virtual Maybe<void> Apply(Job* job, JobPassCtx* ctx) const = 0;
};

class JobPassState {
 public:
  virtual ~JobPassState() = default;

 protected:
  JobPassState() = default;
};

class JobPassCtx {
 public:
  JobPassCtx(const JobPassCtx&) = delete;
  JobPassCtx(JobPassCtx&&) = delete;
  JobPassCtx(const JobDesc& job_desc) : job_desc_(&job_desc) {}
  ~JobPassCtx() = default;

  const JobDesc& job_desc() const { return *job_desc_; }

  template<typename T>
  Maybe<const T&> GetState(const std::string& key) const {
    const auto& iter = key2state_.find(key);
    CHECK_OR_RETURN(iter != key2state_.end());
    const T* ptr = dynamic_cast<T*>(iter->second.get());
    CHECK_NOTNULL_OR_RETURN(ptr) << typeid(*iter->second).name();
    return *ptr;
  }

  template<typename T>
  Maybe<T*> MutableState(const std::string& key) {
    const auto& iter = key2state_.find(key);
    CHECK_OR_RETURN(iter != key2state_.end());
    T* ptr = dynamic_cast<T*>(iter->second.get());
    CHECK_NOTNULL_OR_RETURN(ptr) << typeid(*iter->second).name();
    return ptr;
  }

  template<typename T>
  Maybe<bool> HasState(const std::string& key) const {
    const auto& iter = key2state_.find(key);
    return (iter != key2state_.end());
  }

  Maybe<void> ResetState(const std::string& key, std::unique_ptr<JobPassState>&& state) {
    if (!state) {
      key2state_.erase(key);
    } else {
      key2state_.emplace(key, std::move(state));
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> ResetState(const std::string& key) {
    key2state_.erase(key);
    return Maybe<void>::Ok();
  }

 private:
  const JobDesc* job_desc_;
  HashMap<std::string, std::unique_ptr<JobPassState>> key2state_;
};

#define REGISTER_JOB_PASS(pass_name, pass_type) COMMAND(RegisterJobPass(pass_name, new pass_type))

void RegisterJobPass(const std::string& pass_name, const JobPass* pass);
bool HasJobPass(const std::string& pass_name);
const JobPass& JobPass4Name(const std::string& pass_name);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_JOB_PASS_H_
