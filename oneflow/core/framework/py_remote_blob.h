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
#ifndef ONEFLOW_CORE_FRAMEWORK_PY_REMOTE_BLOB_H_
#define ONEFLOW_CORE_FRAMEWORK_PY_REMOTE_BLOB_H_

#include "oneflow/core/common/global.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/py_blob_desc.h"

namespace oneflow {

namespace compatible_py {

namespace {

Maybe<JobBuildAndInferCtxMgr*> GlobalJobBuildAndInferCtxMgr() {
  if (EagerExecutionEnabled()) {
    return JUST(GlobalMaybe<EagerJobBuildAndInferCtxMgr>());
  } else {
    return JUST(GlobalMaybe<LazyJobBuildAndInferCtxMgr>());
  }
}

}  // namespace

class ConsistentBlob : public BlobDesc {
 public:
  ConsistentBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                 const std::shared_ptr<Distribute>& distribute)
      : BlobDesc(lbi, distribute), parallel_size_(0) {
    if (job_name.empty()) {
      auto* mgr = CHECK_JUST(GlobalJobBuildAndInferCtxMgr());
      job_name_ = *CHECK_JUST(mgr->GetCurrentJobName());
    } else {
      job_name_ = job_name;
    }
  }
  ConsistentBlob(const ConsistentBlob& consistent_blob) = default;
  ~ConsistentBlob() = default;

  std::string job_name() const { return job_name_; }
  std::shared_ptr<BlobDesc> Clone() const override;

  std::shared_ptr<BlobDesc> with_distribute(
      const std::shared_ptr<Distribute>& distribute) const override {
    std::shared_ptr<BlobDesc> ret = Clone();
    ret->set_distribute(distribute);
    return ret;
  }

  int64_t parallel_size() {
    if (parallel_size_ == 0) {
      std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
      ParallelConf proto_parallel_conf;
      cfg_parallel_conf->ToProto(&proto_parallel_conf);
      ParallelDesc parallel_desc(proto_parallel_conf);
      parallel_size_ = parallel_desc.parallel_num();
    }
    return parallel_size_;
  }

  void set_job_name(std::string job_name) { job_name_ = job_name; }

 private:
  std::string job_name_;
  int64_t parallel_size_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PY_REMOTE_BLOB_H_
