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

#include <vector>
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/framework/py_blob_desc.h"

namespace oneflow {

namespace compatible_py {

namespace {

inline Maybe<JobBuildAndInferCtxMgr*> GlobalJobBuildAndInferCtxMgr() {
  if (EagerExecutionEnabled()) {
    return JUST(GlobalMaybe<EagerJobBuildAndInferCtxMgr>());
  } else {
    return JUST(GlobalMaybe<LazyJobBuildAndInferCtxMgr>());
  }
}

inline Maybe<JobBuildAndInferCtx*> GetJobBuildAndInferCtx(const std::string& job_name) {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->FindJobBuildAndInferCtx(job_name);
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
  virtual std::shared_ptr<BlobDesc> Clone() const override;

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

class LazyConsistentBlob : public ConsistentBlob {
 public:
  LazyConsistentBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                     const std::shared_ptr<Distribute>& distribute)
      : ConsistentBlob(lbi, job_name, distribute) {}
  LazyConsistentBlob(const LazyConsistentBlob& lazy_consistent_blob) = default;
  ~LazyConsistentBlob() = default;

  std::shared_ptr<BlobDesc> Clone() const override {
    std::shared_ptr<BlobDesc> ret;
    BlobDesc* blob_desc = new LazyConsistentBlob(lbi(), job_name(), distribute());
    ret.reset(blob_desc);
    return ret;
  }

  virtual std::string get_shape_log_warning() const { return std::string(""); }

  std::shared_ptr<Shape> shape() const override {
    const std::string& log_warning = get_shape_log_warning();
    if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    auto shape = CHECK_JUST(ctx->GetStaticShape(logical_blob_name()));
    return shape;
  }

  DataType dtype() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    return CHECK_JUST(ctx->GetDataType(logical_blob_name()));
  }

  int64_t batch_axis() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    auto opt_int64 = CHECK_JUST(ctx->GetBatchAxis(logical_blob_name()));
    if (opt_int64->has_value()) { return opt_int64->value(); }
    return INVALID_BATCH_AXIS;
  }

  int64_t split_axis() const {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    auto opt_int64 = CHECK_JUST(ctx->GetSplitAxisFromProducerView(logical_blob_name()));
    if (opt_int64->has_value()) { return opt_int64->value(); }
    return INVALID_SPLIT_AXIS;
  }

  bool is_dynamic() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    return CHECK_JUST(ctx->IsDynamic(logical_blob_name()));
  }

  bool is_tensor_list() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    return CHECK_JUST(ctx->IsTensorList(logical_blob_name()));
  }

  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    return CHECK_JUST(ctx->GetParallelDescFromProducerView(logical_blob_name()))
        ->cfg_parallel_conf();
  }

  bool IdenticalTo(const std::shared_ptr<LazyConsistentBlob>& rhs) {
    return true && unique_name() == rhs->unique_name() && *shape() == *rhs->shape()
           && batch_axis() == rhs->batch_axis() && split_axis() == rhs->split_axis()
           && is_dynamic() == rhs->is_dynamic() && is_tensor_list() == rhs->is_tensor_list();
  }

  std::shared_ptr<LazyConsistentBlob> with_gradient_distribute(
      const std::shared_ptr<Distribute>& distribute) {
    UNIMPLEMENTED();
  }
};

class MirroredBlob : public BlobDesc {
 public:
  MirroredBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
               const std::shared_ptr<Distribute>& distribute)
      : BlobDesc(lbi, distribute), parallel_size_(0) {
    if (job_name.empty()) {
      auto* mgr = CHECK_JUST(GlobalJobBuildAndInferCtxMgr());
      job_name_ = *CHECK_JUST(mgr->GetCurrentJobName());
    } else {
      job_name_ = job_name;
    }
  }
  MirroredBlob(const MirroredBlob& mirrored_blob) = default;
  ~MirroredBlob() = default;

  std::string job_name() const { return job_name_; }

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

class LazyMirroredBlob : public MirroredBlob {
 public:
  LazyMirroredBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                   const std::shared_ptr<Distribute>& distribute)
      : MirroredBlob(lbi, job_name, distribute) {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(this->job_name()));
    int lbi_num = CHECK_JUST(ctx->MirroredBlobGetNumSubLbi(this->logical_blob_name()));
    for (int i = 0; i < lbi_num; ++i) {
      std::shared_ptr<cfg::LogicalBlobId> sub_lbi = std::make_shared<cfg::LogicalBlobId>(
          *CHECK_JUST(ctx->MirroredBlobGetSubLbi(this->logical_blob_name(), i)));
      sub_consistent_blob_list_.emplace_back(
          std::make_shared<LazyConsistentBlob>(sub_lbi, "", GlobalAutoDistribute()));
    }
  }
  LazyMirroredBlob(const LazyMirroredBlob& lazy_mirrored_blob) = default;
  ~LazyMirroredBlob() = default;

  std::vector<std::shared_ptr<LazyConsistentBlob>> sub_consistent_blob_list() {
    return sub_consistent_blob_list_;
  }

  virtual std::string get_shape_log_warning() const { return std::string(""); }

  std::shared_ptr<Shape> shape() const override {
    const std::string& log_warning = get_shape_log_warning();
    if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    auto shape = CHECK_JUST(ctx->MirroredBlobGetStaticShape(logical_blob_name()));
    return shape;
  }

  DataType dtype() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    return CHECK_JUST(ctx->MirroredBlobGetDataType(logical_blob_name()));
  }

  int64_t batch_axis() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetBatchAxis(logical_blob_name()));
    if (opt_int64->has_value()) { return opt_int64->value(); }
    return INVALID_BATCH_AXIS;
  }

  int64_t split_axis() const {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetSplitAxisFromProducerView(logical_blob_name()));
    if (opt_int64->has_value()) { return opt_int64->value(); }
    return INVALID_SPLIT_AXIS;
  }

  bool is_dynamic() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    return CHECK_JUST(ctx->MirroredBlobIsDynamic(logical_blob_name()));
  }

  bool is_tensor_list() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    return CHECK_JUST(ctx->MirroredBlobIsTensorList(logical_blob_name()));
  }

  std::shared_ptr<cfg::ParallelConf> parallel_conf() const override {
    auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
    return CHECK_JUST(ctx->MirroredBlobGetParallelDescFromProducerView(logical_blob_name()))
        ->cfg_parallel_conf();
  }

  std::shared_ptr<LazyMirroredBlob> with_gradient_distribute(
      const std::shared_ptr<Distribute>& distribute) {
    UNIMPLEMENTED();
  }

 private:
  std::vector<std::shared_ptr<LazyConsistentBlob>> sub_consistent_blob_list_;
};

}  // namespace compatible_py

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_PY_REMOTE_BLOB_H_
