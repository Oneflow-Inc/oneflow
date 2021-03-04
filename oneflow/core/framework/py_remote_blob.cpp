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
#include "oneflow/core/framework/py_remote_blob.h"

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

Maybe<JobBuildAndInferCtx*> GetJobBuildAndInferCtx(const std::string& job_name) {
  auto* mgr = JUST(GlobalJobBuildAndInferCtxMgr());
  return mgr->FindJobBuildAndInferCtx(job_name);
}

}  // namespace

ConsistentBlob::ConsistentBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                               const std::string& job_name,
                               const std::shared_ptr<Distribute>& distribute)
    : BlobDesc(lbi, distribute), parallel_size_(0) {
  if (job_name.empty()) {
    auto* mgr = CHECK_JUST(GlobalJobBuildAndInferCtxMgr());
    job_name_ = *CHECK_JUST(mgr->GetCurrentJobName());
  } else {
    job_name_ = job_name;
  }
}

std::string ConsistentBlob::job_name() const { return job_name_; }

int64_t ConsistentBlob::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

void ConsistentBlob::set_job_name(std::string job_name) { job_name_ = job_name; }

LazyConsistentBlob::LazyConsistentBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                       const std::string& job_name,
                                       const std::shared_ptr<Distribute>& distribute)
    : ConsistentBlob(lbi, job_name, distribute) {}

std::string LazyConsistentBlob::get_lazy_shape_log_warning() const { return std::string(""); }

std::shared_ptr<Shape> LazyConsistentBlob::shape() const {
  const std::string& log_warning = get_lazy_shape_log_warning();
  if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetStaticShape(logical_blob_name()));
}

DataType LazyConsistentBlob::dtype() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetDataType(logical_blob_name()));
}

int64_t LazyConsistentBlob::split_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->GetSplitAxisFromProducerView(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_SPLIT_AXIS;
}

bool LazyConsistentBlob::is_dynamic() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->IsDynamic(logical_blob_name()));
}

bool LazyConsistentBlob::is_tensor_list() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->IsTensorList(logical_blob_name()));
}

std::shared_ptr<cfg::ParallelConf> LazyConsistentBlob::parallel_conf() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetParallelDescFromProducerView(logical_blob_name()))->cfg_parallel_conf();
}

bool LazyConsistentBlob::IdenticalTo(const std::shared_ptr<LazyConsistentBlob>& rhs) const {
  return true && unique_name() == rhs->unique_name() && *shape() == *rhs->shape()
         && split_axis() == rhs->split_axis() && is_dynamic() == rhs->is_dynamic()
         && is_tensor_list() == rhs->is_tensor_list();
}

MirroredBlob::MirroredBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                           const std::string& job_name,
                           const std::shared_ptr<Distribute>& distribute)
    : BlobDesc(lbi, distribute), parallel_size_(0) {
  if (job_name.empty()) {
    auto* mgr = CHECK_JUST(GlobalJobBuildAndInferCtxMgr());
    job_name_ = *CHECK_JUST(mgr->GetCurrentJobName());
  } else {
    job_name_ = job_name;
  }
}

std::string MirroredBlob::job_name() const { return job_name_; }

int64_t MirroredBlob::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

void MirroredBlob::set_job_name(std::string job_name) { job_name_ = job_name; }

LazyMirroredBlob::LazyMirroredBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                   const std::string& job_name,
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

std::vector<std::shared_ptr<LazyConsistentBlob>> LazyMirroredBlob::sub_consistent_blob_list() {
  return sub_consistent_blob_list_;
}

std::string LazyMirroredBlob::get_mirror_shape_log_warning() const { return std::string(""); }

std::shared_ptr<Shape> LazyMirroredBlob::shape() const {
  const std::string& log_warning = get_mirror_shape_log_warning();
  if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto shape = CHECK_JUST(ctx->MirroredBlobGetStaticShape(logical_blob_name()));
  return shape;
}

DataType LazyMirroredBlob::dtype() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobGetDataType(logical_blob_name()));
}

int64_t LazyMirroredBlob::split_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetSplitAxisFromProducerView(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_SPLIT_AXIS;
}

bool LazyMirroredBlob::is_dynamic() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobIsDynamic(logical_blob_name()));
}

bool LazyMirroredBlob::is_tensor_list() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobIsTensorList(logical_blob_name()));
}

std::shared_ptr<cfg::ParallelConf> LazyMirroredBlob::parallel_conf() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobGetParallelDescFromProducerView(logical_blob_name()))
      ->cfg_parallel_conf();
}

EagerBlobTrait::EagerBlobTrait() : parallel_size_(0) {}

EagerBlobTrait::~EagerBlobTrait() {
  registered_blob_access_->blob_register()->CloseRegisteredBlobAccess(logical_blob_name_);
}

int64_t EagerBlobTrait::numpy_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

int64_t EagerBlobTrait::numpy_list_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

std::shared_ptr<Shape> EagerBlobTrait::shape() const {
  return blob_object()->op_arg_blob_attr()->shape();
}

cfg::DataType EagerBlobTrait::dtype() const {
  return blob_object()->op_arg_blob_attr()->get_dtype();
}

int64_t EagerBlobTrait::split_axis() const {
  auto sbp_parallel = blob_object()->op_arg_parallel_attr()->sbp_parallel();
  if (sbp_parallel->has_split_parallel()) {
    return sbp_parallel->split_parallel().axis();
  } else if (sbp_parallel->has_broadcast_parallel()) {
    return INVALID_SPLIT_AXIS;
  } else if (sbp_parallel->has_partial_sum_parallel()) {
    return INVALID_SPLIT_AXIS;
  } else {
    UNIMPLEMENTED();
  }
}

bool EagerBlobTrait::is_dynamic() const { return blob_object()->op_arg_blob_attr()->is_dynamic(); }

bool EagerBlobTrait::is_tensor_list() const {
  return blob_object()->op_arg_blob_attr()->is_tensor_list();
}

std::shared_ptr<cfg::ParallelConf> EagerBlobTrait::parallel_conf() const {
  return blob_object()->parallel_desc_symbol()->cfg_parallel_conf();
}

int64_t EagerBlobTrait::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

std::shared_ptr<BlobObject> EagerBlobTrait::blob_object() const {
  return registered_blob_access_->blob_object();
}

void EagerBlobTrait::_Init(const std::string logical_blob_name,
                           const std::shared_ptr<BlobObject>& blob_object,
                           const std::shared_ptr<BlobRegister>& blob_register) {
  logical_blob_name_ = logical_blob_name;
  std::shared_ptr<RegisteredBlobAccess> access =
      blob_register->OpenRegisteredBlobAccess(logical_blob_name, blob_object);
  registered_blob_access_ = access;
}

bool EagerBlobTrait::IdenticalTo(const std::shared_ptr<EagerBlobTrait>& rhs) const {
  return (blob_object()->op_arg_blob_attr() == rhs->blob_object()->op_arg_blob_attr())
         && (blob_object()->op_arg_parallel_attr() == rhs->blob_object()->op_arg_parallel_attr());
}

EagerConsistentBlob::EagerConsistentBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                         const std::shared_ptr<BlobObject>& blob_object,
                                         const std::shared_ptr<BlobRegister>& blob_register,
                                         const std::string& job_name,
                                         const std::shared_ptr<Distribute>& distribute)
    : EagerBlobTrait(), ConsistentBlob(lbi, job_name, distribute) {
  std::string logical_blob_name = lbi->op_name() + "/" + lbi->blob_name();
  _Init(logical_blob_name, blob_object, blob_register);
}

EagerMirroredBlob::EagerMirroredBlob(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                     const std::shared_ptr<BlobObject>& blob_object,
                                     const std::shared_ptr<BlobRegister>& blob_register,
                                     const std::string& job_name,
                                     const std::shared_ptr<Distribute>& distribute)
    : EagerBlobTrait(), MirroredBlob(lbi, job_name, distribute) {
  std::string logical_blob_name = lbi->op_name() + "/" + lbi->blob_name();
  _Init(logical_blob_name, blob_object, blob_register);
}

}  // namespace compatible_py

}  // namespace oneflow
