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
#include "oneflow/core/framework/tensor_impl.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

namespace one {

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

static int64_t INVALID_BATCH_AXIS = -22;
static int64_t INVALID_SPLIT_AXIS = -22;

TensorImpl::TensorImpl(const std::shared_ptr<cfg::LogicalBlobId>& lbi, const std::string& job_name,
                       const std::shared_ptr<compatible_py::Distribute>& distribute)
    : lbi_(lbi), distribute_(distribute), job_name_(job_name) {
  lbn_ = lbi->op_name() + "/" + lbi->blob_name();
}

std::shared_ptr<cfg::LogicalBlobId> TensorImpl::lbi() const { return lbi_; }
std::string TensorImpl::logical_blob_name() const { return lbn_; }
std::string TensorImpl::op_name() const { return lbi_->op_name(); }
std::string TensorImpl::blob_name() const { return lbi_->blob_name(); }
bool TensorImpl::has_batch_axis() const { return batch_axis() != INVALID_BATCH_AXIS; }
std::shared_ptr<compatible_py::Distribute> TensorImpl::distribute() const { return distribute_; }
std::string TensorImpl::unique_name() const { return lbn_ + *CHECK_JUST(Distribute2Str()); }
void TensorImpl::set_distribute(const std::shared_ptr<compatible_py::Distribute> distribute) {
  distribute_ = distribute;
}
Maybe<std::string> TensorImpl::Distribute2Str() const {
  if (std::dynamic_pointer_cast<compatible_py::AutoDistribute>(distribute_)) {
    return std::string("");
  } else if (std::dynamic_pointer_cast<compatible_py::BroadcastDistribute>(distribute_)) {
    return std::string(":B");
  } else if (std::dynamic_pointer_cast<compatible_py::SplitDistribute>(distribute_)) {
    return std::string(":S") + std::to_string(distribute_->axis());
  } else {
    OF_UNIMPLEMENTED();
  }
  return std::string("");
}

// MirroredLazyTensorImpl:LazyTensorImpl
std::vector<std::shared_ptr<ConsistentLazyBlobDesc>> MirroredLazyTensorImpl::sub_consistent_blob_list() {
  return sub_consistent_blob_list_;
}

std::string MirroredLazyTensorImpl::get_mirror_shape_log_warning() const { return std::string(""); }

std::shared_ptr<Shape> MirroredLazyTensorImpl::shape() const {
  const std::string& log_warning = get_mirror_shape_log_warning();
  if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto shape = CHECK_JUST(ctx->MirroredBlobGetStaticShape(logical_blob_name()));
  return shape;
}

DataType MirroredLazyTensorImpl::dtype() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobGetDataType(logical_blob_name()));
}

int64_t MirroredLazyTensorImpl::batch_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetBatchAxis(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_BATCH_AXIS;
}

int64_t MirroredLazyTensorImpl::split_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetSplitAxisFromProducerView(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_SPLIT_AXIS;
}

bool MirroredLazyTensorImpl::is_dynamic() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobIsDynamic(logical_blob_name()));
}

bool MirroredLazyTensorImpl::is_tensor_list() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobIsTensorList(logical_blob_name()));
}

std::shared_ptr<cfg::ParallelConf> MirroredLazyTensorImpl::parallel_conf() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobGetParallelDescFromProducerView(logical_blob_name()))
      ->cfg_parallel_conf();
}

// ConsistentLazyTensorImpl
std::string ConsistentLazyTensorImpl::get_lazy_shape_log_warning() const { return std::string(""); }

std::shared_ptr<Shape> ConsistentLazyTensorImpl::shape() const {
  const std::string& log_warning = get_lazy_shape_log_warning();
  if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetStaticShape(logical_blob_name()));
}

DataType ConsistentLazyTensorImpl::dtype() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetDataType(logical_blob_name()));
}

int64_t ConsistentLazyTensorImpl::batch_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->GetBatchAxis(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_BATCH_AXIS;
}

int64_t ConsistentLazyTensorImpl::split_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->GetSplitAxisFromProducerView(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_SPLIT_AXIS;
}

bool ConsistentLazyTensorImpl::is_dynamic() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->IsDynamic(logical_blob_name()));
}

bool ConsistentLazyTensorImpl::is_tensor_list() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->IsTensorList(logical_blob_name()));
}

std::shared_ptr<cfg::ParallelConf> ConsistentLazyTensorImpl::parallel_conf() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetParallelDescFromProducerView(logical_blob_name()))->cfg_parallel_conf();
}

bool LazyConsistentBlob::IdenticalTo(const std::shared_ptr<ConsistentLazyTensorImpl>& rhs) const {
  return true && unique_name() == rhs->unique_name() && *shape() == *rhs->shape()
         && batch_axis() == rhs->batch_axis() && split_axis() == rhs->split_axis()
         && is_dynamic() == rhs->is_dynamic() && is_tensor_list() == rhs->is_tensor_list();
}

// EagerTensorImpl
EagerTensorImpl::~EagerTensorImpl() {
  registered_blob_access_->blob_register()->CloseRegisteredBlobAccess(lbn_);
}

int64_t EagerTensorImpl::numpy_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

int64_t EagerTensorImpl::numpy_list_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

std::shared_ptr<Shape> EagerTensorImpl::shape() const {
  return blob_object()->op_arg_blob_attr()->shape();
}

DataType EagerTensorImpl::dtype() const {
  return static_cast<DataType>(blob_object()->op_arg_blob_attr()->get_dtype());
}

int64_t EagerTensorImpl::batch_axis() const {
  auto opt_batch_axis = blob_object()->op_arg_blob_attr()->batch_axis();
  if (opt_batch_axis->has_value()) {
    return opt_batch_axis->value();
  } else {
    return INVALID_BATCH_AXIS;
  }
}

int64_t EagerTensorImpl::split_axis() const {
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

bool EagerTensorImpl::is_dynamic() const { return blob_object()->op_arg_blob_attr()->is_dynamic(); }

bool EagerTensorImpl::is_tensor_list() const {
  return blob_object()->op_arg_blob_attr()->is_tensor_list();
}

std::shared_ptr<cfg::ParallelConf> EagerTensorImpl::parallel_conf() const {
  return blob_object()->parallel_desc_symbol()->cfg_parallel_conf();
}

int64_t EagerTensorImpl::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

std::shared_ptr<compatible_py::BlobObject> EagerTensorImpl::blob_object() const {
  return registered_blob_access_->blob_object();
}

void EagerTensorImpl::_Init(const std::shared_ptr<compatible_py::BlobObject>& blob_object,
                            const std::shared_ptr<compatible_py::BlobRegister>& blob_register) {
  std::shared_ptr<compatible_py::RegisteredBlobAccess> access =
      blob_register->OpenRegisteredBlobAccess(lbn_, blob_object);
  registered_blob_access_ = access;
}

bool EagerTensorImpl::IdenticalTo(const std::shared_ptr<EagerTensorImpl>& rhs) const {
  return (blob_object()->op_arg_blob_attr() == rhs->blob_object()->op_arg_blob_attr())
         && (blob_object()->op_arg_parallel_attr() == rhs->blob_object()->op_arg_parallel_attr());
}

// MirroredEagerTensorImpl
MirroredEagerTensorImpl::MirroredEagerTensorImpl (const std::shared_ptr<Shape>& shape, 
                                                  DataType dtype,
                                                  const std::shared_ptr<Device>& device)
      : EagerTensorImpl(shape, dtype, device) {}

//ConsistentEagerTensorImpl
ConsistentEagerTensorImpl::ConsistentEagerTensorImpl(const std::shared_ptr<Shape>& shape, DataType dtype,
                                                     const std::shared_ptr<compatible_py::Distribute>& distribute, 
                                                     const std::shared_ptr<cfg::ParallelConf>& parallel_conf)
                                :EagerTensorImpl(shape, dtype, distribute, parallel_conf){}


// EagerBlobDesc 
EagerBlobDesc::EagerBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                 const std::string& job_name,
                                 const std::shared_ptr<compatible_py::Distribute>& distribute)
    : TensorImpl(lbi, job_name, distribute) {
  parallel_size_ = 0;
}

EagerBlobDesc::~EagerBlobDesc() {
  registered_blob_access_->blob_register()->CloseRegisteredBlobAccess(logical_blob_name_);
}

int64_t EagerBlobDesc::numpy_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

int64_t EagerBlobDesc::numpy_list_size() const {
  return blob_object()->parallel_desc_symbol()->parallel_num();
}

std::shared_ptr<Shape> EagerBlobDesc::shape() const {
  return blob_object()->op_arg_blob_attr()->shape();
}

cfg::DataType EagerBlobDesc::dtype() const {
  return blob_object()->op_arg_blob_attr()->get_dtype();
}

int64_t EagerBlobDesc::batch_axis() const {
  auto opt_batch_axis = blob_object()->op_arg_blob_attr()->batch_axis();
  if (opt_batch_axis->has_value()) {
    return opt_batch_axis->value();
  } else {
    return INVALID_BATCH_AXIS;
  }
}

int64_t EagerBlobDesc::split_axis() const {
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

bool EagerBlobDesc::is_dynamic() const { return blob_object()->op_arg_blob_attr()->is_dynamic(); }

bool EagerBlobDesc::is_tensor_list() const {
  return blob_object()->op_arg_blob_attr()->is_tensor_list();
}

std::shared_ptr<cfg::ParallelConf> EagerBlobDesc::parallel_conf() const {
  return blob_object()->parallel_desc_symbol()->cfg_parallel_conf();
}

int64_t EagerBlobDesc::parallel_size() {
  if (parallel_size_ == 0) {
    std::shared_ptr<cfg::ParallelConf> cfg_parallel_conf = parallel_conf();
    ParallelConf proto_parallel_conf;
    cfg_parallel_conf->ToProto(&proto_parallel_conf);
    ParallelDesc parallel_desc(proto_parallel_conf);
    parallel_size_ = parallel_desc.parallel_num();
  }
  return parallel_size_;
}

std::shared_ptr<BlobObject> EagerBlobDesc::blob_object() const {
  return registered_blob_access_->blob_object();
}

void EagerBlobDesc::_Init(const std::string logical_blob_name,
                           const std::shared_ptr<BlobObject>& blob_object,
                           const std::shared_ptr<BlobRegister>& blob_register) {
  logical_blob_name_ = logical_blob_name;
  std::shared_ptr<RegisteredBlobAccess> access =
      blob_register->OpenRegisteredBlobAccess(logical_blob_name, blob_object);
  registered_blob_access_ = access;
}

bool EagerBlobDesc::IdenticalTo(const std::shared_ptr<EagerBlobDesc>& rhs) const {
  return (blob_object()->op_arg_blob_attr() == rhs->blob_object()->op_arg_blob_attr())
         && (blob_object()->op_arg_parallel_attr() == rhs->blob_object()->op_arg_parallel_attr());
}


// MirroredEagerBlobDesc 
MirroredEagerBlobDesc::MirroredEagerBlobDesc(
    const std::shared_ptr<cfg::LogicalBlobId>& lbi,
    const std::shared_ptr<compatible_py::BlobObject>& blob_object,
    const std::shared_ptr<compatible_py::BlobRegister>& blob_register, const std::string& job_name,
    const std::shared_ptr<compatible_py::Distribute>& distribute)
    : EagerBlobDesc(lbi, job_name, distribute) {
  std::string logical_blob_name = lbi->op_name() + "/" + lbi->blob_name();
  _Init(logical_blob_name, blob_object, blob_register);
}

//ConsistentEagerBlobDesc 
ConsistentEagerBlobDesc::ConsistentEagerBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                                 const std::shared_ptr<BlobObject>& blob_object,
                                                 const std::shared_ptr<BlobRegister>& blob_register,
                                                 const std::string& job_name,
                                                 const std::shared_ptr<Distribute>& distribute)
    : EagerBlobDesc(lbi, job_name, distribute) {
  std::string logical_blob_name = lbi->op_name() + "/" + lbi->blob_name();
  _Init(logical_blob_name, blob_object, blob_register);
}

// class MirroredLazyBlobDesc:public LazyBlobDesc
MirroredLazyBlobDesc::MirroredLazyBlobDesc(const std::shared_ptr<cfg::LogicalBlobId>& lbi,
                                           const std::string& job_name,
                                           const std::shared_ptr<Distribute>& distribute)
    :LazyBlobDesc(lbi, job_name, distribute){
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(this->job_name()));
  int lbi_num = CHECK_JUST(ctx->MirroredBlobGetNumSubLbi(this->logical_blob_name()));
  for (int i = 0; i < lbi_num; ++i) {
    std::shared_ptr<cfg::LogicalBlobId> sub_lbi = std::make_shared<cfg::LogicalBlobId>(
        *CHECK_JUST(ctx->MirroredBlobGetSubLbi(this->logical_blob_name(), i)));
    sub_consistent_blob_list_.emplace_back(
        std::make_shared<ConsistentLazyBlobDesc>(sub_lbi, "", GlobalAutoDistribute()));
  }
}
std::vector<std::shared_ptr<ConsistentLazyBlobDesc>> MirroredLazyBlobDesc::sub_consistent_blob_list() {
  return sub_consistent_blob_list_;
}

std::string MirroredLazyBlobDesc::get_mirror_shape_log_warning() const { return std::string(""); }

std::shared_ptr<Shape> MirroredLazyBlobDesc::shape() const {
  const std::string& log_warning = get_mirror_shape_log_warning();
  if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto shape = CHECK_JUST(ctx->MirroredBlobGetStaticShape(logical_blob_name()));
  return shape;
}

DataType MirroredLazyBlobDesc::dtype() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobGetDataType(logical_blob_name()));
}

int64_t MirroredLazyBlobDesc::batch_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetBatchAxis(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_BATCH_AXIS;
}

int64_t MirroredLazyBlobDesc::split_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->MirroredBlobGetSplitAxisFromProducerView(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_SPLIT_AXIS;
}

bool MirroredLazyBlobDesc::is_dynamic() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobIsDynamic(logical_blob_name()));
}

bool MirroredLazyBlobDesc::is_tensor_list() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobIsTensorList(logical_blob_name()));
}

std::shared_ptr<cfg::ParallelConf> MirroredLazyBlobDesc::parallel_conf() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->MirroredBlobGetParallelDescFromProducerView(logical_blob_name()))
      ->cfg_parallel_conf();
}

// ConsistentLazyBlobDesc
std::string ConsistentLazyBlobDesc::get_lazy_shape_log_warning() const { return std::string(""); }

std::shared_ptr<Shape> ConsistentLazyBlobDesc::shape() const {
  const std::string& log_warning = get_lazy_shape_log_warning();
  if (!log_warning.empty()) { LOG(ERROR) << log_warning; }
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetStaticShape(logical_blob_name()));
}

DataType ConsistentLazyBlobDesc::dtype() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetDataType(logical_blob_name()));
}

int64_t ConsistentLazyBlobDesc::batch_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->GetBatchAxis(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_BATCH_AXIS;
}

int64_t ConsistentLazyBlobDesc::split_axis() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  auto opt_int64 = CHECK_JUST(ctx->GetSplitAxisFromProducerView(logical_blob_name()));
  if (opt_int64->has_value()) { return opt_int64->value(); }
  return INVALID_SPLIT_AXIS;
}

bool ConsistentLazyBlobDesc::is_dynamic() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->IsDynamic(logical_blob_name()));
}

bool ConsistentLazyBlobDesc::is_tensor_list() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->IsTensorList(logical_blob_name()));
}

std::shared_ptr<cfg::ParallelConf> ConsistentLazyBlobDesc::parallel_conf() const {
  auto* ctx = CHECK_JUST(GetJobBuildAndInferCtx(job_name()));
  return CHECK_JUST(ctx->GetParallelDescFromProducerView(logical_blob_name()))->cfg_parallel_conf();
}

bool ConsistentLazyBlobDesc::IdenticalTo(const std::shared_ptr<ConsistentLazyBlobDesc>& rhs) const {
  return true && unique_name() == rhs->unique_name() && *shape() == *rhs->shape()
         && batch_axis() == rhs->batch_axis() && split_axis() == rhs->split_axis()
         && is_dynamic() == rhs->is_dynamic() && is_tensor_list() == rhs->is_tensor_list();
}

}  // namespace one

}  // namespace oneflow
