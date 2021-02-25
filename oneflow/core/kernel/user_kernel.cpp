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
#include "oneflow/core/kernel/user_kernel.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/op_kernel_infer_cache.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/kernel/blob_tensor_view.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/kernel/eager_kernel.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_helper.h"

namespace oneflow {

using Arg2Tensor =
    HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::BlobTensorView>>;
using ArgVec = std::vector<std::pair<std::string, int32_t>>;

namespace {

void FillTensorDescWithBlob(const Blob* blob, user_op::TensorDesc* tensor_desc) {
  BlobDescProto proto;
  blob->blob_desc().header_pod_desc().ToProto(proto.mutable_header());
  blob->blob_desc().body().ToProto(proto.mutable_body());
  proto.set_is_tensor_list(blob->blob_desc().is_tensor_list());
  proto.set_is_dynamic(blob->blob_desc().is_dynamic());
  proto.set_header_is_opaque(blob->blob_desc().header_is_opaque());
  *tensor_desc = proto;
  tensor_desc->mut_shape()->CheckNumAxesIdenticalAndAssign(blob->shape());
}

}  // namespace

class UserKernelBaseContext {
 public:
  UserKernelBaseContext(const KernelConf& kernel_conf, const JobDesc& job_desc)
      : job_desc_(job_desc) {
    CHECK(kernel_conf.has_user_conf());
    CHECK(kernel_conf.op_attribute().op_conf().has_user_conf());

    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                           ArgVec* arg_vec) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg_vec->emplace_back(std::make_pair(it->first, i));
        }
      }
    };
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().input(), &inputs_);
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().output(), &outputs_);
    device_tag_ = kernel_conf.op_attribute().op_conf().device_tag();
    device_type_ = CHECK_JUST(DeviceType4DeviceTag(device_tag_));
    parallel_ctx_ = kernel_conf.user_conf().parallel_ctx();
    for (const auto& pair : kernel_conf.user_conf().bn_in_op2blob_desc()) {
      arg2tensor_desc_.emplace(GenUnRepeatedBn(pair.first), user_op::TensorDesc(pair.second));
    }
  }
  ~UserKernelBaseContext() = default;

  DeviceType device_type() const { return device_type_; }
  const std::string& device_tag() const { return device_tag_; }
  const ParallelContext& parallel_ctx() const { return parallel_ctx_; }
  const JobDesc& job_desc() const { return job_desc_; }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; }
    return &(it->second);
  }

  const ArgVec& inputs() const { return inputs_; }
  const ArgVec& outputs() const { return outputs_; }

 private:
  ArgVec inputs_;
  ArgVec outputs_;
  DeviceType device_type_;
  std::string device_tag_;
  ParallelContext parallel_ctx_;
  HashMap<std::pair<std::string, int32_t>, user_op::TensorDesc> arg2tensor_desc_;
  const JobDesc& job_desc_;
};

class KernelCreateContext final : public user_op::KernelCreateContext {
 public:
  explicit KernelCreateContext(const KernelConf& kernel_conf)
      : user_op_conf_(kernel_conf.op_attribute().op_conf()) {}

  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

 private:
  user_op::UserOpConfWrapper user_op_conf_;
};

class UserKernelInitContext final : public user_op::KernelInitContext {
 public:
  explicit UserKernelInitContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf,
                                 const JobDesc& job_desc)
      : user_op::KernelInitContext(
            user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        device_ctx_(device_ctx),
        base_ctx_(UserKernelBaseContext(kernel_conf, job_desc)),
        sbp_signature_(&(kernel_conf.user_conf().sbp_sig())),
        parallel_desc_(kernel_conf.user_conf().parallel_conf()) {
    for (const auto& pair : kernel_conf.user_conf().bn_in_op2logical_blob_desc()) {
      arg2logical_tensor_desc_.emplace(GenUnRepeatedBn(pair.first),
                                       user_op::TensorDesc(pair.second));
    }
  }
  ~UserKernelInitContext() override = default;

  DeviceCtx* device_ctx() override { return device_ctx_; }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    auto it = arg2logical_tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2logical_tensor_desc_.end()) {
      return nullptr;
    } else {
      return &(it->second);
    }
  }
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    const auto& bn2sbp = sbp_signature_->bn_in_op2sbp_parallel();
    std::string bn = GenRepeatedBn(arg_name, index);
    auto it = bn2sbp.find(bn);
    CHECK(it != bn2sbp.end());
    return it->second;
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }
  const ParallelDesc& parallel_desc() const override { return parallel_desc_; }

 private:
  DeviceCtx* device_ctx_;
  UserKernelBaseContext base_ctx_;
  const SbpSignature* sbp_signature_;
  HashMap<std::pair<std::string, int32_t>, user_op::TensorDesc> arg2logical_tensor_desc_;
  ParallelDesc parallel_desc_;
};

class UserKernelOpInferContext : public user_op::InferContext {
 public:
  UserKernelOpInferContext(const OperatorConf& op_conf, const JobDesc& job_desc)
      : user_op::InferContext(user_op::UserOpConfWrapper(op_conf)), job_desc_(job_desc) {
    auto* bn2sbp = sbp_signature_.mutable_bn_in_op2sbp_parallel();
    auto InitArgs7TensorDesc7Sbp = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                                       ArgVec* arg_vec) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          std::pair<std::string, int32_t> arg_pair = std::make_pair(arg_name, i);
          arg_vec->emplace_back(arg_pair);
          arg2tensor_desc_.emplace(arg_pair, nullptr);
          const std::string& bn_in_op = GenRepeatedBn(arg_name, i);
          (*bn2sbp)[bn_in_op].mutable_split_parallel()->set_axis(0);
        }
      }
    };
    InitArgs7TensorDesc7Sbp(op_conf.user_conf().input(), &inputs_);
    InitArgs7TensorDesc7Sbp(op_conf.user_conf().output(), &outputs_);
    parallel_ctx_.set_parallel_id(0);
    parallel_ctx_.set_parallel_num(1);
  }
  ~UserKernelOpInferContext() = default;

  user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                  int32_t index) override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; }
    return it->second.get();
  }
  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_shape();
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_data_type();
  }
  bool* IsDynamic4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_is_dynamic();
  }
  bool* IsTensorList4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return TensorDesc4ArgNameAndIndex(arg_name, index)->mut_is_tensor_list();
  }

  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }
  const JobDesc& job_desc() const override { return job_desc_; }
  const ParallelContext& parallel_ctx() const override { return parallel_ctx_; };
  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string& arg_name,
                                                 int32_t index) const override {
    const auto& bn2sbp = sbp_signature_.bn_in_op2sbp_parallel();
    std::string bn = GenRepeatedBn(arg_name, index);
    auto it = bn2sbp.find(bn);
    CHECK(it != bn2sbp.end());
    return it->second;
  }

  void UpdateArg2TensorDesc(const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
    for (auto& pair : arg2tensor_desc_) {
      const auto& arg_pair = pair.first;
      std::unique_ptr<user_op::TensorDesc>* arg_tensor_desc_ptr = &pair.second;
      Blob* blob = BnInOp2Blob(GenRepeatedBn(arg_pair.first, arg_pair.second));
      CHECK_NOTNULL(blob);
      if (*arg_tensor_desc_ptr) {
        (*arg_tensor_desc_ptr)->mut_shape()->CheckNumAxesIdenticalAndAssign(blob->shape());
      } else {
        arg_tensor_desc_ptr->reset(new user_op::TensorDesc());
        FillTensorDescWithBlob(blob, arg_tensor_desc_ptr->get());
      }
    }
  }

 private:
  const JobDesc& job_desc_;
  ArgVec inputs_;
  ArgVec outputs_;
  ParallelContext parallel_ctx_;
  SbpSignature sbp_signature_;
  HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::TensorDesc>> arg2tensor_desc_;
};

class UserKernelInferContext final : public user_op::KernelInferContext {
 public:
  explicit UserKernelInferContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf,
                                  const JobDesc& job_desc)
      : user_op::KernelInferContext(
            user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        device_ctx_(device_ctx),
        base_ctx_(UserKernelBaseContext(kernel_conf, job_desc)),
        op_infer_ctx_(kernel_conf.op_attribute().op_conf(), job_desc) {
    auto InitArg2Blob = [this](const PbMap<std::string, UserOpConf::ListString>& arg_map) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg2tensor_.emplace(std::make_pair(arg_name, i), nullptr);
        }
      }
    };
    InitArg2Blob(kernel_conf.op_attribute().op_conf().user_conf().input());
    InitArg2Blob(kernel_conf.op_attribute().op_conf().user_conf().output());

    const auto* op_reg_val = user_op::UserOpRegistryMgr::Get().GetOpRegistryResult(
        kernel_conf.op_attribute().op_conf().user_conf().op_type_name());
    CHECK_NOTNULL(op_reg_val);
    tensor_desc_infer_fn_ = op_reg_val->tensor_desc_infer_fn;
  }
  ~UserKernelInferContext() = default;

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

  DeviceCtx* device_ctx() override { return device_ctx_; }
  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) override {
    auto it = arg2tensor_.find(std::make_pair(arg_name, arg_index));
    CHECK(it != arg2tensor_.end()) << "Arg (" << arg_name << "," << arg_index << ") is not found";
    return it->second.get();
  }
  const ShapeView& ShapeView4ArgNameAndIndex(const std::string& arg_name,
                                             int32_t arg_index) override {
    user_op::Tensor* arg_tensor = Tensor4ArgNameAndIndex(arg_name, arg_index);
    CHECK(arg_tensor != nullptr) << "Tensor of arg (" << arg_name << "," << arg_index
                                 << ") is not found";
    return arg_tensor->shape();
  }
  MutShapeView* MutShapeView4ArgNameAndIndex(const std::string& arg_name,
                                             int32_t arg_index) override {
    user_op::Tensor* arg_tensor = Tensor4ArgNameAndIndex(arg_name, arg_index);
    CHECK(arg_tensor != nullptr) << "Tensor of arg (" << arg_name << "," << arg_index
                                 << ") is not found";
    return arg_tensor->mut_shape();
  }

  user_op::InferContext* MutOpInferContext() override { return &op_infer_ctx_; }
  const user_op::TensorDescInferFn& GetOpInferFn() const override { return tensor_desc_infer_fn_; }

  void UpdateArg2Tensor(const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
    for (auto& pair : arg2tensor_) {
      const auto& arg_pair = pair.first;
      std::unique_ptr<user_op::BlobTensorView>* arg_tensor_ptr = &pair.second;
      Blob* blob = BnInOp2Blob(GenRepeatedBn(arg_pair.first, arg_pair.second));
      if (blob == nullptr) { continue; }
      if (*arg_tensor_ptr) {
        arg_tensor_ptr->get()->Reset(blob);
      } else {
        arg_tensor_ptr->reset(new user_op::BlobTensorView(blob));
      }
    }
  }

 private:
  DeviceCtx* device_ctx_;
  UserKernelBaseContext base_ctx_;
  UserKernelOpInferContext op_infer_ctx_;
  user_op::TensorDescInferFn tensor_desc_infer_fn_;
  HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::BlobTensorView>> arg2tensor_;
};

namespace {

struct BnTensorPair {
  std::string bn;
  std::unique_ptr<user_op::BlobTensorView> tensor;
};

BnTensorPair MakeBnTensorPair(const std::string& bn) {
  BnTensorPair pair;
  pair.bn = bn;
  return pair;
}

BnTensorPair MakeBnTensorPair(const std::string& bn,
                              std::unique_ptr<user_op::BlobTensorView>&& tensor) {
  BnTensorPair pair;
  pair.bn = bn;
  pair.tensor = std::move(tensor);
  return pair;
}

}  // namespace

class UserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit UserKernelComputeContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf,
                                    const JobDesc& job_desc)
      : user_op::KernelComputeContext(
            user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        device_ctx_(device_ctx),
        base_ctx_(std::move(UserKernelBaseContext(kernel_conf, job_desc))) {
    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map) {
      for (const auto& it : arg_map) {
        const std::string& arg_name = it.first;
        for (int32_t i = 0; i < it.second.s_size(); ++i) {
          arg2bn_tensor_pair_.emplace(std::make_pair(arg_name, i),
                                      MakeBnTensorPair(GenRepeatedBn(arg_name, i)));
        }
      }
    };
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().input());
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().output());
    arg2bn_tensor_pair_.emplace(std::make_pair("tmp_buffer", 0),
                                MakeBnTensorPair(GenRepeatedBn("tmp_buffer", 0)));
  }
  ~UserKernelComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto it = arg2bn_tensor_pair_.find(std::make_pair(arg_name, index));
    if (it == arg2bn_tensor_pair_.end()) { return nullptr; }
    return it->second.tensor.get();
  }
  DeviceCtx* device_ctx() override { return device_ctx_; }

  void UpdateTensorWithCorrBlob(const std::function<Blob*(const std::string&)>& BnInOp2Blob) {
    for (auto& pair : arg2bn_tensor_pair_) {
      std::unique_ptr<user_op::BlobTensorView>* arg_tensor_ptr = &pair.second.tensor;
      Blob* blob = BnInOp2Blob(pair.second.bn);
      if (blob == nullptr) { continue; }
      if (*arg_tensor_ptr) {
        arg_tensor_ptr->get()->Reset(blob);
      } else {
        arg_tensor_ptr->reset(new user_op::BlobTensorView(blob));
      }
    }
  }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const JobDesc& job_desc() const override { return base_ctx_.job_desc(); }

  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

 private:
  DeviceCtx* device_ctx_;
  HashMap<std::pair<std::string, int32_t>, BnTensorPair> arg2bn_tensor_pair_;
  UserKernelBaseContext base_ctx_;
};

class UserKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit UserKernelRegContext(const KernelConf& kernel_conf, const JobDesc& job_desc)
      : user_op::KernelRegContext(user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        base_ctx_(UserKernelBaseContext(kernel_conf, job_desc)) {}
  ~UserKernelRegContext() = default;

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const std::string& device_tag() const override { return base_ctx_.device_tag(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

 private:
  UserKernelBaseContext base_ctx_;
};

void UserKernel::InitUserKernel(DeviceCtx* device_ctx) {
  ctx_.reset(new UserKernelComputeContext(device_ctx, kernel_conf(), job_desc()));
  infer_ctx_.reset(new UserKernelInferContext(device_ctx, kernel_conf(), job_desc()));
  infer_cache_.reset(new user_op::OpKernelInferCache(kernel_conf(), job_desc()));
  {
    const std::string& op_type_name =
        kernel_conf().op_attribute().op_conf().user_conf().op_type_name();
    const user_op::OpKernelRegistryResult* kernel_reg_val =
        CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
            op_type_name, UserKernelRegContext(kernel_conf(), job_desc())));
    CHECK_NOTNULL(kernel_reg_val);
    KernelCreateContext create_ctx(kernel_conf());
    kernel_.reset(kernel_reg_val->create_fn(&create_ctx));
  }
}

std::shared_ptr<user_op::OpKernelState> UserKernel::CreateOpKernelState(DeviceCtx* device_ctx) {
  UserKernelInitContext init_ctx(device_ctx, kernel_conf(), job_desc());
  return kernel_->CreateOpKernelState(&init_ctx);
}

const std::shared_ptr<user_op::OpKernelState>& UserKernel::GetOpKernelState() const {
  return opkernel_state_;
}

void UserKernel::ForwardUserKernel(std::function<Blob*(const std::string&)> BnInOp2Blob,
                                   user_op::OpKernelState* opkernel_state) const {
  ctx_->UpdateTensorWithCorrBlob(BnInOp2Blob);
  kernel_->Compute(ctx_.get(), opkernel_state);
}

void UserKernel::VirtualKernelInit(DeviceCtx* device_ctx) {
  InitUserKernel(device_ctx);
  CHECK(opkernel_state_.get() == nullptr);
  opkernel_state_ = CreateOpKernelState(device_ctx);
}

void UserKernel::ForwardDataContent(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ForwardUserKernel(BnInOp2Blob, opkernel_state_.get());
}

void UserKernel::ForwardShape(const KernelCtx& ctx,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  infer_ctx_->UpdateArg2Tensor(BnInOp2Blob);
  infer_cache_->UpdateCacheKey(infer_ctx_.get());
  if (!infer_cache_->IsCacheHit()) {
    auto* op_infer_ctx = dynamic_cast<UserKernelOpInferContext*>(infer_ctx_->MutOpInferContext());
    CHECK_NOTNULL(op_infer_ctx);
    op_infer_ctx->UpdateArg2TensorDesc(BnInOp2Blob);
    kernel_->InferShape(infer_ctx_.get());
    for (const auto& out_arg_pair : infer_ctx_->outputs()) {
      const Shape& static_shape =
          infer_ctx_->TensorDesc4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second)->shape();
      const ShapeView& shape_view =
          infer_ctx_->ShapeView4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      CHECK_LE(shape_view.elem_cnt(), static_shape.elem_cnt())
          << "InferShape of OpKernel (op_type_name: " << op_conf().user_conf().op_type_name()
          << ", op_name: " << op_conf().name()
          << ") raise error, output arg's (name: " << out_arg_pair.first
          << ", index: " << out_arg_pair.second << ") runtime shape " << shape_view.ToString()
          << " surpass the limit of static shape " << static_shape.ToString();
    }
    infer_cache_->UpdateCacheValue(infer_ctx_.get());
  } else {
    std::shared_ptr<const OpInferCacheValue> cache_value_ptr = infer_cache_->GetCacheValue();
    FOR_RANGE(int, i, 0, infer_ctx_->outputs().size()) {
      const auto& out_arg_pair = infer_ctx_->outputs().at(i);
      MutShapeView* mut_shape_view =
          infer_ctx_->MutShapeView4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      if (mut_shape_view) { mut_shape_view->set_shape(*cache_value_ptr->obn_idx2shape_sym.at(i)); }
    }
  }
}

bool UserKernel::IsStateless() const { return !kernel_->AlwaysComputeWhenAllOutputsEmpty(); }
NEW_REGISTER_KERNEL(OperatorConf::kUserConf, UserKernel).SetIsMatchedPred([](const KernelConf&) {
  return true;
});

EagerKernel::EagerKernel(const JobDesc* job_desc, const KernelConf& kernel_conf) {
  InitBase(job_desc, kernel_conf);
  InitOpKernel(kernel_conf);
}

void EagerKernel::InitOpKernel(const KernelConf& kernel_conf) {
  const std::string& op_type_name = kernel_conf.op_attribute().op_conf().user_conf().op_type_name();
  auto kernel_reg_val = CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
      op_type_name, UserKernelRegContext(kernel_conf, job_desc())));
  CHECK_NOTNULL(kernel_reg_val);
  KernelCreateContext create_ctx(kernel_conf);
  kernel_.reset(kernel_reg_val->create_fn(&create_ctx));
}

void EagerKernel::Infer(std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (!kernel_conf().need_do_shape()) { return; }
  UserKernelInferContext infer_ctx(nullptr, kernel_conf(), job_desc());
  infer_ctx.UpdateArg2Tensor(BnInOp2Blob);
  auto* op_infer_ctx = dynamic_cast<UserKernelOpInferContext*>(infer_ctx.MutOpInferContext());
  if (op_infer_ctx) { op_infer_ctx->UpdateArg2TensorDesc(BnInOp2Blob); }
  kernel_->InferShape(&infer_ctx);
}

std::shared_ptr<user_op::OpKernelState> EagerKernel::EagerForward(
    const std::shared_ptr<user_op::OpKernelState>& old_opkernel_state, DeviceCtx* device_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  std::shared_ptr<user_op::OpKernelState> new_opkernel_state;
  if (old_opkernel_state) {
    new_opkernel_state = old_opkernel_state;
  } else {
    CHECK_NOTNULL(&job_desc());
    UserKernelInitContext init_ctx(device_ctx, kernel_conf(), job_desc());
    new_opkernel_state = kernel_->CreateOpKernelState(&init_ctx);
  }

  if (IsAllBlobEmpty(op_attribute().output_bns(), BnInOp2Blob)
      && !kernel_->AlwaysComputeWhenAllOutputsEmpty()) {
    return new_opkernel_state;
  }

  // TODO(lixinqi): refactor to a lightweight KernelComputeContext
  UserKernelComputeContext compute_ctx(device_ctx, kernel_conf(), job_desc());
  compute_ctx.UpdateTensorWithCorrBlob(BnInOp2Blob);
  kernel_->Compute(&compute_ctx, new_opkernel_state.get());
  return new_opkernel_state;
}

}  // namespace oneflow
