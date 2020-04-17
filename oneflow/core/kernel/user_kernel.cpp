#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/operator/op_infer_cache.h"
#include "oneflow/core/common/cached_caller.h"

namespace oneflow {

using Arg2Tensor = HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::Tensor>>;
using ArgVec = std::vector<std::pair<std::string, int32_t>>;

class UserKernelBaseContext {
 public:
  UserKernelBaseContext(const KernelConf& kernel_conf) {
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

    device_type_ = kernel_conf.op_attribute().op_conf().device_type();
    parallel_ctx_ = kernel_conf.user_conf().parallel_ctx();
    for (const auto& pair : kernel_conf.user_conf().bn_in_op2blob_desc()) {
      arg2tensor_desc_.emplace(GenUnRepeatedBn(pair.first), user_op::TensorDesc(pair.second));
    }
  }
  ~UserKernelBaseContext() = default;

  DeviceType device_type() const { return device_type_; }
  const ParallelContext& parallel_ctx() const { return parallel_ctx_; }
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
  ParallelContext parallel_ctx_;
  HashMap<std::pair<std::string, int32_t>, user_op::TensorDesc> arg2tensor_desc_;
};

class UserKernelInitContext final : public user_op::KernelInitContext {
 public:
  explicit UserKernelInitContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf)
      : user_op::KernelInitContext(
            user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        device_ctx_(device_ctx),
        base_ctx_(UserKernelBaseContext(kernel_conf)) {}
  ~UserKernelInitContext() = default;

  DeviceCtx* device_ctx() override { return device_ctx_; }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

 private:
  DeviceCtx* device_ctx_;
  UserKernelBaseContext base_ctx_;
};

class UserKernelShapeInferContext final : public user_op::KernelShapeInferContext {
 public:
  explicit UserKernelShapeInferContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf,
                                       const JobDesc& job_desc)
      : user_op::KernelShapeInferContext(
            user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        base_ctx_(UserKernelBaseContext(kernel_conf)),
        device_ctx_(device_ctx),
        kernel_conf_(kernel_conf),
        job_desc_(job_desc) {
    auto InitArg2Blob = [this](const PbMap<std::string, UserOpConf::ListString>& arg_map) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg2blob_.emplace(std::make_pair(arg_name, i), nullptr);
        }
      }
    };
    InitArg2Blob(kernel_conf.op_attribute().op_conf().user_conf().input());
    InitArg2Blob(kernel_conf.op_attribute().op_conf().user_conf().output());
  }
  ~UserKernelShapeInferContext() = default;

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return base_ctx_.TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

  DeviceCtx* device_ctx() override { return device_ctx_; }

  ShapeView ShapeView4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) {
    auto it = arg2blob_.find(std::make_pair(arg_name, arg_index));
    CHECK(it != arg2blob_.end()) << "arg (" << arg_name << "," << arg_index << ") not found";
    return it->second->shape_view();
  }

  MutShapeView MutShapeView4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) {
    auto it = arg2blob_.find(std::make_pair(arg_name, arg_index));
    CHECK(it != arg2blob_.end()) << "arg (" << arg_name << "," << arg_index << ") not found";
    return *it->second->mut_shape_view();
  }

  void UpdateArg2Blob(std::function<Blob*(const std::string&)> BnInOp2Blob) {
    for (auto& pair : arg2blob_) {
      std::string bn_in_op = GenRepeatedBn(pair.first.first, pair.first.second);
      pair.second = BnInOp2Blob(bn_in_op);
    }
  }

 private:
  friend class UserKernelShapeInferCacheContext;
  UserKernelBaseContext base_ctx_;
  HashMap<std::pair<std::string, int32_t>, Blob*> arg2blob_;
  DeviceCtx* device_ctx_;
  const KernelConf& kernel_conf_;
  const JobDesc& job_desc_;
};

class UserKernelShapeInferCacheContext final : public user_op::KernelShapeInferContext {
 public:
  explicit UserKernelShapeInferCacheContext(UserKernelShapeInferContext* ctx)
      : user_op::KernelShapeInferContext(
            user_op::UserOpConfWrapper(ctx->kernel_conf_.op_attribute().op_conf())),
        ctx_(ctx) {
    op_ = ConstructOp(ctx->kernel_conf_.op_attribute().op_conf(), &ctx->job_desc_);
    auto* map = sbp_signature_.mutable_bn_in_op2sbp_parallel();
    op_->ForEachBnInOp([&](const std::string& bn_in_op) {
      bn2blob_desc_[bn_in_op].reset();
      (*map)[bn_in_op].mutable_split_parallel()->set_axis(0);
    });
    parallel_ctx_.set_parallel_id(0);
    parallel_ctx_.set_parallel_num(1);
    cache_key_.job_desc = &ctx->job_desc_;
    cache_key_.op_conf_sym = op_->GetOpConfWithoutOpNameAndLbn();
    cache_key_.ibn_idx2shape_sym.resize(inputs().size());
    cache_key_.dtype_signature_sym = SymbolOf(ctx->kernel_conf_.dtype_signature());
  }

  ~UserKernelShapeInferCacheContext() = default;

  DeviceType device_type() const override { return ctx_->device_type(); }
  const ParallelContext& parallel_ctx() const override { return ctx_->parallel_ctx(); }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ArgVec& inputs() const override { return ctx_->inputs(); }
  const ArgVec& outputs() const override { return ctx_->outputs(); }

  DeviceCtx* device_ctx() override { return ctx_->device_ctx_; }

  ShapeView ShapeView4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) {
    auto it = bn2blob_desc_.find(GenRepeatedBn(arg_name, arg_index));
    CHECK(it != bn2blob_desc_.end()) << "arg (" << arg_name << "," << arg_index << ") not found";
    return ShapeView(it->second->shape());
  }

  MutShapeView MutShapeView4ArgNameAndIndex(const std::string& arg_name, int32_t arg_index) {
    auto it = bn2blob_desc_.find(GenRepeatedBn(arg_name, arg_index));
    CHECK(it != bn2blob_desc_.end()) << "arg (" << arg_name << "," << arg_index << ") not found";
    return MutShapeView(it->second->mut_shape().dim_vec().data(), it->second->shape().NumAxes());
  }

  void InferShape() override {
    auto CachedBlobDesc4BnInOp = WithResultCached([&](const std::string& bn_in_op) -> BlobDesc* {
      const Blob* blob = Blob4BnInOp(bn_in_op);
      if (blob == nullptr) { return nullptr; }
      return BlobDesc4BnInOp(bn_in_op, blob->blob_desc());
    });
    CHECK_JUST(op_->InferOutBlobDescsIf(CachedBlobDesc4BnInOp, &parallel_ctx_, &sbp_signature_,
                                        [](OpContext*) {}));
  }

  void ForwardShape(
      std::function<void(user_op::KernelShapeInferContext*)> shape_infer_fn) override {
    UpdateInputBlobDescs7OpInferCacheKey();
    auto Infer = [&](const OpInferCacheKey& key) -> std::shared_ptr<const OpInferCacheValue> {
      shape_infer_fn(this);
      auto* cache_value = new OpInferCacheValue();
      cache_value->obn_idx2shape_sym.resize(outputs().size());
      FOR_RANGE(int, i, 0, outputs().size()) {
        const auto& out_arg_pair = outputs().at(i);
        const std::string& obn = GenRepeatedBn(out_arg_pair.first, out_arg_pair.second);
        const auto& blob_desc = bn2blob_desc_.at(obn);
        cache_value->obn_idx2shape_sym.at(i).reset(blob_desc->shape());
        auto* blob = Blob4BnInOp(obn);
        if (blob == nullptr) { continue; }
        CHECK_EQ(blob->data_type(), blob_desc->data_type());
        CHECK_EQ(blob->blob_desc().is_dynamic(), blob_desc->is_dynamic());
        CHECK_EQ(blob->blob_desc().is_body_disabled(), blob_desc->is_body_disabled());
      }
      return std::shared_ptr<const OpInferCacheValue>(cache_value);
    };
    size_t cache_size = Global<ResourceDesc>::Get()->thread_local_cache_max_size();
    auto cache_value_ptr = ThreadLocalCachedCall(cache_size, Infer, cache_key_);
    FOR_RANGE(int, i, 0, outputs().size()) {
      const auto& out_arg_pair = outputs().at(i);
      auto mut_shape_view =
          ctx_->MutShapeView4ArgNameAndIndex(out_arg_pair.first, out_arg_pair.second);
      mut_shape_view.set_shape(*cache_value_ptr->obn_idx2shape_sym.at(i));
    }
  }

 private:
  Blob* Blob4BnInOp(const std::string& bn_in_op) {
    auto arg_pair = GenUnRepeatedBn(bn_in_op);
    auto it = ctx_->arg2blob_.find(arg_pair);
    if (it == ctx_->arg2blob_.end()) { return nullptr; }
    return it->second;
  }

  BlobDesc* BlobDesc4BnInOp(const std::string& bn_in_op, const RtBlobDesc& rt_blob_desc) {
    BlobDesc* blob_desc = bn2blob_desc_.at(bn_in_op).get();
    if (blob_desc != nullptr) { return blob_desc; }
    blob_desc = new BlobDesc(rt_blob_desc.body(), rt_blob_desc.is_tensor_list(),
                             rt_blob_desc.is_body_disabled(), rt_blob_desc.is_dynamic());
    bn2blob_desc_.at(bn_in_op).reset(blob_desc);
    return blob_desc;
  }

  void UpdateInputBlobDescs7OpInferCacheKey() {
    auto ResetBlobDescAndGetShapeSym = [&](const std::string& ibn) -> Symbol<Shape> {
      const Blob* blob = Blob4BnInOp(ibn);
      if (blob == nullptr) { return Symbol<Shape>(); }
      auto* blob_desc = BlobDesc4BnInOp(ibn, blob->blob_desc());
      blob_desc->mut_shape().LeftOnesExtendedAssign(blob->shape());
      return SymbolOf(blob_desc->shape());
    };
    FOR_RANGE(int, i, 0, inputs().size()) {
      const auto& in_arg_pair = inputs().at(i);
      cache_key_.ibn_idx2shape_sym.at(i) =
          ResetBlobDescAndGetShapeSym(GenRepeatedBn(in_arg_pair.first, in_arg_pair.second));
    }
  }

  UserKernelShapeInferContext* ctx_;
  std::shared_ptr<Operator> op_;
  HashMap<std::string, std::unique_ptr<BlobDesc>> bn2blob_desc_;
  ParallelContext parallel_ctx_;
  SbpSignature sbp_signature_;
  OpInferCacheKey cache_key_;
};

class UserKernelComputeContext final : public user_op::KernelComputeContext {
 public:
  explicit UserKernelComputeContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf)
      : user_op::KernelComputeContext(
            user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        device_ctx_(device_ctx),
        base_ctx_(std::move(UserKernelBaseContext(kernel_conf))) {
    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg2tensor_.emplace(std::make_pair(arg_name, i), std::unique_ptr<user_op::Tensor>());
        }
      }
    };
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().input());
    InitInOrOut(kernel_conf.op_attribute().op_conf().user_conf().output());
    arg2tensor_.emplace(std::make_pair("tmp_buffer", 0), std::unique_ptr<user_op::Tensor>());
  }
  ~UserKernelComputeContext() = default;

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto it = arg2tensor_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_.end()) { return nullptr; }
    return it->second.get();
  }
  DeviceCtx* device_ctx() override { return device_ctx_; }

  void UpdateTensorWithCorrBlob(std::function<Blob*(const std::string&)> BnInOp2Blob) {
    for (auto& pair : arg2tensor_) {
      std::string bn_in_op = GenRepeatedBn(pair.first.first, pair.first.second);
      Blob* blob = BnInOp2Blob(bn_in_op);
      if (blob == nullptr) {
        pair.second.reset();
      } else {
        pair.second.reset(new user_op::Tensor(blob));
      }
    }
  }

  DeviceType device_type() const override { return base_ctx_.device_type(); }
  const ParallelContext& parallel_ctx() const override { return base_ctx_.parallel_ctx(); }

  const ArgVec& inputs() const override { return base_ctx_.inputs(); }
  const ArgVec& outputs() const override { return base_ctx_.outputs(); }

 private:
  DeviceCtx* device_ctx_;
  Arg2Tensor arg2tensor_;
  UserKernelBaseContext base_ctx_;
};

class UserKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit UserKernelRegContext(const KernelConf& kernel_conf)
      : user_op::KernelRegContext(user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
        base_ctx_(UserKernelBaseContext(kernel_conf)) {}
  ~UserKernelRegContext() = default;

  DeviceType device_type() const override { return base_ctx_.device_type(); }
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

class UserKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserKernel);
  UserKernel() = default;
  ~UserKernel() = default;

  void InitUserKernel(DeviceCtx* device_ctx) {
    ctx_.reset(new UserKernelComputeContext(device_ctx, kernel_conf()));
    shape_infer_ctx_.reset(new UserKernelShapeInferContext(device_ctx, kernel_conf(), job_desc()));
    shape_infer_cache_ctx_.reset(new UserKernelShapeInferCacheContext(shape_infer_ctx_.get()));
    {
      const std::string& op_type_name =
          kernel_conf().op_attribute().op_conf().user_conf().op_type_name();
      auto kernel_reg_val =
          user_op::LookUpInKernelRegistry(op_type_name, UserKernelRegContext(kernel_conf()));
      CHECK_NOTNULL(kernel_reg_val);
      kernel_.reset(kernel_reg_val->create_fn());
    }
  }
  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(DeviceCtx* device_ctx) {
    UserKernelInitContext init_ctx(device_ctx, kernel_conf());
    return kernel_->CreateOpKernelState(&init_ctx);
  }
  void ForwardUserKernel(std::function<Blob*(const std::string&)> BnInOp2Blob,
                         user_op::OpKernelState* opkernel_state) const {
    ctx_->UpdateTensorWithCorrBlob(BnInOp2Blob);
    kernel_->Compute(ctx_.get(), opkernel_state);
  }

 private:
  void VirtualKernelInit(DeviceCtx* device_ctx) override {
    InitUserKernel(device_ctx);
    CHECK(opkernel_state_.get() == nullptr);
    opkernel_state_ = CreateOpKernelState(device_ctx);
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ForwardUserKernel(BnInOp2Blob, opkernel_state_.get());
  }

  void ForwardShape(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    shape_infer_ctx_->UpdateArg2Blob(BnInOp2Blob);
    kernel_->ForwardShape(shape_infer_cache_ctx_.get());
  }

  std::shared_ptr<user_op::OpKernelState> opkernel_state_;
  std::unique_ptr<const user_op::OpKernel> kernel_;
  std::unique_ptr<UserKernelComputeContext> ctx_;
  std::unique_ptr<UserKernelShapeInferContext> shape_infer_ctx_;
  std::unique_ptr<UserKernelShapeInferCacheContext> shape_infer_cache_ctx_;
};

NEW_REGISTER_KERNEL(OperatorConf::kUserConf, UserKernel).SetIsMatchedPred([](const KernelConf&) {
  return true;
});

}  // namespace oneflow
