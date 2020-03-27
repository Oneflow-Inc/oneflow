#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/kernel_context.h"

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

class UserKernelContext final : public user_op::KernelContext {
 public:
  explicit UserKernelContext(DeviceCtx* device_ctx, const KernelConf& kernel_conf)
      : user_op::KernelContext(user_op::UserOpConfWrapper(kernel_conf.op_attribute().op_conf())),
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
  ~UserKernelContext() = default;

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

 private:
  std::unique_ptr<user_op::OpKernel> kernel_;
  std::unique_ptr<UserKernelContext> ctx_;

  void VirtualKernelInit(DeviceCtx* device_ctx) override {
    ctx_.reset(new UserKernelContext(device_ctx, kernel_conf()));

    const std::string& op_type_name =
        kernel_conf().op_attribute().op_conf().user_conf().op_type_name();
    {
      auto kernel_reg_val =
          user_op::LookUpInKernelRegistry(op_type_name, UserKernelRegContext(kernel_conf()));
      CHECK_NOTNULL(kernel_reg_val);

      UserKernelInitContext init_ctx(device_ctx, kernel_conf());
      kernel_.reset(kernel_reg_val->create_fn(&init_ctx));
    }
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ctx_->UpdateTensorWithCorrBlob(BnInOp2Blob);
    kernel_->Compute(ctx_.get());
  }
};

NEW_REGISTER_KERNEL(OperatorConf::kUserConf, UserKernel).SetIsMatchedPred([](const KernelConf&) {
  return true;
});

}  // namespace oneflow
