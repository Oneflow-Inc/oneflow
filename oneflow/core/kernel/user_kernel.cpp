#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/kernel_context.h"

namespace oneflow {

using Arg2Tensor = HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::Tensor>>;

class UserKernelContext final : public user_op::KernelContext {
 public:
  explicit UserKernelContext(DeviceCtx* device_ctx, const OperatorConf& op_conf)
      : user_op::KernelContext(user_op::UserOpConfWrapper(op_conf)),
        device_ctx_(device_ctx),
        arg2tensor_() {
    const auto& user_op_conf = op_conf.user_conf();
    for (auto it = user_op_conf.input().begin(); it != user_op_conf.input().end(); ++it) {
      const std::string& arg_name = it->first;
      for (int32_t i = 0; i < it->second.s_size(); ++i) {
        arg2tensor_.emplace(std::make_pair(arg_name, i), std::unique_ptr<user_op::Tensor>());
      }
    }
    for (auto it = user_op_conf.output().begin(); it != user_op_conf.output().end(); ++it) {
      const std::string& arg_name = it->first;
      for (int32_t i = 0; i < it->second.s_size(); ++i) {
        arg2tensor_.emplace(std::make_pair(arg_name, i), std::unique_ptr<user_op::Tensor>());
      }
    }
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
        pair.second.reset(new user_op::Tensor(blob->shape(), blob->data_type(),
                                              static_cast<char*>(blob->mut_dptr())));
      }
    }
  }

 private:
  DeviceCtx* device_ctx_;
  Arg2Tensor arg2tensor_;
};

class UserKernelRegContext final : public user_op::KernelRegContext {
 public:
  explicit UserKernelRegContext(const KernelConf& kernel_conf,
                                user_op::UserOpConfWrapper&& user_op_conf)
      : user_op::KernelRegContext(std::move(user_op_conf)) {
    CHECK(kernel_conf.has_user_conf());

    device_ = kernel_conf.op_attribute().op_conf().device_type();
    data_type_ = kernel_conf.data_type();
    parallel_ctx_ = kernel_conf.user_conf().parallel_ctx();

    for (const auto& pair : kernel_conf.user_conf().bn_in_op2blob_desc()) {
      arg2tensor_desc_.emplace(GenUnRepeatedBn(pair.first), user_op::TensorDesc(pair.second));
    }
  }
  ~UserKernelRegContext() = default;

  DeviceType device() const override { return device_; }
  DataType data_type() const override { return data_type_; }
  const ParallelContext& parallel_ctx() const override { return parallel_ctx_; }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) { return nullptr; }
    return &(it->second);
  }

 private:
  DeviceType device_;
  DataType data_type_;
  ParallelContext parallel_ctx_;
  HashMap<std::pair<std::string, int32_t>, user_op::TensorDesc> arg2tensor_desc_;
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
    auto kernel_reg_val = user_op::LookUpInKernelRegistry(
        kernel_conf().op_attribute().op_conf().user_conf().op_type_name(),
        UserKernelRegContext(kernel_conf(),
                             user_op::UserOpConfWrapper(kernel_conf().op_attribute().op_conf())));
    CHECK_NOTNULL(kernel_reg_val);

    user_op::KernelInitContext init_ctx;
    kernel_.reset(kernel_reg_val->create_fn(init_ctx));
    ctx_.reset(new UserKernelContext(device_ctx, op_conf()));
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
