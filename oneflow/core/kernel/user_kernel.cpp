#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/op_registration.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/user_op_conf.h"
#include "oneflow/core/framework/kernel_context.h"
#include "oneflow/core/framework/infer_util.h"

namespace oneflow {

using Arg2Tensor = HashMap<std::pair<std::string, int32_t>, std::unique_ptr<user_op::Tensor>>;

class UserKernelContext final : public user_op::KernelContext {
 public:
  explicit UserKernelContext(DeviceCtx* device_ctx, const OperatorConf& op_conf)
      : user_op::KernelContext(user_op::UserOpConfWrapper(op_conf)),
        device_ctx_(device_ctx),
        arg2tensor_() {
    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg2tensor_.emplace(std::make_pair(arg_name, i), std::unique_ptr<user_op::Tensor>());
        }
      }
    };
    InitInOrOut(op_conf.user_conf().input());
    InitInOrOut(op_conf.user_conf().output());
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

class UserKernelInferContext : public user_op::InferContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  UserKernelInferContext(const OperatorConf& op_conf)
      : user_op::InferContext(user_op::UserOpConfWrapper(op_conf)) {
    auto InitInOrOut = [&](const PbMap<std::string, UserOpConf::ListString>& arg_map,
                           ArgVec* arg_vec) {
      for (auto it = arg_map.begin(); it != arg_map.end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          arg2shape_.emplace(std::make_pair(arg_name, i), std::unique_ptr<Shape>());
          arg_vec->emplace_back(std::make_pair(arg_name, i));
        }
      }
    };
    InitInOrOut(op_conf.user_conf().input(), &inputs_);
    InitInOrOut(op_conf.user_conf().output(), &outputs_);
  }
  ~UserKernelInferContext() = default;

  Shape* Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto it = arg2shape_.find(std::make_pair(arg_name, index));
    if (it == arg2shape_.end()) { return nullptr; }
    return it->second.get();
  }
  DataType* Dtype4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    return nullptr;
  }
  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }

  void UpdateShapeWithCorrBlob(std::function<Blob*(const std::string&)> BnInOp2Blob) {
    for (auto& pair : arg2shape_) {
      std::string bn_in_op = GenRepeatedBn(pair.first.first, pair.first.second);
      Blob* blob = BnInOp2Blob(bn_in_op);
      if (blob == nullptr) {
        pair.second.reset();
      } else {
        Shape* shape = new Shape();
        blob->shape().ToShape(shape);
        pair.second.reset(shape);
      }
    }
  }

 private:
  ArgVec inputs_;
  ArgVec outputs_;
  HashMap<std::pair<std::string, int32_t>, std::unique_ptr<Shape>> arg2shape_;
};

class UserKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserKernel);
  UserKernel() = default;
  ~UserKernel() = default;

 private:
  std::unique_ptr<user_op::OpKernel> kernel_;
  std::unique_ptr<UserKernelContext> ctx_;

  user_op::ShapeInferFn shape_infer_fn_;
  std::unique_ptr<UserKernelInferContext> infer_ctx_;

  void VirtualKernelInit(DeviceCtx* device_ctx) override {
    ctx_.reset(new UserKernelContext(device_ctx, op_conf()));

    const std::string& op_type_name =
        kernel_conf().op_attribute().op_conf().user_conf().op_type_name();
    {
      auto kernel_reg_val = user_op::LookUpInKernelRegistry(
          op_type_name,
          UserKernelRegContext(kernel_conf(),
                               user_op::UserOpConfWrapper(kernel_conf().op_attribute().op_conf())));
      CHECK_NOTNULL(kernel_reg_val);

      user_op::KernelInitContext init_ctx;
      kernel_.reset(kernel_reg_val->create_fn(init_ctx));
    }

    {
      const user_op::OpRegistrationVal* val = user_op::LookUpInOpRegistry(op_type_name);
      CHECK(val != nullptr) << "cannot find op_type: " << op_type_name << " in op registry!";
      shape_infer_fn_ = val->shape_infer_fn;
      infer_ctx_.reset(new UserKernelInferContext(op_conf()));
    }
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    ctx_->UpdateTensorWithCorrBlob(BnInOp2Blob);
    kernel_->Compute(ctx_.get());
  }
  void ForwardShape(const KernelCtx& ctx,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    infer_ctx_->UpdateShapeWithCorrBlob(BnInOp2Blob);
    shape_infer_fn_(infer_ctx_.get());
    for (const auto& pair : infer_ctx_->outputs()) {
      Blob* blob = BnInOp2Blob(GenRepeatedBn(pair.first, pair.second));
      blob->mut_shape_view()->set_shape(
          *infer_ctx_->Shape4ArgNameAndIndex(pair.first, pair.second));
    }
  }
};

NEW_REGISTER_KERNEL(OperatorConf::kUserConf, UserKernel).SetIsMatchedPred([](const KernelConf&) {
  return true;
});

}  // namespace oneflow
