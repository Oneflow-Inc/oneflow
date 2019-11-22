#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/blob_info.h"

namespace oneflow {

class UserKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UserKernel);
  UserKernel() = default;
  ~UserKernel() = default;

 private:
  std::unique_ptr<user_op::OpKernel> kernel_;
  mutable std::unique_ptr<user_op::KernelContext> ctx_;

  void VirtualKernelInit() override {
    user_op::KernelRegContext ctx(kernel_conf());

    auto kernel_reg_val = user_op::LookUpInKernelRegistry(
        kernel_conf().op_attribute().op_conf().user_conf().op_type_name(), ctx);
    CHECK_NOTNULL(kernel_reg_val);

    user_op::KernelInitContext init_ctx;
    kernel_.reset(kernel_reg_val->create_fn(init_ctx));
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    if (ctx_ == nullptr) {
      auto Blob4ArgNameAndIndex = [&](const std::string& arg_name, int32_t id) {
        std::string bn_in_op = GenRepeatedBn(arg_name, id);
        return BnInOp2Blob(bn_in_op);
      };
      ctx_.reset(new user_op::KernelContext(ctx, Blob4ArgNameAndIndex));
    }
    kernel_->Compute(*ctx_.get());
  }
};

NEW_REGISTER_KERNEL(OperatorConf::kUserConf, UserKernel);

}  // namespace oneflow
