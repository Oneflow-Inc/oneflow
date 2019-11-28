#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/kernel_registration.h"
#include "oneflow/core/framework/blob.h"

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
    auto kernel_reg_val = user_op::LookUpInKernelRegistry(
        kernel_conf().op_attribute().op_conf().user_conf().op_type_name(),
        user_op::KernelRegContext(kernel_conf()));
    CHECK_NOTNULL(kernel_reg_val);

    user_op::KernelInitContext init_ctx;
    kernel_.reset(kernel_reg_val->create_fn(init_ctx));
  }

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override {
    if (ctx_ == nullptr) {
      const auto& user_op_conf = kernel_conf().op_attribute().op_conf().user_conf();
      user_op::ArgNameAndIndex2Blob blobs;
      for (auto it = user_op_conf.input().begin(); it != user_op_conf.input().end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          Blob* blob = BnInOp2Blob(GenRepeatedBn(arg_name, i));
          blobs.emplace(std::make_pair(arg_name, i),
                        std::make_unique<user_op::Blob>(blob->shape(), blob->data_type(),
                                                        blob->mut_dptr<char>()));
        }
      }
      for (auto it = user_op_conf.output().begin(); it != user_op_conf.output().end(); ++it) {
        const std::string& arg_name = it->first;
        for (int32_t i = 0; i < it->second.s_size(); ++i) {
          Blob* blob = BnInOp2Blob(GenRepeatedBn(arg_name, i));
          blobs.emplace(std::make_pair(arg_name, i),
                        std::make_unique<user_op::Blob>(blob->shape(), blob->data_type(),
                                                        blob->mut_dptr<char>()));
        }
      }

      ctx_.reset(new user_op::KernelContext(ctx.device_ctx, std::move(blobs)));
    }
    kernel_->Compute(ctx_.get());
  }
};

NEW_REGISTER_KERNEL(OperatorConf::kUserConf, UserKernel).SetIsMatchedPred([](const KernelConf&) {
  return true;
});

}  // namespace oneflow
