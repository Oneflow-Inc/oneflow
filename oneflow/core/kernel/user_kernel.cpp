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
    const auto& op_conf = kernel_conf().op_attribute().op_conf();
    const auto& user_kernel_conf = kernel_conf().user_conf();
    user_op::BlobInfo blob_info;
    auto BlobInfo4ArgNameAndIndex = [&](const std::string& arg_name,
                                        int32_t id) -> user_op::BlobInfo* {
      std::string bn_in_op = GenRepeatedBn(arg_name, id);
      const auto& pb_map = user_kernel_conf.bn_in_op2blob_desc();
      auto it = pb_map.find(bn_in_op);
      if (it == pb_map.end()) { return nullptr; }
      blob_info = it->second;
      return &blob_info;
    };
    user_op::KernelRegContext ctx(op_conf.device_type(), kernel_conf().data_type(),
                                  user_kernel_conf.parallel_ctx(), BlobInfo4ArgNameAndIndex);

    auto kernel_reg_val = user_op::LookUpInKernelRegistry(op_conf.user_conf().op_type_name(), ctx);
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
