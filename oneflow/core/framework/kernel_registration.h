#ifndef ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_
#define ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_

#include "oneflow/core/framework/registrar.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/user_op_conf.h"

namespace oneflow {

class KernelConf;

namespace user_op {

class OpKernel;
class KernelInitContext;
class TensorDesc;
class InferContext;

class KernelRegContext {
 public:
  virtual ~KernelRegContext() = default;

  virtual DeviceType device() const = 0;
  virtual const ParallelContext& parallel_ctx() const = 0;
  virtual const TensorDesc* TensorDesc4ArgNameAndIndex(const std::string&, int32_t) const = 0;

  template<typename T>
  T GetAttr(const std::string& attr_name) const {
    return user_op_conf_.attr<T>(attr_name);
  }

 protected:
  KernelRegContext(UserOpConfWrapper&& conf) : user_op_conf_(std::move(conf)) {}
  KernelRegContext(const KernelRegContext&) = delete;

 private:
  UserOpConfWrapper user_op_conf_;
};

using CreateFn = std::function<OpKernel*(const KernelInitContext&)>;
using IsMatchedPredicator = std::function<bool(const KernelRegContext&)>;
using InferTmpSizeFn = std::function<size_t(InferContext*)>;
using AddInplaceArgPair = std::function<Maybe<void>(
    const std::string& out_arg_name, int32_t out_arg_index, const std::string& in_arg_name,
    int32_t in_arg_index, bool is_mutable)>;
using InplaceProposalFn = std::function<Maybe<void>(const InferContext&, AddInplaceArgPair)>;

struct KernelRegistrationVal {
  CreateFn create_fn;
  IsMatchedPredicator is_matched_fn;
  InferTmpSizeFn infer_tmp_size_fn;
  InplaceProposalFn inplace_proposal_fn;
};

struct KernelRegistryWrapper final {
  void InsertToGlobalRegistry();

  std::string op_type_name;
  KernelRegistrationVal reg_val;
};

class KernelRegistryWrapperBuilder final {
 public:
  KernelRegistryWrapperBuilder(const std::string& op_type_name);
  KernelRegistryWrapperBuilder& SetCreateFn(CreateFn fn);
  KernelRegistryWrapperBuilder& SetIsMatchedPred(IsMatchedPredicator fn);
  KernelRegistryWrapperBuilder& SetInferTmpSizeFn(InferTmpSizeFn fn);
  KernelRegistryWrapperBuilder& SetInplaceProposalFn(InplaceProposalFn fn);

  KernelRegistryWrapper Build();

 private:
  KernelRegistryWrapper wrapper_;
};

const KernelRegistrationVal* LookUpInKernelRegistry(const std::string& op_type_name,
                                                    const KernelRegContext&);

std::vector<std::string> GetAllUserOpInKernelRegistry();

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_KERNEL(name)                                                       \
  static ::oneflow::user_op::Registrar<::oneflow::user_op::KernelRegistryWrapperBuilder> \
      OF_PP_CAT(g_registrar, __COUNTER__) = ::oneflow::user_op::KernelRegistryWrapperBuilder(name)

#endif  // ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_
