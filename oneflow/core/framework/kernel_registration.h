#ifndef ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_
#define ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_

#include "oneflow/core/framework/registrar.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/job/placement.pb.h"

namespace oneflow {

class OpKernel;
class BlobInfo;

namespace user_op {

class KernelRegCtx final {
 public:
  explicit KernelRegCtx();
  DeviceType device() const { return device_; }
  DataType data_type() const { return data_type_; }
  const BlobInfo& BlobDesc4ArgNameAndIndex(const std::string& arg_name, int32_t index) const {
    return blob_desc4arg_name_and_index_fn_(arg_name, index);
  }
  const ParallelContext& parallel_ctx() const { return parallel_ctx_; }

 private:
  DeviceType device_;
  DataType data_type_;
  ParallelContext parallel_ctx_;
  std::function<const BlobInfo&(const std::string&, int32_t)> blob_desc4arg_name_and_index_fn_;
};

using CreateFn = std::function<OpKernel*(/*TODO(niuchong)*/)>;
using IsMatchedPredicator = std::function<bool(const KernelRegCtx&)>;
using InferTmpSizeFn = std::function<size_t(/*TODO(niuchong)*/)>;

struct KernelRegistrationVal {
  CreateFn create_fn;
  IsMatchedPredicator is_matched_fn;
  InferTmpSizeFn infer_tmp_size_fn;
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

  KernelRegistryWrapper Build();

 private:
  KernelRegistryWrapper wrapper_;
};

const KernelRegistrationVal* LookUpInKernelRegistry(const std::string& op_type_name,
                                                    const KernelRegCtx&);

std::vector<std::string> GetAllUserOpInKernelRegistry();

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_KERNEL(name)                                                       \
  static ::oneflow::user_op::Registrar<::oneflow::user_op::KernelRegistryWrapperBuilder> \
      OF_PP_CAT(g_registrar, __COUNTER__) = ::oneflow::user_op::KernelRegistryWrapperBuilder(name)

#endif  // ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_
