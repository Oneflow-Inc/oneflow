#ifndef ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_
#define ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_

#include "oneflow/core/framework/registrar.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class KernelConf;

namespace user_op {

class OpKernel;
class KernelInitContext;
class BlobDef;
class InferContext;

using BlobDef4ArgNameAndIndexFn =
    std::function<std::shared_ptr<BlobDef>(const std::string&, int32_t)>;

class KernelRegContext final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelRegContext);
  explicit KernelRegContext(DeviceType, DataType, const ParallelContext&,
                            BlobDef4ArgNameAndIndexFn);
  explicit KernelRegContext(const KernelConf&);
  ~KernelRegContext() = default;

  DeviceType device() const { return device_; }
  DataType data_type() const { return data_type_; }
  const ParallelContext& parallel_ctx() const { return parallel_ctx_; }
  std::shared_ptr<const BlobDef> BlobDesc4ArgNameAndIndex(const std::string&, int32_t) const;

 private:
  DeviceType device_;
  DataType data_type_;
  ParallelContext parallel_ctx_;
  BlobDef4ArgNameAndIndexFn fn_;
};

using CreateFn = std::function<OpKernel*(const KernelInitContext&)>;
using IsMatchedPredicator = std::function<bool(const KernelRegContext&)>;
using InferTmpSizeFn = std::function<size_t(const InferContext&)>;

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
                                                    const KernelRegContext&);

std::vector<std::string> GetAllUserOpInKernelRegistry();

}  // namespace user_op

}  // namespace oneflow

#define REGISTER_USER_KERNEL(name)                                                       \
  static ::oneflow::user_op::Registrar<::oneflow::user_op::KernelRegistryWrapperBuilder> \
      OF_PP_CAT(g_registrar, __COUNTER__) = ::oneflow::user_op::KernelRegistryWrapperBuilder(name)

#endif  // ONEFLOW_CORE_FRAMEWORK_KERNEL_REGISTRATION_H_
