#ifndef ONEFLOW_CORE_KERNEL_KERNEL_REGISTRAR_H_
#define ONEFLOW_CORE_KERNEL_KERNEL_REGISTRAR_H_

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/kernel/kernel.pb.h"

namespace oneflow {

class Kernel;
class KernelConf;

namespace kernel_registration {

using CreateFn = std::function<Kernel*()>;

template<OperatorConf::OpTypeCase op_type, typename KeyT>
HashMap<KeyT, CreateFn>& KernelRegistry() {
  static HashMap<KeyT, CreateFn> creator_fns;
  return creator_fns;
}

template<OperatorConf::OpTypeCase op_type, typename KeyT>
class KernelRegistrar {
 public:
  KernelRegistrar(const KeyT& key, CreateFn fn) {
    HashMap<KeyT, CreateFn>& creator_fns = KernelRegistry<op_type, KeyT>();
    creator_fns.insert(std::make_pair(key, fn));
  }
};

namespace helper {

template<typename HelperClass, typename... Args>
struct Assurer4RunOnlyOnce {
  Assurer4RunOnlyOnce(Args&&... args) { static HelperClass x(std::forward<Args>(args)...); }
};

template<OperatorConf::OpTypeCase op_type>
class RegisterHelper4DeviceAndDType {
 public:
  static Kernel* CreateKernel(const KernelConf& kernel_conf) {
    const auto& registry = KernelRegistry<op_type, std::string>();
    return registry.at(
        GetHashKey(kernel_conf.op_attribute().op_conf().device_type(), kernel_conf.data_type()))();
  }
  RegisterHelper4DeviceAndDType() {
    REGISTER_CLASS_CREATOR(op_type, Kernel, RegisterHelper4DeviceAndDType::CreateKernel,
                           const KernelConf&);
  }
};

}  // namespace helper

}  // namespace kernel_registration

#define REGISTER_KERNEL_BASE(op_type_case, KeyT, RegisterHelperT, key, ...)                       \
  namespace {                                                                                     \
  static kernel_registration::helper::Assurer4RunOnlyOnce<RegisterHelperT<op_type_case>>          \
      OF_PP_CAT(g_assurer, __LINE__);                                                             \
  static kernel_registration::KernelRegistrar<op_type_case, KeyT> OF_PP_CAT(g_registrar,          \
                                                                            __LINE__)(key, []() { \
    return new __VA_ARGS__();                                                                     \
  });                                                                                             \
  }  // namespace

#define REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type_case, device_type, dtype, ...) \
  REGISTER_KERNEL_BASE(op_type_case, std::string,                                    \
                       kernel_registration::helper::RegisterHelper4DeviceAndDType,   \
                       GetHashKey(device_type, GetDataType<dtype>::value), __VA_ARGS__)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_REGISTRAR_H_
