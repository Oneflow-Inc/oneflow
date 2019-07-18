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

template<typename KeyType, OperatorConf::OpTypeCase op_type>
HashMap<KeyType, CreateFn>& KernelRegistry() {
  static HashMap<KeyType, CreateFn> creator_fns;
}

template<typename KeyType, typename DerivedKernel>
class KernelRegistrar {
 public:
  KernelRegistrar(const OperatorConf::OpTypeCase& op_type_case, const KeyType& key, CreateFn fn) {
    HashMap<KeyType, CreateFn>& creator_fns = KernelRegistry<KeyType, op_type_case>();
    creator_fns.insert(std::make_pair(key, []() { return new DerivedKernel; }));
  }
};

template<typename HelperClass>
struct Assurer4RunOnlyOnce {
  Assurer4RunOnlyOnce() { static HelperClass x; }
};

template<typename DerivedKernelT>
class RegisterHelper4DeviceAndDType {
 public:
  static Kernel* CreateKernel(const KernelConf& kernel_conf) {
    const auto& registry = KernelRegistrar<std::string, DerivedKernelT>();
    return registry.at(
        GetHashKey(kernel_conf.op_attribute().op_conf().device_type(), kernel_conf.data_type()))();
  }
  RegisterHelper4DeviceAndDType(const OperatorConf::OpTypeCase& op_type_case) {
    REGISTER_CLASS_CREATOR(op_type_case, Kernel, RegisterHelper4DeviceAndDType::CreateKernel,
                           const KernelConf&);
  }
};

#define REGISTER_KERNEL_BASE(op_type_case, DerivedKernelT, KeyT, RegisterHelperT, key)            \
  namespace {                                                                                     \
  static Assurer4RunOnlyOnce<RegisterHelperT> g_assurer##__COUNTER__;                             \
  static KernelRegistrar<KeyT, DerivedKernelT> g_registrar##__COUNTER__(op_type_case, key, []() { \
    return DerivedKernelT;                                                                        \
  });                                                                                             \
  }  // namespace

#define REGISTER_KERNEL_WITH_DEVICE_AND_DType(op_type_case, DerivedKernelT, device_type, dtype) \
  REGISTER_KERNEL_BASE(op_type_case, DerivedKernelT, std::string,                               \
                       RegisterHelper4DeviceAndDType<DerivedKernelT>,                           \
                       GetHashKey(device_type, dtype))

}  // namespace kernel_registration

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_REGISTRAR_H_
