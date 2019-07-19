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
class KernelRegistrar final {
 public:
  KernelRegistrar(const KeyT& key, CreateFn fn) {
    HashMap<KeyT, CreateFn>& creator_fns = KernelRegistry<op_type, KeyT>();
    creator_fns.insert(std::make_pair(key, fn));
  }
};

namespace builder {

class RegKeyBuilder {
 public:
  RegKeyBuilder() = default;
  virtual ~RegKeyBuilder() = default;

  std::string Build() const;

  RegKeyBuilder& Device(DeviceType device_type);

  template<typename DType>
  RegKeyBuilder& Type() {
    return Type(GetDataType<DType>::value);
  }

  RegKeyBuilder& Type(DataType dtype);

 private:
  std::unique_ptr<DeviceType> device_;
  std::unique_ptr<DataType> dtype_;
};

class Key final : public RegKeyBuilder {
 public:
  explicit Key() = default;
};

}  // namespace builder

namespace helper {

template<typename HelperClass, typename... Args>
struct Assurer4RunOnlyOnce final {
  Assurer4RunOnlyOnce(Args&&... args) { static HelperClass x(std::forward<Args>(args)...); }
};

template<OperatorConf::OpTypeCase op_type>
struct RegisterHelper4DeviceAndDType final {
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

template<OperatorConf::OpTypeCase op_type>
struct RegisterHelper4Device final {
  static Kernel* CreateKernel(const KernelConf& kernel_conf) {
    const auto& registry = KernelRegistry<op_type, std::string>();
    return registry.at(GetHashKey(kernel_conf.op_attribute().op_conf().device_type()))();
  }
  RegisterHelper4Device() {
    REGISTER_CLASS_CREATOR(op_type, Kernel, RegisterHelper4Device::CreateKernel, const KernelConf&);
  }
};

}  // namespace helper

}  // namespace kernel_registration

#define REGISTER_KERNEL_BASE(op_type, KeyT, RegisterHelperT, key, ...)                         \
  namespace {                                                                                  \
  static kernel_registration::helper::Assurer4RunOnlyOnce<RegisterHelperT<op_type>> OF_PP_CAT( \
      g_assurer, __LINE__);                                                                    \
  static kernel_registration::KernelRegistrar<op_type, KeyT> OF_PP_CAT(g_registrar,            \
                                                                       __LINE__)(key, []() {   \
    return new __VA_ARGS__();                                                                  \
  });                                                                                          \
  }  // namespace

#define REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(op_type, reg_key_builder, ...)       \
  REGISTER_KERNEL_BASE(op_type, std::string,                                       \
                       kernel_registration::helper::RegisterHelper4DeviceAndDType, \
                       kernel_registration::builder::reg_key_builder.Build(), __VA_ARGS__)

#define REGISTER_KERNEL_WITH_DEVICE(op_type, reg_key_builder, ...)                               \
  REGISTER_KERNEL_BASE(op_type, std::string, kernel_registration::helper::RegisterHelper4Device, \
                       kernel_registration::builder::reg_key_builder.Build(), __VA_ARGS__)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_KERNEL_REGISTRAR_H_
