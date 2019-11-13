#ifndef ONEFLOW_XRT_KERNEL_OP_KERNEL_H_
#define ONEFLOW_XRT_KERNEL_OP_KERNEL_H_

#include "oneflow/xrt/kernel/op_context.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

template <typename T>
class OpKernel {
 public:
  virtual void Compile(T *ctx) = 0;

  OpKernel() = default;
  virtual ~OpKernel() = default;
};

template <typename T>
class OpKernelRegistrar {
 private:
  std::function<OpKernel<T> *()> factory_;

  std::string op_name_;

  XrtEngine engine_field_;
  std::vector<XrtDevice> device_{CPU_X86, GPU_CUDA};

  bool train_phase_enabled_ = false;
  util::Map<std::string, Any> attributes_;

 public:
  explicit OpKernelRegistrar(const std::string &name) : op_name_(name) {}

  virtual ~OpKernelRegistrar() = default;

  auto FieldRegistry()
      -> util::FieldRegistry<XrtField, std::string, decltype(factory_)> * {
    return util::FieldRegistry<XrtField, std::string,
                               decltype(factory_)>::Global();
  }

  OpKernelRegistrar &SetFactory(decltype(factory_) factory) {
    factory_ = factory;
    return *this;
  }

  OpKernelRegistrar &SetField(const XrtEngine &field) {
    engine_field_ = field;
    return *this;
  }

  OpKernelRegistrar &SetDevice(const std::vector<XrtDevice> &device) {
    device_ = device;
    return *this;
  }

  OpKernelRegistrar &SetMutableVariables(
      const util::Set<std::string> &variables) {
    attributes_[MutableVariablesAttrName] = variables;
    return *this;
  }

  OpKernelRegistrar &EnableTrainPhase() {
    train_phase_enabled_ = true;
    return *this;
  }

  OpKernelRegistrar &Finalize() {
    attributes_[TrainPhaseEnabledAttrName] = train_phase_enabled_;
    for (const auto &device : device_) {
      XrtField field = MakeXrtField(device, engine_field_);
      FieldRegistry()->Get(field)->Register(op_name_, factory_, attributes_);
    }
    return *this;
  }
};

template <typename T>
struct OpKernelBuilder {
  OpKernel<T> *operator()(const XrtField &field, const std::string &op_name) {
    return util::FieldRegistry<XrtField, std::string,
                               std::function<OpKernel<T> *()>>::Global()
        ->Get(field)
        ->Lookup(op_name)();
  }
};

inline bool OpKernelRegistered(const std::string &op_type,
                               const XrtField &field) {
  auto *rm = util::RegistryManager<XrtField>::Global();
  if (rm->HasRegistry(field)) {
    return rm->GetRegistry(field)->IsRegistered(op_type);
  }
  return false;
}

inline bool TrainPhaseEnabled(const std::string &op_type,
                              const XrtField &field) {
  auto *rm = util::RegistryManager<XrtField>::Global();
  CHECK(rm->HasRegistry(field))
      << "Field registry has not been found. Field is (engine: "
      << field.engine() << ", device: " << field.device() << ").";
  const auto &attrs = rm->GetRegistry(field)->LookupAttr(op_type);
  CHECK_GT(attrs.count(TrainPhaseEnabledAttrName), 0)
      << "TrainPhaseEnabled attrs is not set for type " << op_type;
  return any_cast<bool>(attrs.at(TrainPhaseEnabledAttrName));
}

inline util::Set<std::string> MutableVariables(const std::string &op_type,
                                               const XrtField &field) {
  auto *rm = util::RegistryManager<XrtField>::Global();
  CHECK(rm->HasRegistry(field))
      << "Field registry has not been found. Field is (engine: "
      << field.engine() << ", device: " << field.device() << ").";
  const auto &attrs = rm->GetRegistry(field)->LookupAttr(op_type);
  const auto &it = attrs.find(MutableVariablesAttrName);
  if (it != attrs.end()) {
    return any_cast<util::Set<std::string>>(it->second);
  }
  return util::Set<std::string>();
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_KERNEL_OP_KERNEL_H_
