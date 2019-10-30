#ifndef ONEFLOW_XRT_XLA_OPS_OP_KERNEL_H_
#define ONEFLOW_XRT_XLA_OPS_OP_KERNEL_H_

#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/xla_macro.h"

namespace oneflow {
namespace xrt {
namespace mola {

class OpKernel {
 public:
  virtual void Compile(OpContext *ctx) = 0;

  OpKernel() = default;
  virtual ~OpKernel() = default;
};

template <typename KernelType>
class OpKernelRegistrar {
 private:
  std::function<OpKernel *()> factory_;
  std::vector<XrtDevice> device_{CPU_X86, GPU_CUDA};
  XrtEngine engine_field_;

  std::string op_name_;
  util::Map<std::string, Any> attributes_;

 public:
  explicit OpKernelRegistrar(const std::string &name) : op_name_(name) {}

  virtual ~OpKernelRegistrar() = default;

  OpKernelRegistrar<KernelType> &SetFactory(decltype(factory_) factory) {
    factory_ = factory;
    return *this;
  }

  OpKernelRegistrar<KernelType> &SetField(const XrtEngine &field) {
    engine_field_ = field;
    return *this;
  }

  OpKernelRegistrar<KernelType> &SetDevice(
      const std::vector<XrtDevice> &device) {
    device_ = device;
    return *this;
  }

  OpKernelRegistrar<KernelType> &SetMutableVariables(
      const std::vector<std::string> &variables) {
    attributes_["MutableVars"] = variables;
    return *this;
  }

  OpKernelRegistrar<KernelType> &Finalize() {
    Factory()->Register(op_name_, factory_, attributes_);
    for (const auto &device : device_) {
      auto field = std::make_pair(engine_field_, device);
      FactoryManager()->Insert(field, Factory());
    }
    return *this;
  }

  auto Factory() -> util::Registry<std::string, decltype(factory_)> * {
    return util::Registry<std::string, decltype(factory_)>::Global();
  }

  typedef std::pair<XrtEngine, XrtDevice> FieldType;
  auto FactoryManager() -> util::RegistryManager<FieldType> * {
    return util::RegistryManager<FieldType>::Global();
  }
};

#define REGISTER_XLA_OP_COMPILER(OpName, KernelType)              \
  static OpKernelRegistrar<KernelType> _xla_op_kernel_##OpName##_ \
      __attribute__((unused)) =                                   \
          OpKernelRegistrar<KernelType>(#OpName)                  \
              .SetField(XrtEngine::XLA)                           \
              .SetFactory([]() -> OpKernel * { return new KernelType; })

struct OpKernelBuilder {
  OpKernel *operator()(const std::string &op_name) {
    return util::Registry<std::string, std::function<OpKernel *()>>::Global()
        ->Lookup(op_name)();
  }
};

inline std::shared_ptr<OpKernel> BuildOpKernel(const XrtDevice &device,
                                               const std::string &op_name) {
  return std::shared_ptr<OpKernel>(OpKernelBuilder()(op_name));
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_OP_KERNEL_H_
