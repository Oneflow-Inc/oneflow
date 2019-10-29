#ifndef ONEFLOW_XRT_XLA_OPS_OP_COMPILER_H_
#define ONEFLOW_XRT_XLA_OPS_OP_COMPILER_H_

#include "oneflow/xrt/platform.h"
#include "oneflow/xrt/utility/registry.h"
#include "oneflow/xrt/xla/op_context.h"
#include "oneflow/xrt/xla/xla_macro.h"

namespace oneflow {
namespace xrt {
namespace mola {

class OpCompiler {
 public:
  virtual void Compile(OpContext *ctx) = 0;

  OpCompiler() = default;
  virtual ~OpCompiler() = default;
};

template <typename OpType>
class OpRegistrar {
 private:
  std::function<OpCompiler *()> factory_;
  std::vector<XrtDevice> device_{CPU_X86, GPU_CUDA};
  XrtEngine engine_field_;

  std::string op_name_;
  util::Map<std::string, Any> attributes_;

 public:
  explicit OpRegistrar(const std::string &name) : op_name_(name) {}

  virtual ~OpRegistrar() = default;

  OpRegistrar<OpType> &SetFactory(decltype(factory_) factory) {
    factory_ = factory;
    return *this;
  }

  OpRegistrar<OpType> &SetField(const XrtEngine &field) {
    engine_field_ = field;
    return *this;
  }

  OpRegistrar<OpType> &SetDevice(const std::vector<XrtDevice> &device) {
    device_ = device;
    return *this;
  }

  OpRegistrar<OpType> &SetMutableVariables(
      const std::vector<std::string> &variables) {
    attributes_["MutableVars"] = variables;
    return *this;
  }

  OpRegistrar<OpType> &Finalize() {
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

#define REGISTER_XLA_OP_COMPILER(OpName, CompilerType)       \
  static OpRegistrar<CompilerType> _xla_compiler_##OpName##_ \
      __attribute__((unused)) =                              \
          OpRegistrar<CompilerType>(#OpName)                 \
              .SetField(XrtEngine::XLA)                      \
              .SetFactory([]() -> OpCompiler * { return new CompilerType; })

struct CompilerBuilder {
  OpCompiler *operator()(const std::string &op_name) {
    return util::Registry<std::string, std::function<OpCompiler *()>>::Global()
        ->Lookup(op_name)();
  }
};

inline std::shared_ptr<OpCompiler> CreateOpCompiler(
    const XrtDevice &device, const std::string &op_name) {
  return std::shared_ptr<OpCompiler>(CompilerBuilder()(op_name));
}

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_OPS_OP_COMPILER_H_
