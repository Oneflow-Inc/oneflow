#include "oneflow/core/extension/extension.h"
namespace oneflow {
using namespace oneflow::extension;
class MyStruct {
 public:
  MyStruct(int, float) {}
  int a;
  float b;
};
class MyExtension final : public ExtensionBase {
  std::string name() override { return "kernel sync"; }
  void callback(const Event* ev) override {
    const KernelEvent* kernel_ev = dynamic_cast<const KernelEvent*>(ev);
    KernelExtensionContext* kernel_ctx = kernel_ev->context;
    if (ev->name == "Kernel/WillForward") {
      kernel_ctx->set_state(name(), std::shared_ptr<void>(new MyStruct(10, 12.2)));
    } else if (ev->name == "Kernel/DidForward") {
      LOG(ERROR) << "Kernel/DidForward: " << kernel_ev->kernel_ptr->op_conf().name();
      MyStruct* my_struct = reinterpret_cast<MyStruct*>(kernel_ctx->get_state(name()).get());
    }
  }
};
REGISTER_EXTENSION("Kernel/DidForward", []() { return new MyExtension(); });
}  // namespace oneflow
