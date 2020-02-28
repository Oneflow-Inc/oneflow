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
  void callback(Event ev) override {
    ExtensionContext* ctx = ev.context;
    auto kernel_ctx = dynamic_cast<KernelExtensionContext*>(ctx);
    if (ev.name == "Kernel/WillForward") {
      ctx->set_state(name(), reinterpret_cast<void*>(new MyStruct(10, 12.2)));
      // TODO: by niuchong 也可考虑传递std::shared_ptr<void>，而非void*
    } else if (ev.name == "Kernel/DidForward") {
      MyStruct* my_struct = reinterpret_cast<MyStruct*>(ctx->get_state(name()));
    }
  }
};
REGISTER_EXTENSION("Kernel/DidLaunch", []() { return new MyExtension(); });
}  // namespace oneflow
