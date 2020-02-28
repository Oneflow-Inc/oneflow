#include <string>
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {
namespace extension {

class ExtensionContext {
 public:
  virtual ~ExtensionContext() = default;
  void* get_state(std::string ext_name) { return state_; }
  void set_state(std::string ext_name, void* new_state) { state_ = new_state; }

 private:
  // FIXME: by tsai, member state_ only for demo, state should be managed by oneflow runtime
  void* state_;
};
class RuntimeExtensionContext : public ExtensionContext {};
class ThreadExtensionContext : public ExtensionContext {
  RuntimeExtensionContext* runtime_cxt;
};
class ActorExtensionContext : public ExtensionContext {
  std::function<Blob*(const std::string&)> BnInOp2Blob;
  ThreadExtensionContext* thread_cxt;
};
class KernelExtensionContext : public ExtensionContext {
  Kernel* kernel_ptr;
  ActorExtensionContext* actor_ctx;
};
class Event {
 public:
  std::string name;
  ExtensionContext* context;
};
class ExtensionBase {
 public:
  virtual std::string name() = 0;
  virtual void callback(Event event) = 0;
};
}  // namespace extension
}  // namespace oneflow
