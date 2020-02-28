#include <string>
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {
namespace extension {

class ExtensionContext {
 public:
  virtual ~ExtensionContext() = default;
  void* get_state(std::string ext_name) {
    auto it = ext_name2state_.find(ext_name);
    if (it == ext_name2state_.end()) {
      ext_name2state_[ext_name] = nullptr;
      return nullptr;
    } else {
      return it->second;
    }
  }
  void set_state(std::string ext_name, void* new_state) { ext_name2state_[ext_name] = new_state; }

 private:
  HashMap<std::string, void*> ext_name2state_;
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
