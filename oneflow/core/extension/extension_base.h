#include <string>
#include "oneflow/core/actor/actor.h"
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
 public:
  const RuntimeExtensionContext* runtime_cxt() { return runtime_cxt_; }

 private:
  RuntimeExtensionContext* runtime_cxt_;
};
class ActorExtensionContext : public ExtensionContext {
 public:
  std::function<Blob*(const std::string&)> BnInOp2Blob;
  const ThreadExtensionContext* thread_cxt() { return thread_cxt_; }

 private:
  ThreadExtensionContext* thread_cxt_;
};
class KernelExtensionContext : public ExtensionContext {
 public:
  Kernel* kernel_ptr;
  const ActorExtensionContext* actor_ctx() { return actor_ctx_; }

 private:
  friend Kernel;
  ActorExtensionContext* actor_ctx_;
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
