#ifndef ONEFLOW_CORE_EXTENSION_EXTENSION_BASE_H_
#define ONEFLOW_CORE_EXTENSION_EXTENSION_BASE_H_

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
  RuntimeExtensionContext* runtime_cxt;
};
class ActorExtensionContext : public ExtensionContext {
 public:
  std::function<Blob*(const std::string&)> BnInOp2Blob;
  ThreadExtensionContext* thread_cxt;
};
class KernelExtensionContext : public ExtensionContext {
 public:
  ActorExtensionContext* actor_ctx;
};
class Event {
 public:
  virtual ~Event() = default;
  std::string name;
};
class KernelEvent : public Event {
 public:
  const Kernel* kernel_ptr;
  KernelExtensionContext* context;
};
class ExtensionBase {
 public:
  virtual std::string name() = 0;
  virtual void callback(const Event* event) = 0;
};
}  // namespace extension
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EXTENSION_EXTENSION_BASE_H_
