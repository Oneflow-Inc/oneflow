#ifndef ONEFLOW_CORE_EXTENSION_EXTENSION_BASE_H_
#define ONEFLOW_CORE_EXTENSION_EXTENSION_BASE_H_

#include "oneflow/core/actor/actor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel.h"

namespace oneflow {
namespace extension {

class ExtensionContext {
 public:
  virtual ~ExtensionContext() = default;
  std::shared_ptr<void> get_state(std::string ext_name) {
    auto it = ext_name2state_.find(ext_name);
    if (it == ext_name2state_.end()) {
      ext_name2state_[ext_name] = nullptr;
      return nullptr;
    } else {
      return it->second;
    }
  }
  void set_state(std::string ext_name, std::shared_ptr<void> new_state) {
    ext_name2state_[ext_name] = new_state;
  }

 private:
  HashMap<std::string, std::shared_ptr<void>> ext_name2state_;
};
class RuntimeExtensionContext : public ExtensionContext {};
class ThreadExtensionContext : public ExtensionContext {
 public:
  void set_actor_ext_ctx(std::shared_ptr<RuntimeExtensionContext> new_ext_ctx) {
    runtime_cxt_ = new_ext_ctx;
  }
  const std::shared_ptr<RuntimeExtensionContext> get_actor_ext_ctx() { return runtime_cxt_; }

 private:
  std::shared_ptr<RuntimeExtensionContext> runtime_cxt_;
};
class ActorExtensionContext : public ExtensionContext {
 public:
  void set_actor_ext_ctx(std::shared_ptr<ThreadExtensionContext> new_ext_ctx) {
    thread_cxt_ = new_ext_ctx;
  }
  const std::shared_ptr<ThreadExtensionContext> get_actor_ext_ctx() { return thread_cxt_; }

 private:
  std::shared_ptr<ThreadExtensionContext> thread_cxt_;
};
class KernelExtensionContext : public ExtensionContext {
 public:
  void set_actor_ext_ctx(std::shared_ptr<ActorExtensionContext> new_ext_ctx) {
    actor_ext_ctx_ = new_ext_ctx;
  }
  const std::shared_ptr<ActorExtensionContext> get_actor_ext_ctx() { return actor_ext_ctx_; }

 private:
  std::shared_ptr<ActorExtensionContext> actor_ext_ctx_;
};
class Event {
 public:
  virtual ~Event() = default;
  std::string name;
};
class KernelEvent : public Event {
 public:
  const Kernel* kernel_ptr;
  std::function<Blob*(const std::string&)> BnInOp2Blob;
  std::shared_ptr<KernelExtensionContext> kernel_ext_ctx;
};
class ExtensionBase {
 public:
  virtual std::string name() = 0;
  virtual void callback(const Event* event) = 0;
};
}  // namespace extension
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EXTENSION_EXTENSION_BASE_H_
