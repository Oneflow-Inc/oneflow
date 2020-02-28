#ifndef ONEFLOW_CORE_EXTENSION_EVENT_HANDLER_H
#define ONEFLOW_CORE_EXTENSION_EVENT_HANDLER_H
#include "oneflow/core/extension/extension_base.h"
#include "oneflow/core/extension/extension_registration.h"
namespace oneflow {
namespace extension {
void kernel_event(std::string event_name, const Kernel* kernel) {
  for (extension::ExtensionBase* ext : LookUpExtensionRegistry(event_name)) {
    KernelExtensionContext* ctx = kernel->kernel_ext_ctx.get();
    KernelEvent event;
    event.name = event_name;
    event.context = ctx;
    event.kernel_ptr = kernel;
    ext->callback(&event);
  }
}
}  // namespace extension
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EXTENSION_EVENT_HANDLER_H
