#ifndef ONEFLOW_CORE_EXTENSION_EVENT_HANDLER_H
#define ONEFLOW_CORE_EXTENSION_EVENT_HANDLER_H

#include "oneflow/core/extension/extension_base.h"
#include "oneflow/core/extension/extension_registration.h"

namespace oneflow {

namespace extension {

void kernel_event(std::string event_name, const Kernel* kernel,
                  std::function<Blob*(const std::string&)> BnInOp2Blob) {
  auto* ext_constructors = LookUpExtensionRegistry(event_name);
  if (ext_constructors == nullptr) {
    return;
  } else {
    for (const std::function<extension::ExtensionBase*()> ext_constructor : *ext_constructors) {
      KernelEvent event;
      event.name = event_name;
      event.kernel_ext_ctx = kernel->get_kernel_ext_ctx();
      event.kernel_ptr = kernel;
      event.BnInOp2Blob = BnInOp2Blob;
      ext_constructor()->callback(&event);
    }
  }
}

}  // namespace extension

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EXTENSION_EVENT_HANDLER_H
