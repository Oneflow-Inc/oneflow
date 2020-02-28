#ifndef ONEFLOW_CORE_EXTENSION_REGISTRAR_H_
#define ONEFLOW_CORE_EXTENSION_REGISTRAR_H_

#include "oneflow/core/extension/extension_base.h"
#include <string>

namespace oneflow {
namespace extension {

struct Registrar final {
  Registrar(std::string ev_name, std::function<extension::ExtensionBase*()> ext_contructor);
};

const std::vector<extension::ExtensionBase*> LookUpExtensionRegistry(const std::string& ev_name);

}  // namespace extension
}  // namespace oneflow
#endif  // ONEFLOW_CORE_EXTENSION_REGISTRAR_H_
