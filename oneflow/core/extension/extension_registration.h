#ifndef ONEFLOW_CORE_EXTENSION_REGISTRAR_H_
#define ONEFLOW_CORE_EXTENSION_REGISTRAR_H_

#include "oneflow/core/extension/extension_base.h"

namespace oneflow {

namespace extension {

struct Registrar final {
  Registrar(const std::string&, std::function<extension::ExtensionBase*()>);
  Registrar(const std::vector<std::string>&, std::function<extension::ExtensionBase*()>);
};

const std::vector<std::function<extension::ExtensionBase*()>>* LookUpExtensionRegistry(
    const std::string&);

}  // namespace extension

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EXTENSION_REGISTRAR_H_
