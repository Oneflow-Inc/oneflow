#ifndef ONEFLOW_CORE_EXTENSION_REGISTRAR_H_
#define ONEFLOW_CORE_EXTENSION_REGISTRAR_H_

#include "oneflow/core/extension/extension.h"
#include <string>
namespace oneflow {

namespace {
HashMap<std::string, std::vector<std::function<extension::ExtensionBase*()>>>* MutKernelRegistry() {
  static HashMap<std::string, std::vector<std::function<extension::ExtensionBase*()>>> creators;
  return &creators;
}

}  // namespace

namespace extension {

struct Registrar final {
  Registrar(std::string ev_name, std::function<extension::ExtensionBase*()> ext_contructor) {
    auto* creators = MutKernelRegistry();
    (*creators)[ev_name].emplace_back(std::move(ext_contructor));
  }
};

}  // namespace extension

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EXTENSION_REGISTRAR_H_
