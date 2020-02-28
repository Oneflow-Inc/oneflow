#include "oneflow/core/extension/extension_registration.h"

namespace oneflow {

namespace {
HashMap<std::string, std::vector<std::function<extension::ExtensionBase*()>>>*
MutExtensionRegistry() {
  static HashMap<std::string, std::vector<std::function<extension::ExtensionBase*()>>> creators;
  return &creators;
}

}  // namespace

namespace extension {

Registrar::Registrar(std::string ev_name,
                     std::function<extension::ExtensionBase*()> ext_contructor) {
  auto* creators = MutExtensionRegistry();
  LOG(ERROR) << "registering: " << ev_name;
  (*creators)[ev_name].emplace_back(std::move(ext_contructor));
}

const std::vector<std::function<extension::ExtensionBase*()>>* LookUpExtensionRegistry(
    const std::string& ev_name) {
  const auto registry = MutExtensionRegistry();
  auto it = registry->find(ev_name);
  if (it == registry->end()) {
    return nullptr;
  } else {
    return &it->second;
  }
}

}  // namespace extension

}  // namespace oneflow
