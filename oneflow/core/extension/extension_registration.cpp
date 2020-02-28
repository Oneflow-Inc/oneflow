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
  (*creators)[ev_name].emplace_back(std::move(ext_contructor));
}

const std::vector<extension::ExtensionBase*> LookUpExtensionRegistry(const std::string& ev_name) {
  std::vector<extension::ExtensionBase*> ext_vec;
  const auto registry = MutExtensionRegistry();
  auto it = registry->find(ev_name);
  if (it != registry->end()) {
    for (std::function<extension::ExtensionBase*()> ext_contructor : it->second) {
      ext_vec.push_back(ext_contructor());
    }
  }
  return ext_vec;
}

}  // namespace extension

}  // namespace oneflow
