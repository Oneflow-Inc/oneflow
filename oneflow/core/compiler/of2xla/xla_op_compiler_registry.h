#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_COMPILER_REGISTRY_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_COMPILER_REGISTRY_H_

#include <string>
#include <vector>
#include <unordered_map>
#include "glog/logging.h"

namespace oneflow {
namespace mola {

class XlaOpCompiler;

class XlaOpCompilerRegistry {
 public:
  typedef XlaOpCompiler* (*CreatorFunc)();

  struct XlaBackendRegistry {
    void Register(const std::string &type, CreatorFunc creator) {
      DCHECK_EQ(creator_map_.count(type), 0);
      creator_map_.emplace(type, creator);
    }

    CreatorFunc Get(const std::string &type) {
      DCHECK_GT(creator_map_.count(type), 0) << DebugString();
      return creator_map_[type];
    }

    CreatorFunc operator[](const std::string &type) { return this->Get(type); }

    std::vector<std::string> ListAllRegisteredTypes() {
      std::vector<std::string> all_register_types(creator_map_.size());
      int index = 0;
      for (const auto &it : creator_map_) {
        all_register_types[index] = it.first;
        ++index;
      }
      return all_register_types;
    }

    std::string DebugString() {
      const auto all_register_types = ListAllRegisteredTypes();
      int size = all_register_types.size();
      std::string debug_str = "All registered types for backend " + backend_ + " (";
      for (int i = 0; i < size - 1; ++i) { debug_str += all_register_types[i] + ", "; }
      if (size > 0) { debug_str += all_register_types[size - 1]; }
      debug_str += ")";
      return debug_str;
    }
    // Currently CPU or CUDA
    std::string backend_;
    // Registered creator factory
    std::unordered_map<std::string, CreatorFunc> creator_map_;
  };

  typedef std::unordered_map<std::string, XlaBackendRegistry> XlaBackendFactory;

  static void Register(const std::string &backend, const std::string &type, CreatorFunc creator) {
    XlaBackendFactory *factory = XlaOpCompilerRegistry::Factory();
    XlaBackendRegistry *backend_registry = &((*factory)[backend]);
    backend_registry->backend_ = backend;
    backend_registry->Register(type, creator);
  }

  static XlaBackendRegistry &Get(const std::string &backend) {
    XlaBackendFactory *factory = XlaOpCompilerRegistry::Factory();
    DCHECK_GT(factory->count(backend), 0);
    return (*factory)[backend];
  }

 private:
  static XlaBackendFactory *Factory() {
    static XlaBackendFactory *factory = new XlaBackendFactory;
    return factory;
  }
};

class XlaOpCompilerRegistrar {
 public:
  XlaOpCompilerRegistrar(const std::string &backend, const std::string &type,
                          XlaOpCompilerRegistry::CreatorFunc creator) {
    XlaOpCompilerRegistry::Register(backend, type, creator);
  }
};

#define REGISTER_XLA_OP_COMPILER(Type, OpCompiler) \
  REGISTER_XLA_CPU_OP_COMPILER(Type, OpCompiler);  \
  REGISTER_XLA_CUDA_OP_COMPILER(Type, OpCompiler);

#define REGISTER_XLA_CPU_OP_COMPILER(Type, OpCompiler) \
  static XlaOpCompilerRegistrar g_xla_cpu__##Type##__compiler =                \
      XlaOpCompilerRegistrar("CPU", #Type,                                     \
                             []() -> XlaOpCompiler* { return new OpCompiler; });

#define REGISTER_XLA_CUDA_OP_COMPILER(Type, OpCompiler) \
  static XlaOpCompilerRegistrar g_xla_gpu__##Type##__op_compiler =              \
      XlaOpCompilerRegistrar("CUDA", #Type,                                     \
                             []() -> XlaOpCompiler* { return new OpCompiler; });

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_XLA_OP_COMPILER_REGISTRY_H_
