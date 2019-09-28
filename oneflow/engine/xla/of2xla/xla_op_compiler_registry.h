#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_XLA_OP_COMPILER_REGISTRY_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_XLA_OP_COMPILER_REGISTRY_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "glog/logging.h"

#include "oneflow/engine/xla/of2xla/xla_op_compiler.h"

namespace oneflow {
namespace mla {

class XlaOpCompilerRegistry {
 public:
  typedef XlaOpCompiler* (*CreatorFunc)();

  struct XlaBackendRegistry {
    void Register(const std::string &type, CreatorFunc creator) {
      DCHECK_EQ(creator_map_.count(type), 0);
      creator_map_.emplace(type, creator);
    }

    CreatorFunc Build(const std::string &type) {
      DCHECK_GT(creator_map_.count(type), 0) << DebugString();
      return creator_map_[type];
    }

    CreatorFunc operator[](const std::string &type) { return this->Build(type); }

    bool IsRegistered(const std::string &type) {
      return (creator_map_.count(type) > 0) ? true : false;
    }

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
    // Currently the backend is one of "CPU" and "CUDA"
    std::string backend_;
    // Registered creator factories
    std::unordered_map<std::string, CreatorFunc> creator_map_;
  };

  typedef std::unordered_map<std::string, XlaBackendRegistry> XlaBackendFactory;
  typedef std::vector<std::string> StringVec;
  typedef std::unordered_map<std::string, StringVec> XlaMutableFactory;

  static void Register(const std::string &backend, const std::string &type,
                       CreatorFunc creator) {
    XlaBackendFactory *factory = XlaOpCompilerRegistry::Factory();
    XlaBackendRegistry *backend_registry = &((*factory)[backend]);
    backend_registry->backend_ = backend;
    backend_registry->Register(type, creator);
  }

  static XlaBackendRegistry &Build(const std::string &backend) {
    XlaBackendFactory *factory = XlaOpCompilerRegistry::Factory();
    DCHECK_GT(factory->count(backend), 0);
    return (*factory)[backend];
  }

  static XlaOpCompiler *Build(const std::string &backend,
                              const std::string &type) {
    return Build(backend)[type]();
  }

  static bool IsRegistered(const std::string &backend, const std::string &type) {
    bool registered = false;
    XlaBackendFactory *factory = XlaOpCompilerRegistry::Factory();
    if (factory->count(backend) > 0) {
      XlaBackendRegistry *backend_registry = &((*factory)[backend]);
      registered = backend_registry->IsRegistered(type);
    }
    return registered;
  }

  static void SetMutableVariables(const std::string &type,
                                 const std::vector<std::string> &variables) {
    XlaMutableFactory *factory = XlaOpCompilerRegistry::MutableFactory();
    factory->emplace(type, variables);
  }

  static const StringVec &GetMutableVariables(const std::string &type) {
    XlaMutableFactory *factory = XlaOpCompilerRegistry::MutableFactory();
    if (factory->count(type) == 0) {
      factory->emplace(type, StringVec{});
    }
    return factory->at(type);
  }

 private:
  static XlaBackendFactory *Factory() {
    static XlaBackendFactory *factory = new XlaBackendFactory;
    return factory;
  }

  static XlaMutableFactory *MutableFactory() {
    static XlaMutableFactory *factory = new XlaMutableFactory;
    return factory;
  }
};

template <typename OpCompiler>
class XlaOpCompilerRegistrar {
 public:
  XlaOpCompilerRegistrar(const std::string &type)
      : backend_(""), type_(type) {
    auto creator = []() -> XlaOpCompiler* { return new OpCompiler; };
    XlaOpCompilerRegistry::Register("CPU", type_, creator);
    XlaOpCompilerRegistry::Register("CUDA", type_, creator);
  }

  XlaOpCompilerRegistrar(const std::string &backend, const std::string &type)
      : backend_(backend), type_(type) {
    auto creator = []() -> XlaOpCompiler* { return new OpCompiler; };
    XlaOpCompilerRegistry::Register(backend_, type_, creator);
  }

  XlaOpCompilerRegistrar& MutableVariables(
      const std::vector<std::string> &variables) {
    XlaOpCompilerRegistry::SetMutableVariables(type_, variables);
    return *this;
  }

private:
  std::string backend_;
  std::string type_;
};

#define REGISTER_XLA_OP_COMPILER(Type, OpCompiler) \
  static XlaOpCompilerRegistrar<OpCompiler>                      \
      g_xla_all__##Type##__op_compiler __attribute__((unused)) = \
      XlaOpCompilerRegistrar<OpCompiler>(#Type)

#define REGISTER_XLA_CPU_OP_COMPILER(Type, OpCompiler)           \
  static XlaOpCompilerRegistrar<OpCompiler>                      \
      g_xla_cpu__##Type##__op_compiler __attribute__((unused)) = \
      XlaOpCompilerRegistrar<OpCompiler>("CPU", #Type)

#define REGISTER_XLA_CUDA_OP_COMPILER(Type, OpCompiler)          \
  static XlaOpCompilerRegistrar<OpCompiler>                      \
      g_xla_gpu__##Type##__op_compiler __attribute__((unused)) = \
      XlaOpCompilerRegistrar<OpCompiler>("CUDA", #Type)


inline bool IsOpCompilerRegistered(const std::string &backend,
                                   const std::string &op_type) {
  return XlaOpCompilerRegistry::IsRegistered(backend, op_type);
}

inline const std::vector<std::string> &GetMutableVariables(
    const std::string &op_type) {
  return XlaOpCompilerRegistry::GetMutableVariables(op_type);
}

inline std::shared_ptr<XlaOpCompiler> CreateXlaOpCompiler(
    const std::string &backend, const std::string &op_type) {
  return std::shared_ptr<XlaOpCompiler>(
      XlaOpCompilerRegistry::Build(backend, op_type));
}

}  // namespace mla
}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_XLA_OF2XLA_XLA_OP_COMPILER_REGISTRY_H_  
