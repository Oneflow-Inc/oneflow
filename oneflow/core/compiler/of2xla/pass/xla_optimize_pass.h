#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_

#include "oneflow/core/compiler/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

struct OptimizeOptions {
  XlaGraph *graph;

  // Minimum node number in each cluster in `XlaClusterCompiledOps` pass.
  // If the number of nodes contained by a cluster is less than
  // `minimum_nodes_in_cluster`, then this cluster will be given up and
  // not compiled.
  int32_t minimum_nodes_in_cluster; 
};

class XlaOptimizePass {
 public:
  XlaOptimizePass(const OptimizeOptions &options)
      : optimize_options_(options) {}

  virtual ~XlaOptimizePass() {}

  virtual void Run() = 0;
  
  // Create an OptimizePass instance by pass type
  static XlaOptimizePass *Create(const std::string &pass_type,
                                 const OptimizeOptions &options);

  typedef std::function<XlaOptimizePass *(const OptimizeOptions &)> PassFactory;
  // Push a new pass factory
  static void Register(const std::string &pass_type, PassFactory factory);

 protected:
  OptimizeOptions optimize_options_;
};

template <typename OptimizePassClass>
class XlaOptimizePassRegistrar {
 public:
  XlaOptimizePassRegistrar(const std::string &pass_type) {
    auto factory = [](const OptimizeOptions &options) -> OptimizePassClass* {
        return new OptimizePassClass(options);
    };
    XlaOptimizePass::Register(pass_type, factory);
  }
};

#define REGISTER_OPTIMIZE_PASS(PassType, OptimizePassClass)       \
  static XlaOptimizePassRegistrar<OptimizePassClass>              \
      _xla_optimize_pass_##PassType##_ __attribute__((unused)) =  \
      XlaOptimizePassRegistrar<OptimizePassClass>(#PassType);


void RunOptimizePass(const std::string &pass, OptimizeOptions &options);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_
