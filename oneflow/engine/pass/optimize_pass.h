#ifndef ONEFLOW_ENGINE_PASS_OPTIMIZE_PASS_H_
#define ONEFLOW_ENGINE_PASS_OPTIMIZE_PASS_H_

#include "oneflow/engine/xla/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

struct OptimizeOptions {
  XlaGraph *graph;
  // Minimum node number in each cluster in `XlaClusterCompiledOps` pass.
  // If the number of nodes contained by a cluster is less than
  // `minimum_nodes_in_cluster`, then this cluster will be given up and
  // not compiled.
  int32_t minimum_nodes_in_cluster = 0x1;
  int32_t maximum_nodes_in_cluster = 0x7fffffff;

  bool ignore_sbp_policy = false;
  bool ignore_time_shape = false;
};

class OptimizePass {
 public:
  OptimizePass(const OptimizeOptions &options)
      : optimize_options_(options) {}

  virtual ~OptimizePass() {}

  virtual void Run() = 0;
  
  // Create an OptimizePass instance by pass type
  static OptimizePass *Create(const std::string &pass_type,
                                 const OptimizeOptions &options);

  typedef std::function<OptimizePass *(const OptimizeOptions &)> PassFactory;
  // Push a new pass factory
  static void Register(const std::string &pass_type, PassFactory factory);

 protected:
  OptimizeOptions optimize_options_;
};

template <typename OptimizePassClass>
class OptimizePassRegistrar {
 public:
  OptimizePassRegistrar(const std::string &pass_type) {
    auto factory = [](const OptimizeOptions &options) -> OptimizePassClass* {
        return new OptimizePassClass(options);
    };
    OptimizePass::Register(pass_type, factory);
  }
};

#define REGISTER_OPTIMIZE_PASS(PassType, OptimizePassClass)       \
  static OptimizePassRegistrar<OptimizePassClass>              \
      _optimize_pass_##PassType##_ __attribute__((unused)) =  \
      OptimizePassRegistrar<OptimizePassClass>(#PassType);


void RunOptimizePass(const std::string &pass, OptimizeOptions &options);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_PASS_OPTIMIZE_PASS_H_

