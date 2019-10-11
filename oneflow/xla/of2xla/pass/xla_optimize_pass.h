#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_

#include "oneflow/core/job/job.pb.h"  // Job
#include "oneflow/xla/of2xla/xla_graph.h"

namespace oneflow {
namespace mola {

struct ClusteringOptions {
  // Minimum node number in each cluster after clustering. If the number of
  // nodes contained by a cluster is less than `clustering_minimum_nodes`
  // or grater than `clustering_maximum_nodes`, then this cluster will be
  // given up and not compiled.
  int32_t clustering_minimum_nodes = 0x1;
  int32_t clustering_maximum_nodes = 0x7fffffff;

  // Option to clustering with strict dependencies analysis.
  bool strict_clustering = true;

  bool ignore_sbp_policy = false;
  bool ignore_time_shape = false;
};

struct OptimizeOptions {
  // Job is required by `RebuildCompiledJobPass` and `RewriteOptimizerPass`
  Job *job;

  XlaGraph *graph;

  ClusteringOptions clustering_options;
};

class XlaOptimizePass {
 public:
  XlaOptimizePass(const OptimizeOptions &options)
      : options_(options) {}

  virtual ~XlaOptimizePass() {}

  virtual void Run() = 0;
  
  // Create an OptimizePass instance by pass type
  static XlaOptimizePass *Create(const std::string &pass_type,
                                 const OptimizeOptions &options);

  typedef std::function<XlaOptimizePass *(const OptimizeOptions &)> PassFactory;
  // Push a new pass factory
  static void Register(const std::string &pass_type, PassFactory factory);

 protected:
  OptimizeOptions options_;
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

OptimizeOptions CreateDefaultOptimizeOptions();

void RunOptimizePass(const std::string &pass, OptimizeOptions &options);

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_
