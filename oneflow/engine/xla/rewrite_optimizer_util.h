#ifndef ONEFLOW_ENGINE_XLA_REWRITE_OPTIMIZER_UTIL_H_
#define ONEFLOW_ENGINE_XLA_REWRITE_OPTIMIZER_UTIL_H_

#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

typedef enum class {
  kInvalid = 0,
  kNaive = 1,
  kMomentum = 2,
  kRMSProp = 3,
  kLARS = 4,
  kAdam = 5,
} OptimizerMode;

class OptimizerParamBuilder {
 public:
  static OperatorConf Build(const OptimizerMode &mode,
                            const mola::XlaNode *node,
                            const std::string &gradient,
                            const std::string &total_instances,
                            const std::string &learning_rate);

 private:
  class BuilderImpl {
   public:
    BuilderImpl(const mola::XlaNode *node, const std::string &gradient,
                const std::string &total_instances,
                const std::string &learning_rate, OperatorConf *op_conf)
        : node_(node), gradient_(gradient), total_instances_(total_instances),
      learning_rate_(learning_rate), op_conf_(op_conf) {}
    
    template<OptimizerMode mode>
    void ApplyBuild();

   private:
    const mola::XlaNode *node_;
    const std::string &gradient_;
    const std::string &total_instances_;
    const std::string &learning_rate_;
    OperatorConf *op_conf_;
  };

  static void ApplyOptimizerModeVisitor(const OptimizerMode &mode,
                                       BuilderImpl builder);
};

}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_XLA_REWRITE_OPTIMIZER_UTIL_H_
