#ifndef ONEFLOW_XLA_OF2XLA_PASS_REWRITE_OPTIMIZER_H_
#define ONEFLOW_XLA_OF2XLA_PASS_REWRITE_OPTIMIZER_H_

#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/graph/node.h"

namespace oneflow {
namespace xrt {

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
  static OperatorConf Build(const OptimizerMode &mode, const XrtNode *node,
                            const std::string &gradient,
                            const std::string &total_instances,
                            const std::string &learning_rate);

 private:
  class BuilderImpl {
   public:
    BuilderImpl(const XrtNode *node, const std::string &gradient,
                const std::string &total_instances,
                const std::string &learning_rate, OperatorConf *op_conf)
        : node_(node),
          gradient_(gradient),
          total_instances_(total_instances),
          learning_rate_(learning_rate),
          op_conf_(op_conf) {}

    template <OptimizerMode mode>
    inline void ApplyBuild();

   private:
    const XrtNode *node_;
    const std::string &gradient_;
    const std::string &total_instances_;
    const std::string &learning_rate_;
    OperatorConf *op_conf_;
  };

  static void ApplyOptimizerModeVisitor(const OptimizerMode &mode,
                                        BuilderImpl builder);
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XLA_OF2XLA_PASS_REWRITE_OPTIMIZER_H_
