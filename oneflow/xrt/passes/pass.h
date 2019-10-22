#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_

#include "oneflow/xrt/any.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/utility/registry.h"

namespace oneflow {
namespace xrt {

struct ClusteringOptions {
  // Minimum node number in each cluster after clustering. If the number of
  // nodes contained by a cluster is less than `minimum_nodes` or grater than
  // `maximum_nodes`, then this cluster will be given up and not compiled.
  int32_t minimum_nodes = 0x1;
  int32_t maximum_nodes = 0x7fffffff;

  // Option to clustering with strict dependencies analysis.
  bool strict_clustering = true;

  bool ignore_sbp_policy = false;
  bool ignore_time_shape = false;
};

struct XrtPassOptions {
  ClusteringOptions clustering_options;
};

class XrtPass {
 public:
  XrtPass() = default;
  virtual ~XrtPass() = default;

  virtual void Run(XrtGraph *graph, const XrtPassOptions &options) {
    LOG(FATAL) << "Should not call this function.";
  }

  virtual void Run(XrtGraph *graph, const XrtPassOptions &options,
                   const std::vector<Any> &params) {
    LOG(FATAL) << "Should not call this function.";
  }
};

// typedef XrtPass *(*XrtPassCreator)();

#define REGISTER_XRT_PASS(PassName, PassType)                            \
  std::function<XrtPass *()> _##PassName##_creator = []() -> XrtPass * { \
    return new PassType;                                                 \
  };                                                                     \
  XRT_REGISTER_FACTORY(PassName, _##PassName##_creator)

inline void RunPassImpl(const std::string &pass, XrtGraph *graph,
                        const XrtPassOptions &options) {
  auto optimize_pass =
      util::Registry<std::function<XrtPass *()>>::Global()->Lookup(pass)();
  optimize_pass->Run(graph, options);
}

template <typename... Args>
inline void RunPassImpl(const std::string &pass, XrtGraph *graph,
                        const XrtPassOptions &options, Args &&... args) {
  std::vector<Any> params{std::forward<Args>(args)...};
  auto optimize_pass =
      util::Registry<std::function<XrtPass *()>>::Global()->Lookup(pass)();
  optimize_pass->Run(graph, options, params);
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_PASS_XLA_OPTIMIZE_PASS_H_
