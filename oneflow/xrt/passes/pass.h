/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_XRT_PASSES_PASS_H_
#define ONEFLOW_XRT_PASSES_PASS_H_

#include "oneflow/xrt/any.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/utility/registry.h"

namespace oneflow {
namespace xrt {

typedef int64_t XrtEngineOptionBitFlags;

enum XrtEngineOptionBit : int {
  kUseDefault = 0,
  kUseXlaJit = 1,
  kUseTensorRT = 2,
};

struct ClusteringOptions {
  // Minimum node number in each cluster after clustering. If the number of
  // nodes contained by a cluster is less than `minimum_nodes` or grater than
  // `maximum_nodes`, then this cluster will be given up and not compiled.
  int32_t minimum_nodes = 0x1;
  int32_t maximum_nodes = 0x7fffffff;

  // Option to clustering with strict dependencies analysis.
  bool strict_clustering = true;

  // Maximum iteration count for iteratively clustering. You can set it -1 means
  // that it will always iteratively merge nodes as much as possible until no
  // nodes can be merged.
  int32_t max_iteration = 20;

  // Clustering nodes if it's compiled by the engine.
  // XrtEngine engine = XrtEngine::XLA;
  XrtEngineOptionBitFlags engine = 0x0;

  // Clustering subgraph for train phase.
  bool train_phase = true;

  bool ignore_sbp_policy = false;
  bool ignore_time_shape = false;
};

bool CheckUseXrtEngine(const ClusteringOptions &options, const XrtEngine &engine);

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

  virtual void Run(XrtGraph *graph, const XrtPassOptions &options, const std::vector<Any> &params) {
    LOG(FATAL) << "Should not call this function.";
  }

  static auto Registry() -> util::Registry<std::string, std::function<XrtPass *()>> * {
    return util::Registry<std::string, std::function<XrtPass *()>>::Global();
  }
};

// typedef XrtPass *(*XrtPassCreator)();

#define REGISTER_XRT_PASS(PassName, PassType)                                               \
  namespace PassName {                                                                      \
  struct _XrtPassRegistrar {                                                                \
    _XrtPassRegistrar() {                                                                   \
      XrtPass::Registry()->Register(#PassName, []() -> XrtPass * { return new PassType; }); \
    }                                                                                       \
  };                                                                                        \
  _XrtPassRegistrar _xrt_pass_registrar_ __attribute__((unused));                           \
  }  // namespace // PassName

inline void RunPassImpl(const std::string &pass, XrtGraph *graph, const XrtPassOptions &options) {
  auto optimize_pass = std::shared_ptr<XrtPass>(XrtPass::Registry()->Lookup(pass)());
  optimize_pass->Run(graph, options);
}

template<typename... Args>
inline void RunPassImpl(const std::string &pass, XrtGraph *graph, const XrtPassOptions &options,
                        Args &&... args) {
  std::vector<Any> params{std::forward<Args>(args)...};
  auto optimize_pass = std::shared_ptr<XrtPass>(XrtPass::Registry()->Lookup(pass)());
  optimize_pass->Run(graph, options, params);
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_PASSES_PASS_H_
