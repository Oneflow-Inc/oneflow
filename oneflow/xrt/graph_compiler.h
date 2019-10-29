#ifndef ONEFLOW_XRT_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_GRAPH_COMPILER_H_

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"

namespace oneflow {
namespace xrt {

class GraphCompiler {
 public:
  GraphCompiler(const XrtEngine &engine) : engine_(engine) {
    impl_.reset(GraphCompiler::Registry()->Lookup(engine_)());
    CHECK(impl_) << "Internal compiler should be built correctly for engine "
                 << engine_;
  }

  std::shared_ptr<Executable> Compile(
      const XrtGraph *graph, const std::vector<Parameter> &entry_params,
      const std::vector<Parameter> &return_params,
      const std::vector<InputOutputAlias> &aliases) {
    return impl_->Compile(graph, entry_params, return_params, aliases);
  }

  class Impl {
   public:
    Impl() = default;
    virtual ~Impl() = default;
    virtual std::shared_ptr<Executable> Compile(
        const XrtGraph *graph, const std::vector<Parameter> &entry_params,
        const std::vector<Parameter> &return_params,
        const std::vector<InputOutputAlias> &aliases) = 0;
  };

  static auto Registry()
      -> util::Registry<XrtEngine, std::function<Impl *()>> * {
    return util::Registry<XrtEngine, std::function<Impl *()>>::Global();
  }

 private:
  XrtEngine engine_;
  std::shared_ptr<Impl> impl_;
};

#define REGISTER_GRAPH_COMPILER(Engine, Compiler)                          \
  namespace {                                                              \
  struct _XrtGraphCompiler {                                               \
    _XrtGraphCompiler() {                                                  \
      GraphCompiler::Registry()->Register(                                 \
          Engine, []() -> GraphCompiler::Impl * { return new Compiler; }); \
    }                                                                      \
  };                                                                       \
  static _XrtGraphCompiler _xrt_graph_compiler_ __attribute__((unused));   \
  }  // namespace

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_GRAPH_COMPILER_H_
