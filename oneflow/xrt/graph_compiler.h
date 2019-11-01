#ifndef ONEFLOW_XRT_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_GRAPH_COMPILER_H_

#include <functional>

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/types.h"
#include "oneflow/xrt/utility/registry.h"

namespace oneflow {
namespace xrt {

class InputOutputAlias {
 public:
  InputOutputAlias(const std::vector<int> &output_index, int param_number,
                   const std::vector<int> &param_index)
      : param_number_(param_number),
        param_index_(param_index),
        output_index_(output_index) {}

  int param_number() const { return param_number_; }

  const std::vector<int> &param_index() const { return param_index_; }

  const std::vector<int> &output_index() const { return output_index_; }

 private:
  int param_number_;
  std::vector<int> param_index_;
  std::vector<int> output_index_;
};

class GraphCompiler {
 public:
  // Internal compiler interface class. It should be inherited and the
  // `Compile` function should be overrided for every engine compiler.
  class Impl {
   public:
    void set_name(const std::string &name) { name_ = name; }

    void set_device(XrtDevice device) { device_ = device; }

    void set_device_ordinal(int32_t device_ordinal) {
      device_ordinal_ = device_ordinal;
    }

    virtual std::shared_ptr<Executable> Compile(
        const XrtGraph *graph, const std::vector<Parameter> &entry_params,
        const std::vector<Parameter> &return_params,
        const std::vector<InputOutputAlias> &aliases) = 0;

   protected:
    // Compiler name
    std::string name_ = "";

    XrtDevice device_;
    int32_t device_ordinal_ = 0;
  };

 public:
  static auto Registry()
      -> util::Registry<XrtEngine, std::function<Impl *()>> * {
    return util::Registry<XrtEngine, std::function<Impl *()>>::Global();
  }

  explicit GraphCompiler(const std::string &name, const XrtEngine &engine,
                         const XrtDevice &device, int32_t device_ordinal)
      : engine_(engine) {
    impl_.reset(GraphCompiler::Registry()->Lookup(engine_)());
    CHECK(impl_) << "Internal compiler should be built correctly for engine "
                 << engine_;
    impl_->set_name(name);
    impl_->set_device(device);
    impl_->set_device_ordinal(device_ordinal);
  }

  void set_devide(XrtDevice device) { impl_->set_device(device); }

  void set_device_ordinal(int32_t device_ordinal) {
    CHECK_GE(device_ordinal, 0) << "Device ordinal should >= 0.";
    impl_->set_device_ordinal(device_ordinal);
  }

  std::shared_ptr<Executable> Compile(
      const XrtGraph *graph, const std::vector<Parameter> &entry_params,
      const std::vector<Parameter> &return_params,
      const std::vector<InputOutputAlias> &aliases) {
    return impl_->Compile(graph, entry_params, return_params, aliases);
  }

 private:
  GraphCompiler() = delete;

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
