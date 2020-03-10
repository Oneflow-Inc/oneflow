#ifndef ONEFLOW_XRT_TVM_TVM_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_TVM_TVM_GRAPH_COMPILER_H_

#include "oneflow/xrt/graph_compiler.h"
#include <tvm/build_module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>

namespace oneflow {
namespace xrt {

class TVMGraphCompiler final : public GraphCompiler::Impl {
 public:
  explicit TVMGraphCompiler(const std::string& name);
  virtual ~TVMGraphCompiler() = default;


  std::shared_ptr<Executable> Compile(const XrtGraph *graph,
                                      const std::vector<Parameter> &entry_params,
                                      const std::vector<Parameter> &return_params,
                                      const std::vector<InputOutputAlias> &aliases) override;
  
 private:
   tvm::runtime::Module builder_;
};

}

}

#endif // ONEFLOW_XRT_TVM_TVM_GRAPH_COMPILER_H_
