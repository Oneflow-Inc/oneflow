#ifndef ONEFLOW_XRT_TVM_TVM_EXECUTABLE_H_
#define ONEFLOW_XRT_TVM_TVM_EXECUTABLE_H_

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>

namespace oneflow {

namespace xrt {

class TVMExecutable final : public Executable {
 public:
  TVMExecutable(const std::string& name, const int num_inputs,
      const std::vector<Parameter>& outputs,
      const std::string& json,
      const tvm::runtime::Module& built_mod,
      TVMContext ctx);

  bool Run(const std::vector<Parameter> &inputs, const ExecutableRunOptions &run_options,
                   bool block_until_done = true) override;

 private:
  std::string name_;
  int num_inputs_;
  std::vector<DLManagedTensor> outputs_;

  TVMContext ctx_;
  tvm::runtime::Module executor_;
  tvm::runtime::PackedFunc set_input_zero_copy_;
  tvm::runtime::PackedFunc run_;
  tvm::runtime::PackedFunc get_output_;
  tvm::runtime::PackedFunc get_num_outputs_;
};

}

}

#endif // ONEFLOW_XRT_TVM_TVM_EXECUTABLE_H_
