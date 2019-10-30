#ifndef ONEFLOW_XRT_EXECUTABLE_H_
#define ONEFLOW_XRT_EXECUTABLE_H_

#include <vector>

#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/xrt.pb.h"

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

struct ExecutableRunOptions {
  // Specify stream if the engine supports multiple computation streams.
  // It will use the default computation stream if `stream` is not set.
  void *stream = nullptr;

  // Set host threads num.
  int32_t host_num_threads = -1;

  // Limit memory footprint.
  int64_t host_memory_limit = -1;
  int64_t device_memory_limit = -1;

  // Random seed.
  int64_t random_seed = -1;

  // Feed the return parameters to reuse it's storage while running
  // the executable.
  std::vector<Parameter> return_params;
};

class Executable {
 public:
  Executable(const XrtEngine &engine) : engine_(engine) {}
  virtual ~Executable() = default;

  const XrtEngine &engine() const { return engine_; }

  virtual bool Run(const std::vector<Parameter> &inputs,
                   const ExecutableRunOptions &run_options,
                   bool block_until_done = true) = 0;

  bool RunAsync(const std::vector<Parameter> inputs,
                const ExecutableRunOptions &run_options) {
    return Run(inputs, run_options, false);
  }

  const std::vector<Parameter> &Results() const { return results_; }

 protected:
  XrtEngine engine_;
  std::vector<Parameter> results_;
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_EXECUTABLE_H_
