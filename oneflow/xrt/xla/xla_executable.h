#ifndef ONEFLOW_XRT_XLA_XLA_EXECUTABLE_H_
#define ONEFLOW_XRT_XLA_XLA_EXECUTABLE_H_

#include "oneflow/xrt/executable.h"
#include "tensorflow/compiler/xla/client/local_client.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaExecutable : public Executable {
 public:
  XlaExecutable(const std::string &name, const XrtDevice &device,
                const std::vector<xla::Shape> &input_shapes,
                const xla::Shape &output_shape, std::unique_ptr<xla::LocalExecutable> &&executable)
      : Executable(name, XrtEngine::XLA),
        device_(device),
        input_shapes_(input_shapes),
        output_shape_(output_shape),
        executable_(std::move(executable)) {}

  virtual ~XlaExecutable() = default;

  bool Run(const std::vector<Parameter> &inputs, const ExecutableRunOptions &run_options,
           bool block_until_done = true) override;

 private:
  XrtDevice device_;

  std::vector<xla::Shape> input_shapes_;
  // The output shape is always a tuple.
  xla::Shape output_shape_;

  std::unique_ptr<xla::LocalExecutable> executable_;
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_EXECUTABLE_H_
