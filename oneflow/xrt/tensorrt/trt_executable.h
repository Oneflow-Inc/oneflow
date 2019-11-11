#ifndef ONEFLOW_XRT_TENSORRT_TRT_EXECUTABLE_H_
#define ONEFLOW_XRT_TENSORRT_TRT_EXECUTABLE_H_

#include <vector>
#include "NvInfer.h"

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/tensorrt/trt_unique_ptr.h"

namespace oneflow {
namespace xrt {

namespace tensorrt {

class TrtExecutable : public Executable {
 public:
  explicit TrtExecutable(nv::unique_ptr<nvinfer1::ICudaEngine> &&engine)
      : Executable(XrtEngine::TENSORRT), engine_(std::move(engine)) {}

  virtual ~TrtExecutable() = default;

  bool Run(const std::vector<Parameter> &inputs,
           const ExecutableRunOptions &run_options,
           bool block_until_done = true) override;

 private:
  nv::unique_ptr<nvinfer1::ICudaEngine> engine_;
};

}  // namespace tensorrt

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_EXECUTABLE_H_
