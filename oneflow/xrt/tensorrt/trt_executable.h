#ifndef ONEFLOW_XRT_TENSORRT_TRT_EXECUTABLE_H_
#define ONEFLOW_XRT_TENSORRT_TRT_EXECUTABLE_H_

#include <vector>
#include "NvInfer.h"

#include "oneflow/xrt/executable.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/tensorrt/trt_unique_ptr.h"
#include "oneflow/xrt/tensorrt/trt_int8_calibrator.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

namespace tensorrt {

class TrtExecutable : public Executable {
 public:
  explicit TrtExecutable(
      const std::string &name, nv::unique_ptr<nvinfer1::ICudaEngine> &&engine,
      const util::Map<std::string, std::shared_ptr<std::vector<uint8_t>>> &host_weights)
      : Executable(name, XrtEngine::TENSORRT),
        engine_(std::move(engine)),  // NOLINT
        host_weights_(host_weights) {}

  explicit TrtExecutable(
      const std::string &name, nv::unique_ptr<nvinfer1::IBuilder> &&builder,
      nv::unique_ptr<nvinfer1::INetworkDefinition> &&network,
      const util::Map<std::string, std::shared_ptr<std::vector<uint8_t>>> &host_weights)
      : Executable(name, XrtEngine::TENSORRT),
        builder_(std::move(builder)),
        network_(std::move(network)),
        host_weights_(host_weights) {}

  virtual ~TrtExecutable() = default;

  bool Run(const std::vector<Parameter> &inputs, const ExecutableRunOptions &run_options,
           bool block_until_done = true) override;

 private:
  nvinfer1::ICudaEngine *CreateExecutableEngine(const ExecutableRunOptions &run_options,
                                                const int batch_size = 1,
                                                TRTInt8Calibrator *calibrator = nullptr);

  bool ExecuteEngine(const int batch_size, void **buffers, void *stream, bool block_until_done);

  std::string LoadCalibrationTable(const std::string &calibration_path);

 private:
  nv::unique_ptr<nvinfer1::ICudaEngine> engine_;
  nv::unique_ptr<nvinfer1::IBuilder> builder_;
  nv::unique_ptr<nvinfer1::INetworkDefinition> network_;
  nv::unique_ptr<nvinfer1::IExecutionContext> execution_context_;

  std::shared_ptr<TRTInt8Calibrator> calibrator_;

  util::Map<std::string, std::shared_ptr<std::vector<uint8_t>>> host_weights_;
};

}  // namespace tensorrt

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TENSORRT_TRT_EXECUTABLE_H_
