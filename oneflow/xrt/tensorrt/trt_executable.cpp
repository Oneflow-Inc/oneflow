#include "oneflow/xrt/tensorrt/trt_executable.h"

namespace oneflow {
namespace xrt {

namespace tensorrt {

bool TrtExecutable::Run(const std::vector<Parameter> &inputs,
                        const ExecutableRunOptions &run_options,
                        bool block_until_done) {
  // TODO(hjchen2)
}

}  // namespace tensorrt

}  // namespace xrt
}  // namespace oneflow
