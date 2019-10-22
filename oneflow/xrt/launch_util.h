#ifndef ONEFLOW_XRT_LAUNCH_UTIL_H_
#define ONEFLOW_XRT_LAUNCH_UTIL_H_

#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/xrt/utility/stl.h"

namespace oneflow {
namespace xrt {

// `LaunchGraphHelper` is an auxiliary class for xrt graph since we can access
// the actual input and output names throught class `LaunchGraphHelper`.
class LaunchGraphHelper {
 public:
  LaunchGraphHelper() = delete;
  inline explicit LaunchGraphHelper(const XrtLaunchOpConf::Attribute &attr);
  virtual ~LaunchGraphHelper() {}

  inline std::string Input(const std::string &arg_name) const;

  inline std::string Output(const std::string &arg_name) const;

 private:
  // All input arguments.
  util::Map<std::string, std::string> input_args_;
  // All output arguments.
  util::Map<std::string, std::string> output_args_;
};

extern const std::string _XrtInArgumentPrefix;

LaunchGraphHelper::LaunchGraphHelper(const XrtLaunchOpConf::Attribute &attr) {
  for (const auto &argument : attr.argument()) {
    if (absl::StartsWith(argument.name(), _XrtInArgumentPrefix)) {
      input_args_.emplace(argument.in(), argument.out());
    } else {
      output_args_.emplace(argument.out(), argument.in());
    }
  }
}

std::string LaunchGraphHelper::Input(const std::string &arg_name) const {
  DCHECK_GT(input_args_.count(arg_name), 0);
  return input_args_.at(arg_name);
}

std::string LaunchGraphHelper::Output(const std::string &arg_name) const {
  DCHECK_GT(output_args_.count(arg_name), 0);
  return output_args_.at(arg_name);
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_LAUNCH_UTIL_H_
