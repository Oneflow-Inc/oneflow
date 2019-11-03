#ifndef ONEFLOW_XRT_LAUNCH_UTIL_H_
#define ONEFLOW_XRT_LAUNCH_UTIL_H_

#include "absl/strings/str_split.h"

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

  std::string Input(const std::string &arg_name) const {
    return input_args_.at(arg_name);
  }

  std::string Output(const std::string &arg_name) const {
    return output_args_.at(arg_name);
  }

  bool LookupMutability(const std::string &arg_name) const {
    return mutable_args_.count(arg_name) > 0;
  }

 private:
  // All input arguments.
  util::Map<std::string, std::string> input_args_;
  // All output arguments.
  util::Map<std::string, std::string> output_args_;
  // mutable argument names
  util::Set<std::string> mutable_args_;
};

extern const std::string _XrtInArgumentPrefix;

LaunchGraphHelper::LaunchGraphHelper(const XrtLaunchOpConf::Attribute &attr) {
  const auto &mutability_table = attr.mutability();
  for (const auto &argument : attr.argument()) {
    if (absl::StartsWith(argument.name(), _XrtInArgumentPrefix)) {
      input_args_.emplace(argument.in(), argument.out());
      if (mutability_table.count(argument.name())) {
        mutable_args_.insert(argument.in());
      }
    } else {
      output_args_.emplace(argument.out(), argument.in());
    }
  }
}

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_LAUNCH_UTIL_H_
