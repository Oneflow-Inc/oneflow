#ifndef ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_ATTR_H_
#define ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_ATTR_H_

#include <unordered_map>
#include <unordered_set>

#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {
namespace mola {

class LaunchAttrHelper {
 public:
  LaunchAttrHelper(const XlaLaunchOpConf::Attribute &attr) {
    for (const auto &argument : attr.argument()) {
      args_.emplace(argument.in(), argument.out());
      if (argument.is_mutable()) {
        mutable_args_.insert(argument.in());
      }
    }
  }

  bool IsMutableArg(const std::string &arg_name) const {
    return mutable_args_.count(arg_name) > 0;
  }

  std::string OutputArg(const std::string &arg_name) const {
    return args_.at(arg_name);
  }

  const std::unordered_map<std::string, std::string> &args() const {
    return args_;
  }

  const std::unordered_set<std::string> &mutable_args() const {
    return mutable_args_;
  }

 private:
  std::unordered_map<std::string, std::string> args_;
  std::unordered_set<std::string> mutable_args_;
};

}  // namespace mola
}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_XLA_OF2XLA_XLA_LAUNCH_ATTR_H_
