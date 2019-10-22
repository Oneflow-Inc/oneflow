#ifndef ONEFLOW_XRT_IR_ARGUMENT_H_
#define ONEFLOW_XRT_IR_ARGUMENT_H_

#include "oneflow/xrt/shape.h"

namespace oneflow {
namespace xrt {

class XrtArgument {
 public:
  XrtArgument() : has_shape_(false), initialized_(false) {}

  XrtArgument(const std::string &name)
      : name_(name), has_shape_(false), initialized_(true) {}

  XrtArgument(const std::string &name, const XrtShape &shape)
      : name_(name), shape_(shape), has_shape_(true), initialized_(true) {}

  const std::string &name() const { return name_; }

  bool has_shape() const { return has_shape_; }

  const XrtShape &shape() const { return shape_; }

  bool operator==(const XrtArgument &rhs) const {
    return name_ == rhs.name_ && shape_ == rhs.shape_;
  }

  bool is_initialized() const { return initialized_; }

 private:
  std::string name_ = "";

  XrtShape shape_;

  bool has_shape_ = false;

  bool initialized_ = false;
};

}  // namespace xrt
}  // namespace oneflow

namespace std {
template <>
struct hash<oneflow::xrt::XrtArgument> {
  size_t operator()(const oneflow::xrt::XrtArgument &arg) const {
    return std::hash<std::string>()(arg.name());
  }
};
}  // namespace std

#endif  // ONEFLOW_XRT_IR_ARGUMENT_H_
