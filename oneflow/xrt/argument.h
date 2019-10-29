#ifndef ONEFLOW_XRT_ARGUMENT_H_
#define ONEFLOW_XRT_ARGUMENT_H_

#include <string>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace xrt {

class Argument {
 public:
  Argument() : initialized_(false) {}

  Argument(const std::string &name) : name_(name), initialized_(true) {}

  Argument(const std::string &name, const Shape &shape,
           const DataType &data_type)
      : name_(name), shape_(shape), data_type_(data_type), initialized_(true) {}

  const std::string &name() const { return name_; }

  const Shape &shape() const { return shape_; }
  const DataType &data_type() const { return data_type_; }

  bool operator==(const Argument &rhs) const {
    return name_ == rhs.name_ && shape_ == rhs.shape_;
  }

  bool initialized() const { return initialized_; }

 private:
  std::string name_ = "";

  Shape shape_;
  DataType data_type_;

  bool initialized_ = false;
};

}  // namespace xrt
}  // namespace oneflow

namespace std {
template <>
struct hash<oneflow::xrt::Argument> {
  size_t operator()(const oneflow::xrt::Argument &arg) const {
    return std::hash<std::string>()(arg.name());
  }
};
}  // namespace std

#endif  // ONEFLOW_XRT_GRAPH_ARGUMENT_H_
