#ifndef ONEFLOW_XRT_PARAMETER_H_
#define ONEFLOW_XRT_PARAMETER_H_

#include <string>

#include "oneflow/xrt/shape.h"

namespace oneflow {
namespace xrt {

class Parameter {
 public:
  Parameter() = default;
  virtual ~Parameter() = default;

  Parameter(void *data, const XrtShape &shape)
      : storage_(data), shape_(shape) {}

  Parameter(void *data, const XrtShape &shape, const std::string &name)
      : storage_(data), shape_(shape), parameter_name_(name) {}

  void set_data(const void *data) { storage_ = const_cast<void *>(data); }

  template <typename T>
  void set_data(const T *data) {
    storage_ = const_cast<T *>(data);
  }

  void *data() const { return storage_; }

  template <typename T>
  T *data() const {
    return reinterpret_cast<T *>(storage_);
  }

  const XrtShape &shape() const { return shape_; }
  void set_shape(const XrtShape &shape) { shape_ = shape; }
  const std::string &name() const { return parameter_name_; }
  void set_name(const std::string &name) { parameter_name_ = name; }

 private:
  void *storage_ = nullptr;
  XrtShape shape_;
  std::string parameter_name_{""};
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_PARAMETER_H_
