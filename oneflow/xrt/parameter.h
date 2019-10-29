#ifndef ONEFLOW_XRT_PARAMETER_H_
#define ONEFLOW_XRT_PARAMETER_H_

#include <string>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace xrt {

class Parameter {
 public:
  Parameter() = default;
  virtual ~Parameter() = default;

  Parameter(void *data, const Shape &shape, const DataType &data_type)
      : storage_(data), shape_(shape), data_type_(data_type) {}

  Parameter(const std::string &name, void *data, const Shape &shape,
            const DataType &data_type)
      : storage_(data),
        shape_(shape),
        data_type_(data_type),
        parameter_name_(name) {}

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

  const std::string &name() const { return parameter_name_; }
  const Shape &shape() const { return shape_; }
  const DataType &data_type() const { return data_type_; }

  void set_name(const std::string &name) { parameter_name_ = name; }
  void set_shape(const Shape &shape) { shape_ = shape; }
  void set_data_type(const DataType &data_type) { data_type_ = data_type; }

 private:
  void *storage_ = nullptr;
  Shape shape_;
  DataType data_type_;
  std::string parameter_name_{""};
};

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_PARAMETER_H_
