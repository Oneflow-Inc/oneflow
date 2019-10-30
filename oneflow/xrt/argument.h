#ifndef ONEFLOW_XRT_ARGUMENT_H_
#define ONEFLOW_XRT_ARGUMENT_H_

#include <string>

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace xrt {

class ArgumentMetaData {
 public:
  ArgumentMetaData() = default;
  ArgumentMetaData(const std::string &key1, const std::string &key2)
      : produce_key_(key1), consume_key_(key2) {}

  virtual ~ArgumentMetaData() = default;

  void set_produce_key(const std::string &key) { produce_key_ = key; }
  void set_consume_key(const std::string &key) { consume_key_ = key; }

  const std::string &produce_key() const { return produce_key_; }
  const std::string &consume_key() const { return consume_key_; }

 private:
  std::string produce_key_;
  std::string consume_key_;
};

class Argument {
 public:
  Argument() : initialized_(false) {}

  explicit Argument(const std::string &name)
      : Argument(name, ArgumentMetaData()) {}

  explicit Argument(const std::string &name, const Shape &shape,
                    const DataType &data_type)
      : Argument(name, shape, data_type, ArgumentMetaData()) {}

  explicit Argument(const std::string &name, const ArgumentMetaData &meta_data)
      : arg_name_(name), meta_data_(meta_data), initialized_(true) {}

  explicit Argument(const std::string &name, const Shape &shape,
                    const DataType &data_type,
                    const ArgumentMetaData &meta_data)
      : arg_name_(name),
        shape_(shape),
        data_type_(data_type),
        meta_data_(meta_data),
        initialized_(true) {}

  const std::string &name() const { return arg_name_; }

  const Shape &shape() const { return shape_; }
  const DataType &data_type() const { return data_type_; }

  void set_meta_data(const ArgumentMetaData &meta_data) {
    meta_data_ = meta_data;
  }
  const ArgumentMetaData &meta_data() const { return meta_data_; }

  bool initialized() const { return initialized_; }

  bool operator==(const Argument &rhs) const {
    return arg_name_ == rhs.arg_name_ && shape_ == rhs.shape_ &&
           data_type_ == rhs.data_type_;
  }

 private:
  std::string arg_name_{""};
  Shape shape_;
  DataType data_type_;

  ArgumentMetaData meta_data_;

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
