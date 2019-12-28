#ifndef ONEFLOW_XRT_ARGUMENT_H_
#define ONEFLOW_XRT_ARGUMENT_H_

#include <string>

#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {
namespace xrt {

// Each data flow will bind two keys, produce_key and consume_key.
// Such as node A and B, there is a data flow named `a_output` on edge A->B.
//  node A {
//    in: "a_input"
//    out: "a_output"
//  }
//  node B {
//    in: "a_output"
//    out: "b_output"
//  }
// In this case, the data flow named `a_output` has a `produce_key` named
// \"out\" produced by node A and a `consume_key` named \"in\" consumed by
// node B.
struct ArgumentMetaData {
  std::string produce_key;
  std::string consume_key;
};

// Descriptor of data flow on graph edges include data name, shape and
// data type. Also it may be attached by a metadata which giving the key
// of producing and consuming.
class Argument {
 public:
  Argument() : initialized_(false) {}

  explicit Argument(const std::string &name) : Argument(name, ArgumentMetaData()) {}

  explicit Argument(const std::string &name, const Shape &shape, const DataType &data_type)
      : Argument(name, shape, data_type, ArgumentMetaData()) {}

  explicit Argument(const std::string &name, const ArgumentMetaData &meta_data)
      : arg_name_(name), meta_data_(meta_data), initialized_(true) {}

  explicit Argument(const std::string &name, const Shape &shape, const DataType &data_type,
                    const ArgumentMetaData &meta_data)
      : arg_name_(name),
        shape_(shape),
        data_type_(data_type),
        meta_data_(meta_data),
        initialized_(true) {}

  const std::string &name() const { return arg_name_; }

  const Shape &shape() const { return shape_; }
  const DataType &data_type() const { return data_type_; }

  void set_meta_data(const ArgumentMetaData &meta_data) { meta_data_ = meta_data; }
  const ArgumentMetaData &meta_data() const { return meta_data_; }

  bool initialized() const { return initialized_; }

  bool operator==(const Argument &rhs) const {
    return arg_name_ == rhs.arg_name_ && shape_ == rhs.shape_ && data_type_ == rhs.data_type_;
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
template<>
struct hash<oneflow::xrt::Argument> {
  size_t operator()(const oneflow::xrt::Argument &arg) const {
    return std::hash<std::string>()(arg.name());
  }
};
}  // namespace std

#endif  // ONEFLOW_XRT_GRAPH_ARGUMENT_H_
