#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_DESC_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_DESC_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/blob_desc.pb.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

namespace user_op {

class TensorDesc final {
 public:
  TensorDesc() = default;
  ~TensorDesc() = default;
  TensorDesc(const TensorDesc&);
  TensorDesc(const BlobDescProto&);

  TensorDesc& operator=(const TensorDesc&);
  TensorDesc& operator=(const BlobDescProto&);

  const Shape& shape() const { return shape_; }
  Shape* mut_shape() { return &shape_; }
  DataType data_type() const { return data_type_; }
  DataType* mut_data_type() { return &data_type_; }

  bool is_dynamic() const { return is_dynamic_; }
  bool* mut_is_dynamic() { return &is_dynamic_; }
  void set_is_dynamic(bool val) { is_dynamic_ = val; }
  bool is_tensor_list() const { return is_tensor_list_; }
  bool* mut_is_tensor_list() { return &is_tensor_list_; }
  void set_is_tensor_list(bool val) { is_tensor_list_ = val; }

 private:
  Shape shape_;
  DataType data_type_;
  bool is_dynamic_;
  bool is_tensor_list_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_DESC_H_
