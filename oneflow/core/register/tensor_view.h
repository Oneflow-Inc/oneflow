#ifndef ONEFLOW_CORE_COMMON_REGISTER_TENSOR_VIEW_H_
#define ONEFLOW_CORE_COMMON_REGISTER_TENSOR_VIEW_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/register/dense_shape_view.h"

namespace oneflow {

class TensorView final {
 public:
  OF_DISALLOW_COPY(TensorView);
  TensorView(TensorView&&) = default;
  TensorView(const int64_t* shape_ptr, int64_t num_axes, DataType data_type, const void* dptr)
      : shape_(shape_ptr, num_axes), data_type_(data_type), dptr_(dptr) {}
  ~TensorView() = default;

  const DenseShapeView& shape() const { return shape_; }
  const int64_t* shape_ptr() const { return shape_.ptr(); }
  DataType data_type() const { return data_type_; }
  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>(data_type_);
    return static_cast<const T*>(dptr_);
  }

 private:
  const DenseShapeView shape_;
  const DataType data_type_;
  const void* dptr_;
};

class MutTensorView {
 public:
  const DenseShapeView& shape() const { return shape_; }
  DataType data_type() const { return data_type_; }

  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>(data_type_);
    return static_cast<const T*>(dptr_);
  }

  template<typename T = void>
  T* mut_dptr() const {
    CheckDataType<T>(data_type_);
    return static_cast<T*>(dptr_);
  }

 protected:
  OF_DISALLOW_COPY(MutTensorView);
  MutTensorView(MutTensorView&&) = default;
  MutTensorView(const int64_t* shape_ptr, int64_t num_axes, DataType data_type, void* dptr)
      : shape_(shape_ptr, num_axes), data_type_(data_type), dptr_(dptr) {}
  ~MutTensorView() = default;

 private:
  DenseShapeView shape_;
  const DataType data_type_;
  void* dptr_;
};

class DataOnlyMutTensorView final : public MutTensorView {
 public:
  OF_DISALLOW_COPY(DataOnlyMutTensorView);
  DataOnlyMutTensorView(DataOnlyMutTensorView&&) = default;
  DataOnlyMutTensorView(const int64_t* shape_ptr, int64_t num_axes, DataType data_type, void* dptr)
      : MutTensorView(shape_ptr, num_axes, data_type, dptr) {}
  ~DataOnlyMutTensorView() = default;

  const int64_t* shape_ptr() const { return shape().ptr(); }
};

class Shape;

class FullyMutTensorView final : public MutTensorView {
 public:
  OF_DISALLOW_COPY(FullyMutTensorView);
  FullyMutTensorView(FullyMutTensorView&&) = default;
  FullyMutTensorView(int64_t* shape_ptr, int64_t num_axes, DataType data_type, void* dptr,
                     size_t data_capacity);
  ~FullyMutTensorView() = default;

  int64_t* mut_shape_ptr() const { return mut_shape_.mut_ptr(); }

  void set_shape(const Shape& shape);
  void set_shape(const DenseShapeView& shape);

 private:
  DenseShapeMutView mut_shape_;
  size_t max_elem_cnt_;
};

class TensorListView final {
 public:
  TensorListView(int64_t size, const int64_t* shape_ptr, int64_t num_axes, DataType data_type,
                 const void* dptr);
  ~TensorListView() = default;

  size_t size() const { return tensors_->size(); }
  DataType data_type() const { return data_type_; }
  const TensorView& tensor(int32_t i) const { return tensors_->at(i); }

 private:
  const DataType data_type_;
  std::shared_ptr<std::vector<TensorView>> tensors_;
};

class MutTensorListView final {
 public:
  MutTensorListView(int64_t size, const int64_t* shape_ptr, int64_t num_axes, DataType data_type,
                    void* dptr);
  ~MutTensorListView() = default;

  size_t size() const { return tensors_->size(); }
  DataType data_type() const { return data_type_; }
  DataOnlyMutTensorView* mutable_tensor(int32_t i) const { return &tensors_->at(i); }

 private:
  const DataType data_type_;
  std::shared_ptr<std::vector<DataOnlyMutTensorView>> tensors_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_REGISTER_TENSOR_VIEW_H_
