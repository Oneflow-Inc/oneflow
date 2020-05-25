#ifndef ONEFLOW_CORE_COMMON_REGISTER_TENSOR_VIEW_H_
#define ONEFLOW_CORE_COMMON_REGISTER_TENSOR_VIEW_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/memory/memory_case.pb.h"

namespace oneflow {

class Blob;
int64_t NumAxes4Blob(const Blob* blob);
const MemoryCase& MemCase4Blob(const Blob* blob);
DataType DataType4Blob(const Blob* blob);

template<typename ShapeViewType, typename ByteType>
class TensorViewBase {
 public:
  const ShapeViewType& shape() const { return shape_; }
  const MemoryCase& mem_case() const { return MemCase4Blob(blob_); }
  DataType data_type() const { return DataType4Blob(blob_); }
  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>(data_type());
    return reinterpret_cast<const T*>(dptr_);
  }
  size_t ByteSize() const { return shape().elem_cnt() * GetSizeOfDataType(data_type()); }
  void reset(typename ShapeViewType::DimType* shape_ptr, ByteType* dptr) {
    shape_.set_ptr(shape_ptr);
    dptr_ = dptr;
  }

 protected:
  TensorViewBase(const TensorViewBase&) = default;
  TensorViewBase(const Blob* blob, typename ShapeViewType::DimType* shape_ptr, ByteType* dptr)
      : blob_(blob), shape_(shape_ptr, NumAxes4Blob(blob)), dptr_(dptr) {}
  ~TensorViewBase() = default;

  const Blob* blob() const { return blob_; }
  ByteType* mem_dptr() const { return dptr_; }
  ShapeViewType* shape_view_ptr() { return &shape_; }

 private:
  const Blob* blob_;
  ShapeViewType shape_;
  ByteType* dptr_;
};

class TensorView final : public TensorViewBase<ShapeView, const char> {
 public:
  TensorView(const TensorView&) = default;
  TensorView(const Blob* blob, const int64_t* shape_ptr, const char* dptr)
      : TensorViewBase<ShapeView, const char>(blob, shape_ptr, dptr) {}
  ~TensorView() = default;
};

template<typename ShapeViewType>
class MutTensorView : public TensorViewBase<ShapeViewType, char> {
 public:
  template<typename T = void>
  T* mut_dptr() const {
    CheckDataType<T>(this->data_type());
    return reinterpret_cast<T*>(this->mem_dptr());
  }

 protected:
  MutTensorView(const MutTensorView&) = default;
  MutTensorView(const Blob* blob, typename ShapeViewType::DimType* shape_ptr, char* dptr)
      : TensorViewBase<ShapeViewType, char>(blob, shape_ptr, dptr) {}
  ~MutTensorView() = default;
};

class DataOnlyMutTensorView final : public MutTensorView<ShapeView> {
 public:
  DataOnlyMutTensorView(const DataOnlyMutTensorView&) = default;
  DataOnlyMutTensorView(const Blob* blob, const int64_t* shape_ptr, char* dptr)
      : MutTensorView<ShapeView>(blob, shape_ptr, dptr) {}
  ~DataOnlyMutTensorView() = default;
};

class Shape;

class FullyMutTensorView final : public MutTensorView<MutShapeView> {
 public:
  FullyMutTensorView(const FullyMutTensorView&) = default;
  FullyMutTensorView(const Blob* blob, int64_t* shape_ptr, char* dptr);
  ~FullyMutTensorView() = default;

  void set_shape(const Shape& shape);
  void set_shape(const ShapeView& shape);

 private:
  void CheckCapacity(size_t shape_elem_cnt) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_REGISTER_TENSOR_VIEW_H_
