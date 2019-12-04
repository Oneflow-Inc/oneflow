#include "oneflow/core/register/tensor_view.h"
#include "oneflow/core/common/shape.h"

namespace oneflow {

FullyMutTensorView::FullyMutTensorView(int64_t* shape_ptr, int64_t num_axes, DataType data_type,
                                       void* dptr, size_t data_capacity)
    : MutTensorView(shape_ptr, num_axes, data_type, dptr),
      mut_shape_(shape_ptr, num_axes),
      max_elem_cnt_(data_capacity / GetSizeOfDataType(data_type)) {
  CHECK_EQ(data_capacity % GetSizeOfDataType(data_type), 0);
}

void FullyMutTensorView::set_shape(const Shape& shape) {
  CHECK_LE(shape.elem_cnt(), max_elem_cnt_);
  mut_shape_.set_shape(shape);
}

void FullyMutTensorView::set_shape(const DenseShapeView& shape) {
  CHECK_LE(shape.elem_cnt(), max_elem_cnt_);
  mut_shape_.set_shape(shape);
}

TensorListView::TensorListView(int64_t size, const int64_t* shape_ptr, int64_t num_axes,
                               DataType data_type, const void* dptr)
    : data_type_(data_type), tensors_(new std::vector<TensorView>()) {
  const char* mem_ptr = reinterpret_cast<const char*>(dptr);
  tensors_->reserve(size);
  FOR_RANGE(int32_t, i, 0, size) {
    tensors_->emplace_back(shape_ptr + i * num_axes, num_axes, data_type,
                           reinterpret_cast<const void*>(mem_ptr));
    mem_ptr += tensors_->back().shape().elem_cnt() * GetSizeOfDataType(data_type);
  }
}

MutTensorListView::MutTensorListView(int64_t size, const int64_t* shape_ptr, int64_t num_axes,
                                     DataType data_type, void* dptr)
    : data_type_(data_type), tensors_(new std::vector<DataOnlyMutTensorView>()) {
  char* mem_ptr = reinterpret_cast<char*>(dptr);
  tensors_->reserve(size);
  FOR_RANGE(int32_t, i, 0, size) {
    tensors_->emplace_back(shape_ptr + i * num_axes, num_axes, data_type,
                           reinterpret_cast<void*>(mem_ptr));
    mem_ptr += tensors_->back().shape().elem_cnt() * GetSizeOfDataType(data_type);
  }
}

}  // namespace oneflow
