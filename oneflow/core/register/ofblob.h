#ifndef ONEFLOW_CORE_REGISTER_OFBLOB_H_
#define ONEFLOW_CORE_REGISTER_OFBLOB_H_

#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

class Blob;

class OfBlob final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfBlob);
  OfBlob(DeviceCtx* device_ctx, Blob* blob)
      : device_ctx_(device_ctx), blob_(blob), tensor_back_inserter_(new TensorBackInserter(blob)) {
    mem_case_.mutable_host_mem();
  }
  ~OfBlob() = default;

  int data_type() const { return blob_->data_type(); }
  size_t NumAxes() const { return blob_->shape().NumAxes(); }
  size_t is_tensor_list() const { return blob_->blob_desc().is_tensor_list(); }
  bool is_dynamic() const { return blob_->blob_desc().is_dynamic(); }
  void CopyShapeTo(int64_t* ptr, int64_t num_axis) const;
  void CopyShapeFrom(const int64_t* ptr, int64_t num_axis) const;

  int64_t TotalNumOfTensors() const;
  int64_t NumOfTensorListSlices() const;
  int64_t TensorIndex4SliceId(int32_t slice_id) const;
  void AddTensorListSlice() const;

  void ResetTensorIterator();
  void IncTensorIterator();
  bool CurTensorIteratorEqEnd() const { return blob_->IsEndTensor(*cur_tensor_); }
  void CurTensorCopyShapeTo(int64_t* ptr, int64_t num_axis) const;
  template<typename T>
  void CurTensorAutoMemCopyTo(T* ptr, int64_t len) const;

  void ClearTensorLists();
  void AddTensor();
  bool CurMutTensorAvailable() const { return tensor_back_inserter_->IsCurMutTensorAvailable(); }
  void CurMutTensorCopyShapeFrom(const int64_t* ptr, int64_t num_axis) const;
  template<typename T>
  void CurMutTensorAutoMemCopyFrom(const T* ptr, int64_t len) const;

 private:
  void ClearShape(FullyMutTensorView* tensor) const;

  DeviceCtx* device_ctx_;
  Blob* blob_;
  MemoryCase mem_case_;
  std::unique_ptr<TensorBackInserter> tensor_back_inserter_;
  std::unique_ptr<TensorView> cur_tensor_;
};

inline void OfBlob::CopyShapeFrom(const int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  Shape shape(DimVector(ptr, ptr + num_axis));
  if (blob_->blob_desc().is_dynamic() == false) {
    CHECK_EQ(shape, blob_->static_shape());
    return;
  }
  CHECK_LE(shape.elem_cnt(), blob_->static_shape().elem_cnt());
  blob_->mut_shape_view()->set_shape(shape);
}

inline void OfBlob::CopyShapeTo(int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  FOR_RANGE(int32_t, i, 0, num_axis) { ptr[i] = blob_->shape().At(i); }
}

inline int64_t OfBlob::TotalNumOfTensors() const { return blob_->total_num_of_tensors(); }

inline int64_t OfBlob::NumOfTensorListSlices() const { return blob_->num_of_tensor_list_slices(); }

inline int64_t OfBlob::TensorIndex4SliceId(int32_t slice_id) const {
  return blob_->tensor_index4slice_id(slice_id);
}

inline void OfBlob::AddTensorListSlice() const { tensor_back_inserter_->add_tensor_list_slice(); }

inline void OfBlob::ResetTensorIterator() {
  cur_tensor_.reset(new TensorView(blob_->BeginTensor()));
}

inline void OfBlob::ClearTensorLists() { tensor_back_inserter_->ClearTensorLists(); }

inline void OfBlob::AddTensor() {
  tensor_back_inserter_->add_tensor();
  ClearShape(tensor_back_inserter_->cur_mut_tensor());
}

inline void OfBlob::IncTensorIterator() { blob_->MoveToNextTensor(cur_tensor_.get()); }

inline void OfBlob::CurTensorCopyShapeTo(int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  MutShapeView(ptr, num_axis).set_shape(cur_tensor_->shape());
}

inline void OfBlob::ClearShape(FullyMutTensorView* tensor) const {
  if (tensor == nullptr) { return; }
  std::vector<int64_t> zeros(NumAxes(), 0LL);
  tensor->set_shape(ShapeView(zeros.data(), NumAxes()));
}

inline void OfBlob::CurMutTensorCopyShapeFrom(const int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  tensor_back_inserter_->cur_mut_tensor()->set_shape(ShapeView(ptr, num_axis));
}

template<typename T>
void OfBlob::CurTensorAutoMemCopyTo(T* ptr, int64_t len) const {
  CHECK_EQ(cur_tensor_->shape().elem_cnt(), len);
  CHECK(cur_tensor_->data_type() == GetDataType<T>::value);
  SyncAutoMemcpy(device_ctx_, ptr, cur_tensor_->dptr(), len * sizeof(T), mem_case_,
                 blob_->mem_case());
}

template<typename T>
void OfBlob::CurMutTensorAutoMemCopyFrom(const T* ptr, int64_t len) const {
  CHECK_EQ(tensor_back_inserter_->cur_mut_tensor()->shape().elem_cnt(), len);
  CHECK(tensor_back_inserter_->cur_mut_tensor()->data_type() == GetDataType<T>::value);
  SyncAutoMemcpy(device_ctx_, tensor_back_inserter_->cur_mut_tensor()->mut_dptr(), ptr,
                 len * sizeof(T), blob_->mem_case(), mem_case_);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_OFBLOB_H_
