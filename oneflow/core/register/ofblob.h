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
  OfBlob(DeviceCtx* device_ctx, Blob* blob) : device_ctx_(device_ctx), blob_(blob) {
    mem_case_.mutable_host_mem();
  }
  ~OfBlob() = default;

  int data_type() const { return blob_->data_type(); }
  size_t NumAxes() const { return blob_->shape().NumAxes(); }
  size_t num_of_lod_levels() const { return blob_->blob_desc().num_of_lod_levels(); }
  bool is_dynamic() const { return blob_->blob_desc().is_dynamic(); }
  LoDTree GetLoDTree() const;
  void SetLoDTree(const LoDTree& lod_tree) const;
  void CopyShapeTo(int64_t* ptr, int64_t num_axis) const;
  void CopyShapeFrom(const int64_t* ptr, int64_t num_axis) const;

  template<typename T>
  void AutoMemCopyFrom(const T* ptr, int64_t len) const;

  int64_t TotalNumOfTensors() const;
  int64_t NumOfTensorListSlices() const;
  int64_t TensorIndex4SliceId(int32_t slice_id) const;
  void AddTensorListSlice() const;

  void ClearTensorLists();
  void ResetTensorIterator();
  void ResetMutTensorIterator();
  void IncTensorIterator();
  void IncMutTensorIterator();
  bool CurTensorIteratorEqEnd() const { return !cur_tensor_; }
  bool CurMutTensorIteratorEqEnd() const { return !cur_mut_tensor_; }
  void CurTensorCopyShapeTo(int64_t* ptr, int64_t num_axis) const;
  void CurMutTensorCopyShapeFrom(const int64_t* ptr, int64_t num_axis) const;
  template<typename T>
  void CurTensorAutoMemCopyTo(T* ptr, int64_t len) const;
  template<typename T>
  void CurMutTensorAutoMemCopyFrom(const T* ptr, int64_t len) const;

 private:
  void ClearShape(FullyMutTensorView* tensor) const;

  DeviceCtx* device_ctx_;
  Blob* blob_;
  MemoryCase mem_case_;
  std::unique_ptr<TensorView> cur_tensor_;
  std::unique_ptr<FullyMutTensorView> cur_mut_tensor_;
};

inline void OfBlob::CopyShapeFrom(const int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  Shape shape(DimVector(ptr, ptr + num_axis));
  if (blob_->blob_desc().is_dynamic() == false) {
    CHECK_EQ(shape, blob_->static_shape());
    return;
  }
  int64_t num_of_lod_levels = blob_->blob_desc().num_of_lod_levels();
  if (num_of_lod_levels > 0) {
    CHECK_GT(num_of_lod_levels, 1);
    CHECK_LE(shape.At(0), blob_->static_shape().Count(0, num_of_lod_levels));
    CHECK_LE(shape.Count(1), blob_->static_shape().Count(num_of_lod_levels));
  } else {
    CHECK_LE(shape.elem_cnt(), blob_->static_shape().elem_cnt());
  }
  blob_->dense_shape_mut_view()->set_shape(shape);
}

inline void OfBlob::CopyShapeTo(int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  FOR_RANGE(int32_t, i, 0, num_axis) { ptr[i] = blob_->shape().At(i); }
}

template<typename T>
void OfBlob::AutoMemCopyFrom(const T* ptr, int64_t len) const {
  CHECK_EQ(blob_->shape().elem_cnt(), len);
  CHECK(blob_->data_type() == GetDataType<T>::value);
  SyncAutoMemcpy(device_ctx_, blob_->mut_dptr(), ptr, len * sizeof(T), blob_->mem_case(),
                 mem_case_);
}

inline LoDTree OfBlob::GetLoDTree() const {
  CHECK(blob_->blob_desc().num_of_lod_levels());
  LoDTree lod_tree = blob_->tree_lod_view().lod_tree();
  CHECK_EQ(lod_tree.offset(), 0);
  CHECK_EQ(lod_tree.length(), blob_->shape().At(0));
  return lod_tree;
}

inline void OfBlob::SetLoDTree(const LoDTree& lod_tree) const {
  CHECK(blob_->blob_desc().num_of_lod_levels());
  CHECK_EQ(lod_tree.offset(), 0);
  CHECK_EQ(lod_tree.length(), blob_->shape().At(0));
  blob_->tree_lod_mut_view().UpdateLoD(lod_tree);
}

inline int64_t OfBlob::TotalNumOfTensors() const { return blob_->total_num_of_tensors(); }

inline int64_t OfBlob::NumOfTensorListSlices() const { return blob_->num_of_tensor_list_slices(); }

inline int64_t OfBlob::TensorIndex4SliceId(int32_t slice_id) const {
  return blob_->tensor_index4slice_id(slice_id);
}

inline void OfBlob::AddTensorListSlice() const { return blob_->add_tensor_list_slice(); }

inline void OfBlob::ResetTensorIterator() { cur_tensor_ = blob_->first_tensor(); }

inline void OfBlob::ClearTensorLists() { blob_->clear_tensor_lists(); }

inline void OfBlob::ResetMutTensorIterator() {
  blob_->clear_tensor_lists();
  cur_mut_tensor_.reset();
  cur_mut_tensor_ = blob_->add_tensor(cur_mut_tensor_.get());
  ClearShape(cur_mut_tensor_.get());
}

inline void OfBlob::IncTensorIterator() { cur_tensor_ = blob_->next_tensor(*cur_tensor_); }

inline void OfBlob::IncMutTensorIterator() {
  cur_mut_tensor_ = blob_->add_tensor(cur_mut_tensor_.get());
}

inline void OfBlob::CurTensorCopyShapeTo(int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  DenseShapeMutView(ptr, num_axis).set_shape(cur_tensor_->shape());
  ClearShape(cur_mut_tensor_.get());
}

inline void OfBlob::ClearShape(FullyMutTensorView* tensor) const {
  if (tensor == nullptr) { return; }
  std::vector<int64_t> zeros(NumAxes(), 0LL);
  tensor->set_shape(DenseShapeView(zeros.data(), NumAxes()));
}

inline void OfBlob::CurMutTensorCopyShapeFrom(const int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  cur_mut_tensor_->set_shape(DenseShapeView(ptr, num_axis));
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
  CHECK_EQ(cur_mut_tensor_->shape().elem_cnt(), len);
  CHECK(cur_mut_tensor_->data_type() == GetDataType<T>::value);
  SyncAutoMemcpy(device_ctx_, cur_mut_tensor_->mut_dptr(), ptr, len * sizeof(T), blob_->mem_case(),
                 mem_case_);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_OFBLOB_H_
