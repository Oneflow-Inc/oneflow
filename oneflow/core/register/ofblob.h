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
  void AutoMemCopyTo(T* ptr, int64_t len) const;
  template<typename T>
  void AutoMemCopyFrom(const T* ptr, int64_t len) const;

 private:
  DeviceCtx* device_ctx_;
  Blob* blob_;
  MemoryCase mem_case_;
};

inline void OfBlob::CopyShapeFrom(const int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  Shape shape(std::vector<int64_t>(ptr, ptr + num_axis));
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
  blob_->dense_shape_mut_view().set_shape(shape);
}

inline void OfBlob::CopyShapeTo(int64_t* ptr, int64_t num_axis) const {
  CHECK_EQ(num_axis, NumAxes());
  FOR_RANGE(int32_t, i, 0, num_axis) { ptr[i] = blob_->shape().At(i); }
}

template<typename T>
void OfBlob::AutoMemCopyTo(T* ptr, int64_t len) const {
  CHECK_EQ(blob_->shape().elem_cnt(), len);
  CHECK(blob_->data_type() == GetDataType<T>::value);
  SyncAutoMemcpy(device_ctx_, ptr, blob_->dptr(), len * sizeof(T), mem_case_, blob_->mem_case());
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

}  // namespace oneflow

#endif  // ONEFLOW_CORE_REGISTER_OFBLOB_H_
