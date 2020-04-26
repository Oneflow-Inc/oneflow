#include "oneflow/customized/data/coco_dataset.h"
#include "oneflow/customized/data/coco_data_reader.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

COCODataset::COCODataset(user_op::KernelInitContext* ctx, COCOMeta* meta)
    : meta_(meta), cur_idx_(0) {
  int64_t total_empty_size = ctx->GetAttr<int64_t>("empty_tensor_size");
  int32_t tensor_init_bytes = ctx->GetAttr<int32_t>("tensor_init_bytes");
  empty_tensor_mgr_.reset(new COCOImageManager(total_empty_size, tensor_init_bytes));
};

int64_t COCODataset::Size() { return meta_->Size(); }

COCODataset::LoadTargetShdPtrVec COCODataset::Next() {
  LoadTargetShdPtrVec ret;
  ret.push_back(std::move(At(cur_idx_)));
  cur_idx_ = (cur_idx_ + 1) % Size();
  return ret;
}

COCODataset::LoadTargetShdPtr COCODataset::At(int64_t idx) {
  LoadTargetShdPtr ret = empty_tensor_mgr_->Get();
  int64_t image_id = meta_->GetImageId(idx);
  ret->id = image_id;
  ret->height = meta_->GetImageHeight(idx);
  ret->width = meta_->GetImageWidth(idx);
  const std::string& image_file_path = meta_->GetImageFilePath(image_id);
  PersistentInStream in_stream(DataFS(), image_file_path);
  int64_t file_size = DataFS()->GetFileSize(image_file_path);
  ret->data.Resize(Shape({file_size}), DataType::kChar);
  CHECK_EQ(in_stream.ReadFully(ret->data.mut_data<char>(), ret->data.nbytes()), 0);
  return ret;
}

}  // namespace oneflow
