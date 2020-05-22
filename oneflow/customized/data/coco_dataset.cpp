#include "oneflow/customized/data/coco_dataset.h"
#include "oneflow/customized/data/coco_data_reader.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {
namespace data {

COCODataset::COCODataset(user_op::KernelInitContext* ctx,
                         const std::shared_ptr<const COCOMeta>& meta)
    : meta_(meta) {
  int64_t total_empty_size = ctx->GetAttr<int64_t>("empty_tensor_size");
  int32_t tensor_init_bytes = ctx->GetAttr<int32_t>("tensor_init_bytes");
  empty_tensor_mgr_.reset(new EmptyTensorManager<COCOImage>(total_empty_size, tensor_init_bytes));
};

COCODataset::LoadTargetShdPtr COCODataset::At(int64_t idx) const {
  LoadTargetShdPtr ret = empty_tensor_mgr_->Get();
  ret->index = idx;
  ret->id = meta_->GetImageId(idx);
  ret->height = meta_->GetImageHeight(idx);
  ret->width = meta_->GetImageWidth(idx);
  const std::string& image_file_path = meta_->GetImageFilePath(idx);
  PersistentInStream in_stream(DataFS(), image_file_path);
  int64_t file_size = DataFS()->GetFileSize(image_file_path);
  ret->data.Resize(Shape({file_size}), DataType::kChar);
  CHECK_EQ(in_stream.ReadFully(ret->data.mut_data<char>(), ret->data.nbytes()), 0);
  return ret;
}

size_t COCODataset::Size() const { return meta_->Size(); }

}  // namespace data
}  // namespace oneflow
