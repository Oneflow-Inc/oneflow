#include "oneflow/customized/data/coco_dataset.h"
#include "oneflow/customized/data/coco_data_reader.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {
namespace data {

COCODataset::LoadTargetShdPtrVec COCODataset::At(int64_t index) const {
  LoadTargetShdPtrVec ret;
  LoadTargetShdPtr sample(new COCOImage());
  sample->index = index;
  sample->id = meta_->GetImageId(index);
  sample->height = meta_->GetImageHeight(index);
  sample->width = meta_->GetImageWidth(index);
  const std::string& image_file_path = meta_->GetImageFilePath(index);
  PersistentInStream in_stream(DataFS(), image_file_path);
  int64_t file_size = DataFS()->GetFileSize(image_file_path);
  sample->data.Resize(Shape({file_size}), DataType::kChar);
  CHECK_EQ(in_stream.ReadFully(sample->data.mut_data<char>(), sample->data.nbytes()), 0);
  ret.emplace_back(std::move(sample));
  return ret;
}

size_t COCODataset::Size() const { return meta_->Size(); }

}  // namespace data
}  // namespace oneflow
