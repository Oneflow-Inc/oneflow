/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/data/coco_dataset.h"
#include "oneflow/user/data/coco_data_reader.h"
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
  PersistentInStream in_stream(session_id_, DataFS(session_id_), image_file_path);
  int64_t file_size = DataFS(session_id_)->GetFileSize(image_file_path);
  sample->data.Resize(Shape({file_size}), DataType::kChar);
  CHECK_EQ(in_stream.ReadFully(sample->data.mut_data<char>(), sample->data.nbytes()), 0);
  ret.emplace_back(std::move(sample));
  return ret;
}

size_t COCODataset::Size() const { return meta_->Size(); }

}  // namespace data
}  // namespace oneflow
