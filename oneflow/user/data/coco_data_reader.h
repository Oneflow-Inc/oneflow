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
#ifndef ONEFLOW_USER_DATA_COCO_DATA_READER_H_
#define ONEFLOW_USER_DATA_COCO_DATA_READER_H_

#include "oneflow/user/data/data_reader.h"
#include "oneflow/user/data/coco_parser.h"
#include "oneflow/core/common/str_util.h"
#include <json.hpp>

namespace oneflow {
namespace data {

class COCODataReader final : public DataReader<COCOImage> {
 public:
  COCODataReader(user_op::KernelInitContext* ctx);
  ~COCODataReader() = default;

 protected:
  using DataReader<COCOImage>::loader_;
  using DataReader<COCOImage>::parser_;
};

class COCOMeta final {
 public:
  COCOMeta(int64_t session_id, const std::string& annotation_file, const std::string& image_dir,
           bool remove_images_without_annotations);
  ~COCOMeta() = default;

  int64_t Size() const { return image_ids_.size(); }
  int64_t GetImageId(int64_t index) const { return image_ids_.at(index); }
  int32_t GetImageHeight(int64_t index) const {
    int64_t image_id = image_ids_.at(index);
    return image_id2image_.at(image_id)["height"].get<int32_t>();
  }
  int32_t GetImageWidth(int64_t index) const {
    int64_t image_id = image_ids_.at(index);
    return image_id2image_.at(image_id)["width"].get<int32_t>();
  }
  std::string GetImageFilePath(int64_t index) const {
    int64_t image_id = image_ids_.at(index);
    const auto& image_json = image_id2image_.at(image_id);
    return JoinPath(image_dir_, image_json["file_name"].get<std::string>());
  }
  template<typename T>
  std::vector<T> GetBboxVec(int64_t index) const;
  template<typename T>
  std::vector<T> GetLabelVec(int64_t index) const;
  template<typename T>
  void ReadSegmentationsToTensorBuffer(int64_t index, TensorBuffer* segm,
                                       TensorBuffer* segm_offset_mat) const;

 private:
  bool ImageHasValidAnnotations(int64_t image_id) const;

  static constexpr int kMinKeypointsPerImage = 10;
  nlohmann::json annotation_json_;
  std::string image_dir_;
  std::vector<int64_t> image_ids_;
  HashMap<int64_t, const nlohmann::json&> image_id2image_;
  HashMap<int64_t, const nlohmann::json&> anno_id2anno_;
  HashMap<int64_t, std::vector<int64_t>> image_id2anno_ids_;
  HashMap<int32_t, int32_t> category_id2contiguous_id_;
};

template<typename T>
std::vector<T> COCOMeta::GetBboxVec(int64_t index) const {
  std::vector<T> bbox_vec;
  int64_t image_id = image_ids_.at(index);
  const auto& anno_ids = image_id2anno_ids_.at(image_id);
  for (int64_t anno_id : anno_ids) {
    const auto& bbox_json = anno_id2anno_.at(anno_id)["bbox"];
    CHECK(bbox_json.is_array());
    CHECK_EQ(bbox_json.size(), 4);
    // COCO bounding box format is [left, top, width, height]
    // we need format xyxy
    const T alginment = static_cast<T>(1);
    const T min_size = static_cast<T>(0);
    T left = bbox_json[0].get<T>();
    T top = bbox_json[1].get<T>();
    T width = bbox_json[2].get<T>();
    T height = bbox_json[3].get<T>();
    T right = left + std::max(width - alginment, min_size);
    T bottom = top + std::max(height - alginment, min_size);
    // clip to image
    int32_t image_height = GetImageHeight(index);
    int32_t image_width = GetImageWidth(index);
    left = std::min(std::max(left, min_size), image_width - alginment);
    top = std::min(std::max(top, min_size), image_height - alginment);
    right = std::min(std::max(right, min_size), image_width - alginment);
    bottom = std::min(std::max(bottom, min_size), image_height - alginment);
    // ensure bbox is not empty
    if (right > left && bottom > top) {
      bbox_vec.insert(bbox_vec.end(), {left, top, right, bottom});
    }
  }
  return bbox_vec;
}

template<typename T>
std::vector<T> COCOMeta::GetLabelVec(int64_t index) const {
  std::vector<T> label_vec;
  int64_t image_id = image_ids_.at(index);
  const auto& anno_ids = image_id2anno_ids_.at(image_id);
  for (int64_t anno_id : anno_ids) {
    int32_t category_id = anno_id2anno_.at(anno_id)["category_id"].get<int32_t>();
    label_vec.push_back(category_id2contiguous_id_.at(category_id));
  }
  return label_vec;
}

template<typename T>
void COCOMeta::ReadSegmentationsToTensorBuffer(int64_t index, TensorBuffer* segm,
                                               TensorBuffer* segm_index) const {
  if (segm == nullptr || segm_index == nullptr) { return; }
  int64_t image_id = image_ids_.at(index);
  const auto& anno_ids = image_id2anno_ids_.at(image_id);
  std::vector<T> segm_vec;
  for (int64_t anno_id : anno_ids) {
    const auto& segm_json = anno_id2anno_.at(anno_id)["segmentation"];
    if (!segm_json.is_array()) { continue; }
    for (const auto& poly_json : segm_json) {
      CHECK(poly_json.is_array());
      for (const auto& elem : poly_json) { segm_vec.push_back(elem.get<T>()); }
    }
  }
  CHECK_EQ(segm_vec.size() % 2, 0);
  int64_t num_pts = segm_vec.size() / 2;
  segm->Resize(Shape({num_pts, 2}), GetDataType<T>::value);
  std::copy(segm_vec.begin(), segm_vec.end(), segm->mut_data<T>());

  segm_index->Resize(Shape({num_pts, 3}), DataType::kInt32);
  int32_t* index_ptr = segm_index->mut_data<int32_t>();
  int i = 0;
  int32_t segm_idx = 0;
  for (int64_t anno_id : anno_ids) {
    const auto& segm_json = anno_id2anno_.at(anno_id)["segmentation"];
    CHECK(segm_json.is_array());
    FOR_RANGE(int32_t, poly_idx, 0, segm_json.size()) {
      const auto& poly_json = segm_json[poly_idx];
      CHECK(poly_json.is_array());
      CHECK_EQ(poly_json.size() % 2, 0);
      FOR_RANGE(int32_t, pt_idx, 0, poly_json.size() / 2) {
        index_ptr[i * 3 + 0] = pt_idx;
        index_ptr[i * 3 + 1] = poly_idx;
        index_ptr[i * 3 + 2] = segm_idx;
        i += 1;
      }
    }
    segm_idx += 1;
  }
  CHECK_EQ(i, num_pts);
}

}  // namespace data
}  // namespace oneflow

#endif  // ONEFLOW_USER_DATA_COCO_DATA_READER_H_
