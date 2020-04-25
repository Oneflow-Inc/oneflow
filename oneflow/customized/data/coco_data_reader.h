#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_DATA_READER_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_DATA_READER_H_

#include "oneflow/customized/data/data_reader.h"
#include "oneflow/customized/data/coco_parser.h"
#include "oneflow/core/common/str_util.h"
#include <json.hpp>

namespace oneflow {

class COCODataReader final : public DataReader<COCOImage> {
 public:
  COCODataReader(user_op::KernelInitContext* ctx);
  ~COCODataReader() = default;

 protected:
  using DataReader<COCOImage>::loader_;
  using DataReader<COCOImage>::parser_;

 private:
  std::unique_ptr<COCOMeta> meta_;
};

class COCOMeta final {
 public:
  COCOMeta(user_op::KernelInitContext* ctx);
  ~COCOMeta() = default;

  int64_t Size() const { return image_ids_.size(); }
  int64_t GetImageId(int64_t index) const { return image_ids_.at(index); }
  int32_t GetImageHeight(int64_t image_id) const {
    return image_id2image_.at(image_id)["height"].get<int32_t>();
  }
  int32_t GetImageWidth(int64_t image_id) const {
    return image_id2image_.at(image_id)["width"].get<int32_t>();
  }
  std::string GetImageFilePath(int64_t image_id) const {
    const auto& image_json = image_id2image_.at(image_id);
    return JoinPath(image_dir_, image_json["file_name"].get<std::string>());
  }
  template<typename T>
  std::vector<T> GetBboxVec(int64_t image_id) const;
  template<typename T>
  std::vector<T> GetLabelVec(int64_t image_id) const;
  template<typename T>
  void ReadSegmentationsToTensorBuffer(int64_t image_id, TensorBuffer* segm,
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
std::vector<T> COCOMeta::GetBboxVec(int64_t image_id) const {
  std::vector<T> bbox_vec;
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
    int32_t image_height = GetImageHeight(image_id);
    int32_t image_width = GetImageWidth(image_id);
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
std::vector<T> COCOMeta::GetLabelVec(int64_t image_id) const {
  std::vector<T> label_vec;
  const auto& anno_ids = image_id2anno_ids_.at(image_id);
  for (int64_t anno_id : anno_ids) {
    int32_t category_id = anno_id2anno_.at(anno_id)["category_id"].get<int32_t>();
    label_vec.push_back(category_id2contiguous_id_.at(category_id));
  }
  return label_vec;
}

template<typename T>
void COCOMeta::ReadSegmentationsToTensorBuffer(int64_t image_id, TensorBuffer* segm,
                                               TensorBuffer* segm_offset_mat) const {
  if (segm == nullptr || segm_offset_mat == nullptr) { return; }
  const auto& anno_ids = image_id2anno_ids_.at(image_id);
  std::vector<T> segm_vec;
  for (int64_t anno_id : anno_ids) {
    const auto& segm_json = anno_id2anno_.at(anno_id)["segmentation"];
    if (!segm_json.is_array()) { continue; }
    for (const auto& poly : segm_json) {
      for (const auto& elem : poly) { segm_vec.push_back(elem.get<T>()); }
    }
  }
  int64_t num_elems = segm_vec.size();
  segm->Resize(Shape({num_elems}), GetDataType<T>::value);
  std::copy(segm_vec.begin(), segm_vec.end(), segm->mut_data<T>());
  segm_offset_mat->Resize(Shape({num_elems, 3}), DataType::kInt32);
  int32_t* offset_ptr = segm_offset_mat->mut_data<int32_t>();
  int i = 0;
  int32_t offset_of_poly_in_segm = 0;
  int32_t offset_of_segm_in_img = 0;
  for (int64_t anno_id : anno_ids) {
    const auto& segm_json = anno_id2anno_.at(anno_id)["segmentation"];
    if (!segm_json.is_array()) { continue; }
    for (const auto& poly : segm_json) {
      FOR_RANGE(int32_t, offset_of_elem_in_poly, 0, poly.size()) {
        offset_ptr[i * 3 + 0] = offset_of_elem_in_poly;
        offset_ptr[i * 3 + 1] = offset_of_poly_in_segm;
        offset_ptr[i * 3 + 2] = offset_of_segm_in_img;
        i += 1;
      }
      offset_of_poly_in_segm += 1;
    }
    offset_of_segm_in_img += 1;
  }
  CHECK_EQ(i, segm_vec.size());
}

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_DATA_READER_H_
