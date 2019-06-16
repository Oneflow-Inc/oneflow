#include "oneflow/core/dataset/coco_dataset.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/record/coco.pb.h"
#include <opencv2/opencv.hpp>

extern "C" {
#include "maskApi.h"
}

namespace oneflow {

void COCODataset::VirtualInit() {
  CHECK_EQ(dataset_proto().dataset_catalog_case(), DatasetProto::kCoco);
  const COCODatasetCatalog& coco_dataset = dataset_proto().coco();
  PersistentInStream in_stream(
      DataFS(), JoinPath(dataset_proto().dataset_dir(), coco_dataset.annotation_file()));
  std::string json_str;
  std::string line;
  while (in_stream.ReadLine(&line) == 0) { json_str += line; }
  std::istringstream in_str_stream(json_str);
  in_str_stream >> annotation_json_;

  // build image id array and map
  for (const auto& image : annotation_json_["images"]) {
    int64_t id = image["id"].get<int64_t>();
    image_ids_.push_back(id);
    CHECK(image_id2image_.emplace(id, image).second);
  }
  std::sort(image_ids_.begin(), image_ids_.end());
  // build anno map
  for (const auto& anno : annotation_json_["annotations"]) {
    int64_t id = anno["id"].get<int64_t>();
    int64_t image_id = anno["image_id"].get<int64_t>();
    CHECK(anno_id2anno_.emplace(id, anno).second);
    image_id2anno_id_[image_id].push_back(id);
  }
  // build categories map
  std::vector<int32_t> category_ids;
  for (const auto& cate : annotation_json_["categories"]) {
    int32_t id = cate["id"].get<int32_t>();
    category_ids.push_back(id);
  }
  std::sort(category_ids.begin(), category_ids.end());
  int32_t contiguous_id = 0;
  for (int32_t category_id : category_ids) {
    ++contiguous_id;
    CHECK(category_id2contiguous_id_.emplace(category_id, contiguous_id).second);
  }
}

std::unique_ptr<OFRecord> COCODataset::EncodeOneRecord(int64_t idx) const {
  std::unique_ptr<OFRecord> ofrecord(new OFRecord());
  // Encode image binary data
  int64_t image_id = image_ids_.at(idx);
  Feature& image_feature = (*ofrecord->mutable_feature())["image"];
  EncodeImage(image_id, image_feature);
  // Encode
  Feature& segm_feature = (*ofrecord->mutable_feature())["gt_segm"];
  EncodeSegmentation(image_id, segm_feature);
  return ofrecord;
}

void COCODataset::EncodeImage(int64_t image_id, Feature& feature) const {
  const auto& image = image_id2image_.at(image_id);
  auto image_file_path = JoinPath(dataset_proto().dataset_dir(), dataset_proto().coco().image_dir(),
                                  image["file_name"].get<std::string>());
  PersistentInStream in_stream(DataFS(), image_file_path);
  std::vector<char> buffer(DataFS()->GetFileSize(image_file_path));
  CHECK_EQ(in_stream.Read(buffer.data(), buffer.size()), 0);
  feature.ParseFromArray(buffer.data(), buffer.size());
}

void COCODataset::EncodeSegmentation(int64_t image_id, Feature& feature) const {
  const auto& anno_ids = image_id2anno_id_.at(image_id);
  auto* segm_bytes_list = feature.mutable_bytes_list();
  for (int64_t anno_id : anno_ids) {
    PolygonList polygon_list;
    const auto& anno = anno_id2anno_.at(anno_id);
    if (anno["segmentation"].is_object()) {
      auto rle_cnt_vec = anno["segmentation"]["counts"].get<std::vector<uint32_t>>();
      auto h = anno["segmentation"]["size"][0].get<uint32_t>();
      auto w = anno["segmentation"]["size"][1].get<uint32_t>();
      size_t total_cnt = 0;
      for (auto cnt : rle_cnt_vec) { total_cnt += cnt; }
      CHECK_EQ(total_cnt, h * w);

      std::vector<uint8_t> mask(total_cnt);
      RLE rle({h, w, rle_cnt_vec.size(), rle_cnt_vec.data()});
      rleDecode(&rle, mask.data(), mask.size());

      cv::Mat mat(h, w, CV_8UC1, mask.data());
      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      cv::findContours(mat, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);

      for (const auto& contour : contours) {
        auto* polygon = polygon_list.add_polygons();
        for (const auto& point : contour) {
          polygon->add_value(static_cast<float>(point.x));
          polygon->add_value(static_cast<float>(point.y));
        }
      }
      // contours hierarchy are not supported yet
      for (const auto& cnt_ids : hierarchy) {
        FOR_RANGE(int, i, 0, 4) { CHECK_LT(cnt_ids[i], 0); }
      }
    } else if (anno["segmentation"].is_array()) {
      for (const auto& segm_array : anno["segmentation"]) {
        auto* polygon = polygon_list.add_polygons();
        for (const auto& segm : segm_array) { polygon->add_value(segm.get<float>()); }
      }
    } else {
      UNIMPLEMENTED();
    }
    segm_bytes_list->add_value(polygon_list.SerializeAsString());
  }
}

REGISTER_DATASET(DatasetProto::kCoco, COCODataset);

}  // namespace oneflow
