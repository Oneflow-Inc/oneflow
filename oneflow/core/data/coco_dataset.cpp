#include "oneflow/core/data/coco_dataset.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

extern "C" {
#include "maskApi.h"
}

namespace oneflow {
namespace data {

COCODataset::COCODataset(const DatasetProto& proto) : Dataset(proto) {
  CHECK_EQ(proto.dataset_catalog_case(), DatasetProto::kCoco);
  const COCODatasetCatalog& coco_dataset = proto.coco();
  PersistentInStream in_stream(DataFS(),
                               JoinPath(proto.dataset_dir(), coco_dataset.annotation_file()));
  std::string json_str;
  std::string line;
  while (in_stream.ReadLine(&line) == 0) { json_str += line; }
  std::istringstream in_str_stream(json_str);
  in_str_stream >> annotation_json_;

  // initialize image_ids_, image_id2image_ and image_id2anno_id_ 
  for (const auto& image : annotation_json_["images"]) {
    int64_t id = image["id"].get<int64_t>();
    image_ids_.push_back(id);
    CHECK(image_id2image_.emplace(id, image).second);
    CHECK(image_id2anno_id_.emplace(id, std::vector<int64_t>()).second);
  }
  // build anno map
  for (const auto& anno : annotation_json_["annotations"]) {
    int64_t id = anno["id"].get<int64_t>();
    int64_t image_id = anno["image_id"].get<int64_t>();
    // ignore crowd object for now
    if (anno["iscrowd"].get<int>() == 1) { continue; }
    // check if empty bbox, bbox format is (left, top, width, height)
    const auto bbox = anno["bbox"];
    if (!(bbox[2].get<float>() > 1.0f && bbox[3].get<float>() > 1.0f)) {
      LOG(INFO) << "coco dataset ignore too small bbox, image_id: " << image_id
                << ", anno_id: " << id << ", bbox: (" << bbox[0].get<float>() << ","
                << bbox[1].get<float>() << "," << bbox[2].get<float>() << ","
                << bbox[3].get<float>() << ")";
      continue;
    }
    // check if invalid segmentation
    const auto segm = anno["segmentation"];
    if (segm.is_array()) {
      for (const auto& poly : segm) {
        // at least 3 points can compose a polygon
        // every point needs 2 element (x, y) to present
        CHECK_GT(poly.size(), 6);
      }
    }
    CHECK(anno_id2anno_.emplace(id, anno).second);
    image_id2anno_id_.at(image_id).push_back(id);
  }
  // remove images without annotations if necessary
  if (coco_dataset.remove_images_without_annotations()) {
    HashSet<int64_t> to_remove_image_ids;
    for (int64_t image_id : image_ids_) {
      if (!ImageHasValidAnnotations(image_id)) {
        to_remove_image_ids.insert(image_id);
      }
    }
    image_ids_.erase(std::remove_if(image_ids_.begin(), image_ids_.end(),
                                    [&to_remove_image_ids](int64_t image_id) {
                                      return to_remove_image_ids.find(image_id)
                                             != to_remove_image_ids.end();
                                    }),
                     image_ids_.end());
  }
  // sort image ids for reproducible results
  std::sort(image_ids_.begin(), image_ids_.end());
  // build categories map
  std::vector<int32_t> category_ids;
  for (const auto& cat : annotation_json_["categories"]) {
    category_ids.emplace_back(cat["id"].get<int32_t>());
  }
  std::sort(category_ids.begin(), category_ids.end());
  int32_t contiguous_id = 1;
  for (int32_t category_id : category_ids) {
    CHECK(category_id2contiguous_id_.emplace(category_id, contiguous_id++).second);
  }
  if (coco_dataset.group_by_aspect_ratio()) { sampler().reset(new GroupedDataSampler(this)); }
}

bool COCODataset::ImageHasValidAnnotations(int64_t image_id) const {
  const std::vector<int64_t>& anno_id_vec = image_id2anno_id_.at(image_id);
  if (anno_id_vec.empty()) { return false; }

  bool bbox_area_all_close_to_zero = true;
  size_t visible_keypoints_count = 0;
  for (int64_t anno_id : anno_id_vec) {
    const auto& anno = anno_id2anno_.at(anno_id);
    if (anno["bbox"][2] > 1 && anno["bbox"][3] > 1) { bbox_area_all_close_to_zero = false; }
    if (anno.contains("keypoints")) {
      const auto& keypoints = anno["keypoints"];
      CHECK_EQ(keypoints.size() % 3, 0);
      FOR_RANGE(size_t, i, 0, keypoints.size() / 3) {
        int32_t keypoints_label = keypoints[i * 3 + 2].get<int32_t>();
        if (keypoints_label > 0) { visible_keypoints_count += 1; }
      }
    }
  }
  // check if all boxes have close to zero area
  if (bbox_area_all_close_to_zero) { return false; }
  // keypoints task have a slight different critera for considering
  // if an annotation is valid
  if (!anno_id2anno_.at(anno_id_vec.at(0)).contains("keypoints")) { return true; }
  // for keypoint detection tasks, only consider valid images those
  // containing at least min_keypoints_per_image
  if (visible_keypoints_count >= kMinKeypointsPerImage) { return true; }
  return false;
}

int64_t COCODataset::GetGroupId(int64_t idx) const {
  int64_t image_id = image_ids_.at(idx);
  const auto& image_info = image_id2image_.at(image_id);
  float image_height = image_info["height"].get<float>();
  float image_width = image_info["width"].get<float>();
  float aspect_ratio = image_height * 1.0f / image_width;
  if (aspect_ratio >= 1.0f) { return 1; }
  return 0;
}

void COCODataset::GetData(int64_t idx, DataInstance* data_inst) const {
  auto* image_field = data_inst->GetField<DataSourceCase::kImage>();
  auto* image_size_field = data_inst->GetField<DataSourceCase::kImageSize>();
  auto* bbox_field = data_inst->GetField<DataSourceCase::kObjectBoundingBox>();
  auto* label_field = data_inst->GetField<DataSourceCase::kObjectLabel>();
  DataField* segm_field = nullptr;
  if (data_inst->HasField<DataSourceCase::kObjectSegmentationMask>()) {
    segm_field = data_inst->GetOrCreateField<DataSourceCase::kObjectSegmentation>(
        dataset_proto().coco().max_segm_poly_points());
  } else {
    segm_field = data_inst->GetField<DataSourceCase::kObjectSegmentation>();
  }

  int64_t image_id = image_ids_.at(idx);
  auto* image_id_field =
      dynamic_cast<ArrayDataField<int64_t>*>(data_inst->GetField<DataSourceCase::kImageId>());
  if (image_id_field) { image_id_field->data().push_back(image_id); }
  // Get image data
  const auto& image = image_id2image_.at(image_id);
  GetImage(image, image_field, image_size_field);
  for (int64_t anno_id : image_id2anno_id_.at(image_id)) {
    const auto& anno = anno_id2anno_.at(anno_id);
    // Get bbox data
    GetBbox(anno["bbox"], image, bbox_field);
    // Get segmentation data
    GetSegmentation(anno["segmentation"], segm_field);
    // Get object label
    GetLabel(anno["category_id"], label_field);
  }
}

void COCODataset::GetImage(const nlohmann::json& image_json, DataField* image_field,
                           DataField* image_size_field) const {
  auto* image = dynamic_cast<ImageDataField*>(image_field);
  auto* image_size = dynamic_cast<ArrayDataField<int32_t>*>(image_size_field);
  if (image) {
    auto image_file_path =
        JoinPath(dataset_proto().dataset_dir(), dataset_proto().coco().image_dir(),
                 image_json["file_name"].get<std::string>());
    PersistentInStream in_stream(DataFS(), image_file_path);
    std::vector<char> buffer(DataFS()->GetFileSize(image_file_path));
    CHECK_EQ(in_stream.ReadFully(buffer.data(), buffer.size()), 0);
    cv::_InputArray bytes_array(buffer.data(), buffer.size());
    auto& image_mat = image->data();
    image_mat = cv::imdecode(bytes_array, cv::IMREAD_ANYCOLOR | cv::IMREAD_ANYDEPTH);
    CHECK_EQ(image_mat.depth(), CV_8U);
    if (image_mat.channels() != 3) { cv::cvtColor(image_mat, image_mat, CV_GRAY2BGR); }
    CHECK_EQ(image_mat.channels(), 3);
  }
  if (image_size) {
    image_size->data().push_back(image_json["height"].get<int32_t>());
    image_size->data().push_back(image_json["width"].get<int32_t>());
  }
}

void COCODataset::GetBbox(const nlohmann::json& bbox_json, const nlohmann::json& image_json,
                          DataField* data_field) const {
  CHECK(bbox_json.is_array());
  auto* bbox_field = dynamic_cast<ArrayDataField<float>*>(data_field);
  if (!bbox_field) { return; }
  // COCO bounding box format is [left, top, width, height]
  // we need format xyxy
  auto& bbox_vec = bbox_field->data();
  const auto to_remove = GetOneVal<float>();
  const auto min_size = GetZeroVal<float>();
  float left = bbox_json[0].get<float>();
  float top = bbox_json[1].get<float>();
  float right = left + std::max(bbox_json[2].get<float>() - to_remove, min_size);
  float bottom = top + std::max(bbox_json[3].get<float>() - to_remove, min_size);
  // clip to image
  float image_height = image_json["height"].get<float>();
  float image_width = image_json["width"].get<float>();
  left = std::max(left, 0.0f);
  CHECK_LT(left, image_width - to_remove);
  top = std::max(top, 0.0f);
  CHECK_LT(top, image_height - to_remove);
  right = std::min(right, image_width - to_remove);
  CHECK_GT(right, left);
  bottom = std::min(bottom, image_height - to_remove);
  CHECK_GT(bottom, top);
  // push to data_field
  bbox_vec.push_back(left);
  bbox_vec.push_back(top);
  bbox_vec.push_back(right);
  bbox_vec.push_back(bottom);
}

void COCODataset::GetLabel(const nlohmann::json& label_json, DataField* data_field) const {
  CHECK(label_json.is_number_integer());
  auto* label_field = dynamic_cast<ArrayDataField<int32_t>*>(data_field);
  if (!label_field) { return; }
  label_field->data().push_back(category_id2contiguous_id_.at(label_json.get<int32_t>()));
}

void COCODataset::GetSegmentation(const nlohmann::json& segmentation, DataField* data_field) const {
  using DataFieldT = typename DataFieldTrait<DataSourceCase::kObjectSegmentation>::type;
  using T = typename DataFieldT::data_type;

  auto* segm_field = dynamic_cast<DataFieldT*>(data_field);
  if (!segm_field) { return; }

  if (segmentation.is_object()) {
    auto rle_cnt_vec = segmentation["counts"].get<std::vector<uint32_t>>();
    auto h = segmentation["size"][0].get<uint32_t>();
    auto w = segmentation["size"][1].get<uint32_t>();
    size_t total_cnt = 0;
    for (auto cnt : rle_cnt_vec) { total_cnt += cnt; }
    CHECK_EQ(total_cnt, h * w);

    std::vector<uint8_t> mask(total_cnt, 0);
    RLE rle({h, w, rle_cnt_vec.size(), rle_cnt_vec.data()});
    rleDecode(&rle, mask.data(), mask.size());

    cv::Mat mat(h, w, CV_8UC1, mask.data());
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mat, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);

    for (const auto& contour : contours) {
      for (const auto& point : contour) {
        segm_field->PushBack(static_cast<T>(point.x));
        segm_field->PushBack(static_cast<T>(point.y));
      }
      segm_field->AppendLodLength(2, contour.size());
    }
    segm_field->AppendLodLength(1, contours.size());
    // contours hierarchy are not supported yet
    for (const auto& cnt_ids : hierarchy) {
      FOR_RANGE(int, i, 0, 4) { CHECK_LT(cnt_ids[i], 0); }
    }
  } else if (segmentation.is_array()) {
    for (const auto& segm_array : segmentation) {
      CHECK_EQ(segm_array.size() % 2, 0);
      for (const auto& segm : segm_array) { segm_field->PushBack(segm.get<T>()); }
      segm_field->AppendLodLength(2, segm_array.size() / 2);
    }
    segm_field->AppendLodLength(1, segmentation.size());
  } else {
    UNIMPLEMENTED();
  }
  segm_field->IncreaseLodLength(0, 1);
}

REGISTER_DATASET(DatasetProto::kCoco, COCODataset);

}  // namespace data
}  // namespace oneflow
