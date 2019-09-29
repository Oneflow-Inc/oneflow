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
  if (coco_dataset.group_by_aspect_ratio()) { sampler().reset(new GroupedDataSampler(this)); }
  PersistentInStream in_stream(
      DataFS(), JoinPath(proto.dataset_dir(), coco_dataset.annotation_file()));
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

int64_t COCODataset::GetGroupId(int64_t idx) const {
  int64_t image_id = image_ids_.at(idx);
  const auto& image_info = image_id2image_.at(image_id);
  int64_t image_height = image_info["height"].get<int64_t>();
  int64_t image_width = image_info["width"].get<int64_t>();
  float aspect_ratio = image_height * 1.0f / image_width;
  if (aspect_ratio > 1.0f) { return 1; }
  return 0;
}

void COCODataset::GetData(int64_t idx, DataInstance* data_inst) const {
  auto* image_field = data_inst->GetField<DataSourceCase::kImage>();
  auto* bbox_field = data_inst->GetField<DataSourceCase::kObjectBoundingBox>();
  auto* segm_field = data_inst->GetField<DataSourceCase::kObjectSegmentation>();
  auto* label_field = data_inst->GetField<DataSourceCase::kObjectLabel>();

  int64_t image_id = image_ids_.at(idx);
  const auto& image = image_id2image_.at(image_id);
  // Get image data
  GetImage(image["file_name"].get<std::string>(), image_field);
  for (int64_t anno_id : image_id2anno_id_.at(image_id)) {
    const auto& anno = anno_id2anno_.at(anno_id);
    // Get bbox data
    GetBbox(anno["bbox"], bbox_field);
    // Get segmentation data
    GetSegmentation(anno["segmentation"], segm_field);
    // Get object label
    GetLabel(anno["category_id"], label_field);
  }
}

void COCODataset::GetImage(const std::string& image_file_name, DataField* data_field) const {
  auto* image_field = dynamic_cast<ImageDataField*>(data_field);
  CHECK_NOTNULL(image_field);

  auto image_file_path =
      JoinPath(dataset_proto().dataset_dir(), dataset_proto().coco().image_dir(), image_file_name);
  PersistentInStream in_stream(DataFS(), image_file_path);
  std::vector<char> buffer(DataFS()->GetFileSize(image_file_path));
  CHECK_EQ(in_stream.Read(buffer.data(), buffer.size()), 0);
  cv::_InputArray bytes_array(buffer.data(), buffer.size());
  image_field->data() = cv::imdecode(bytes_array, cv::IMREAD_ANYCOLOR);
}

void COCODataset::GetBbox(const nlohmann::json& bbox_json, DataField* data_field) const {
  CHECK(bbox_json.is_array());
  auto* bbox_field = dynamic_cast<ArrayDataField<float>*>(data_field);
  CHECK_NOTNULL(bbox_field);

  auto& bbox_vec = bbox_field->data();
  for (const auto& jval : bbox_json) { bbox_vec.push_back(jval.get<float>()); }
}

void COCODataset::GetLabel(const nlohmann::json& label_json, DataField* data_field) const {
  CHECK(label_json.is_number_integer());
  auto* label_field = dynamic_cast<ArrayDataField<int32_t>*>(data_field);
  CHECK_NOTNULL(label_field);
  label_field->data().push_back(label_json.get<int32_t>());
}

void COCODataset::GetSegmentation(const nlohmann::json& segmentation, DataField* data_field) const {
  auto* segm_field = dynamic_cast<NdarrayDataField<float>*>(data_field);
  CHECK_NOTNULL(segm_field);

  if (segmentation.is_object()) {
    auto rle_cnt_vec = segmentation["counts"].get<std::vector<uint32_t>>();
    auto h = segmentation["size"][0].get<uint32_t>();
    auto w = segmentation["size"][1].get<uint32_t>();
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
      for (const auto& point : contour) {
        segm_field->PushBack(static_cast<float>(point.x));
        segm_field->PushBack(static_cast<float>(point.y));
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
      for (const auto& segm : segm_array) { segm_field->PushBack(segm.get<float>()); }
      segm_field->AppendLodLength(2, segm_array.size());
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
