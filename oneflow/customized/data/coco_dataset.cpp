#include "oneflow/customized/data/coco_dataset.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/common/str_util.h"

namespace oneflow {

COCODataset::COCODataset(user_op::KernelInitContext* ctx)
    : image_dir_(ctx->GetAttr<std::string>("image_dir")), cur_idx_(0) {
  // Read content of annotation file (json format) to json obj
  const std::string& anno_path = ctx->GetAttr<std::string>("annotation_file");
  PersistentInStream in_stream(DataFS(), anno_path);
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
    // check if invalid segmentation
    if (anno["segmentation"].is_array()) {
      for (const auto& poly : anno["segmentation"]) {
        // at least 3 points can compose a polygon
        // every point needs 2 element (x, y) to present
        CHECK_GT(poly.size(), 6);
      }
    }
    CHECK(anno_id2anno_.emplace(id, anno).second);
    image_id2anno_id_.at(image_id).push_back(id);
  }
  // remove images without annotations if necessary
  if (ctx->GetAttr<bool>("remove_images_without_annotations")) {
    HashSet<int64_t> to_remove_image_ids;
    for (int64_t image_id : image_ids_) {
      if (!ImageHasValidAnnotations(image_id)) { to_remove_image_ids.insert(image_id); }
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
  // group by ratio sampler
  // if (coco_dataset.group_by_aspect_ratio()) { sampler().reset(new GroupedDataSampler(this)); }
}

COCODataset::LoadTargetPtrList COCODataset::Next() {
  LoadTargetPtrList data_inst_vec;
  data_inst_vec.push_back(At(cur_idx_));
  cur_idx_ = (cur_idx_ + 1) % Size();
  return data_inst_vec;
}

COCODataset::LoadTargetPtr COCODataset::At(int64_t idx) {
  COCODataInstance* data_inst = new COCODataInstance;
  int64_t image_id = image_ids_.at(idx);
  data_inst->image_id = image_id;
  const auto& image = image_id2image_.at(image_id);
  ReadImage(image, &data_inst->image);
  data_inst->image_height = image["height"].get<int32_t>();
  data_inst->image_width = image["width"].get<int32_t>();
  const auto& image_anno_ids = image_id2anno_id_.at(image_id);
  int64_t anno_size = image_anno_ids.size();
  data_inst->object_bboxes.Resize(Shape({anno_size, 4}), DataType::kFloat);
  data_inst->object_labels.Resize(Shape({anno_size}), DataType::kInt32);
  data_inst->segmentations.resize(anno_size);
  FOR_RANGE(int64_t, i, 0, anno_size) {
    const auto& anno = anno_id2anno_.at(image_anno_ids[i]);
    auto* bbox_ptr = data_inst->object_bboxes.mut_data<float>() + i * 4;
    bbox_ptr[0] = anno["bbox"][0].get<float>();
    bbox_ptr[1] = anno["bbox"][1].get<float>();
    bbox_ptr[2] = anno["bbox"][2].get<float>();
    bbox_ptr[3] = anno["bbox"][3].get<float>();
    data_inst->object_labels.mut_data<int32_t>()[i] =
        category_id2contiguous_id_.at(anno["category_id"].get<int32_t>());
    ReadSegmentation(anno["segmentation"], &data_inst->segmentations.at(i));
  }
  return std::shared_ptr<COCODataInstance>(data_inst);
}

void COCODataset::ReadImage(const nlohmann::json& image_json, TensorBuffer* image_buffer) const {
  std::string image_file_path = JoinPath(image_dir_, image_json["file_name"].get<std::string>());
  PersistentInStream in_stream(DataFS(), image_file_path);
  int64_t file_size = DataFS()->GetFileSize(image_file_path);
  image_buffer->Resize(Shape({file_size}), DataType::kChar);
  CHECK_EQ(in_stream.ReadFully(image_buffer->mut_data<char>(), image_buffer->nbytes()), 0);
}

// template<typename T>
// bool COCODataset::ReadBbox(const nlohmann::json& bbox_json, const int32_t image_height,
//                            const int32_t image_width, T* bbox_ptr) const {
//   CHECK(bbox_json.is_array());
//   CHECK_EQ(bbox_json.size(), 4);
//   // COCO bounding box format is [left, top, width, height]
//   // we need format xyxy
//   const T alginment = static_cast<T>(1);
//   const T min_size = static_cast<T>(0);
//   T left = bbox_json[0].get<T>();
//   T top = bbox_json[1].get<T>();
//   T width = bbox_json[2].get<T>();
//   T height = bbox_json[3].get<T>();
//   T right = left + std::max(width - alginment, min_size);
//   T bottom = top + std::max(height - alginment, min_size);
//   // clip to image
//   left = std::min(std::max(left, min_size), image_width - alginment);
//   top = std::min(std::max(top, min_size), image_height - alginment);
//   right = std::min(std::max(right, min_size), image_width - alginment);
//   bottom = std::min(std::max(bottom, min_size), image_height - alginment);
//   // empty bbox is invalid
//   if (right <= left || bottom <= top) { return false; }
//   bbox_ptr[0] = left;
//   bbox_ptr[1] = top;
//   bbox_ptr[2] = right;
//   bbox_ptr[3] = bottom;
//   return true;
// }

void COCODataset::ReadSegmentation(const nlohmann::json& segm_json, COCOSegmentation* segm) const {
  if (segm_json.is_object()) {
    // segmentation is RLE format
    CHECK(segm_json["counts"].is_array());
    CHECK_EQ(segm_json["size"].size(), 2);
    segm->format = COCOSegmentation::Format::kRLE;
    int64_t counts_size = segm_json["counts"].size();
    segm->rle.counts.Resize(Shape({counts_size}), DataType::kInt64);
    FOR_RANGE(int64_t, i, 0, counts_size) {
      segm->rle.counts.mut_data<int64_t>()[i] = segm_json["counts"][i].get<int64_t>();
    }
    segm->rle.height = segm_json["size"][0].get<int32_t>();
    segm->rle.width = segm_json["size"][1].get<int32_t>();
  } else if (segm_json.is_array()) {
    // segmentation is polygon list format
    int64_t poly_cnt = segm_json.size();
    segm->format = COCOSegmentation::Format::kPolygonList;
    segm->polys.elem_cnts.Resize(Shape({poly_cnt}), DataType::kInt32);
    std::vector<float> elem_vec;
    FOR_RANGE(int64_t, i, 0, poly_cnt) {
      int64_t poly_size = segm_json[i].size();
      CHECK_EQ(poly_size % 2, 0);
      segm->polys.elem_cnts.mut_data<int32_t>()[i] = poly_size;
      for (const auto& elem : segm_json[i]) { elem_vec.push_back(elem.get<float>()); }
    }
    segm->polys.elems.Resize(Shape({static_cast<int64_t>(elem_vec.size())}), DataType::kFloat);
    std::copy(elem_vec.begin(), elem_vec.end(), segm->polys.elems.mut_data<float>());
  } else {
    UNIMPLEMENTED();
  }
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
  // check if all boxes are close to zero area
  if (bbox_area_all_close_to_zero) { return false; }
  // keypoints task have a slight different critera for considering
  // if an annotation is valid
  if (!anno_id2anno_.at(anno_id_vec.at(0)).contains("keypoints")) { return true; }
  // for keypoint detection tasks, only consider valid images those
  // containing at least min_keypoints_per_image
  if (visible_keypoints_count >= kMinKeypointsPerImage) { return true; }
  return false;
}

}  // namespace oneflow
