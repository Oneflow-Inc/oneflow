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
#include "oneflow/user/data/coco_data_reader.h"
#include "oneflow/user/data/coco_dataset.h"
#include "oneflow/user/data/distributed_training_dataset.h"
#include "oneflow/user/data/group_batch_dataset.h"
#include "oneflow/user/data/batch_dataset.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {
namespace data {

COCODataReader::COCODataReader(user_op::KernelInitContext* ctx) : DataReader<COCOImage>(ctx) {
  std::shared_ptr<const COCOMeta> meta(new COCOMeta(
      ctx->Attr<int64_t>("session_id"), ctx->Attr<std::string>("annotation_file"),
      ctx->Attr<std::string>("image_dir"), ctx->Attr<bool>("remove_images_without_annotations")));

  std::unique_ptr<RandomAccessDataset<COCOImage>> coco_dataset_ptr(new COCODataset(ctx, meta));
  loader_.reset(new DistributedTrainingDataset<COCOImage>(
      ctx->parallel_ctx().parallel_num(), ctx->parallel_ctx().parallel_id(),
      ctx->Attr<bool>("stride_partition"), ctx->Attr<bool>("shuffle_after_epoch"),
      ctx->Attr<int64_t>("random_seed"), std::move(coco_dataset_ptr)));

  size_t batch_size = ctx->TensorDesc4ArgNameAndIndex("image", 0)->shape().elem_cnt();
  if (ctx->Attr<bool>("group_by_ratio")) {
    auto GetGroupId = [](const std::shared_ptr<COCOImage>& sample) {
      return static_cast<int64_t>(sample->height / sample->width);
    };
    loader_.reset(new GroupBatchDataset<COCOImage>(batch_size, GetGroupId, std::move(loader_)));
  } else {
    loader_.reset(new BatchDataset<COCOImage>(batch_size, std::move(loader_)));
  }

  parser_.reset(new COCOParser(meta));
  StartLoadThread();
}

COCOMeta::COCOMeta(int64_t session_id, const std::string& annotation_file,
                   const std::string& image_dir, bool remove_images_without_annotations)
    : image_dir_(image_dir) {
  // Read content of annotation file (json format) to json obj
  PersistentInStream in_stream(session_id, DataFS(session_id), annotation_file);
  std::string json_str;
  std::string line;
  while (in_stream.ReadLine(&line) == 0) { json_str += line; }
  std::istringstream in_str_stream(json_str);
  in_str_stream >> annotation_json_;
  // initialize image_ids_, image_id2image_ and image_id2anno_ids_
  for (const auto& image : annotation_json_["images"]) {
    int64_t id = image["id"].get<int64_t>();
    image_ids_.push_back(id);
    CHECK(image_id2image_.emplace(id, image).second);
    CHECK(image_id2anno_ids_.emplace(id, std::vector<int64_t>()).second);
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
    image_id2anno_ids_.at(image_id).push_back(id);
  }
  // remove images without annotations if necessary
  if (remove_images_without_annotations) {
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
}

bool COCOMeta::ImageHasValidAnnotations(int64_t image_id) const {
  const std::vector<int64_t>& anno_id_vec = image_id2anno_ids_.at(image_id);
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

}  // namespace data
}  // namespace oneflow
