#include "oneflow/customized/data/coco_dataset.h"
#include "oneflow/customized/data/coco_data_reader.h"
#include "oneflow/core/persistence/file_system.h"
#include "oneflow/core/persistence/persistent_in_stream.h"

namespace oneflow {

COCODataset::COCODataset(user_op::KernelInitContext* ctx, COCOMeta* meta)
    : meta_(meta), cur_idx_(0) {
  int64_t total_empty_size = ctx->GetAttr<int64_t>("empty_tensor_size");
  int32_t tensor_init_bytes = ctx->GetAttr<int32_t>("tensor_init_bytes");
  empty_tensor_mgr_.reset(new COCOImageManager(total_empty_size, tensor_init_bytes));
};

int64_t COCODataset::Size() { return meta_->Size(); }

COCODataset::LoadTargetPtrList COCODataset::Next() {
  LoadTargetPtrList ret;
  ret.push_back(std::move(At(cur_idx_)));
  cur_idx_ = (cur_idx_ + 1) % Size();
  return ret;
}

COCODataset::LoadTargetPtr COCODataset::At(int64_t idx) {
  LoadTargetPtr ret = empty_tensor_mgr_->Get();
  int64_t image_id = meta_->GetImageId(idx);
  ret->id = image_id;
  ret->height = meta_->GetImageHeight(idx);
  ret->width = meta_->GetImageWidth(idx);
  const std::string& image_file_path = meta_->GetImageFilePath(image_id);
  PersistentInStream in_stream(DataFS(), image_file_path);
  int64_t file_size = DataFS()->GetFileSize(image_file_path);
  ret->data.Resize(Shape({file_size}), DataType::kChar);
  CHECK_EQ(in_stream.ReadFully(ret->data.mut_data<char>(), ret->data.nbytes()), 0);
  return ret;
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

// void COCODataset::ReadSegmentation(const nlohmann::json& segm_json, COCOSegmentation* segm) const
// {
//   if (segm_json.is_object()) {
//     // segmentation is RLE format
//     CHECK(segm_json["counts"].is_array());
//     CHECK_EQ(segm_json["size"].size(), 2);
//     segm->format = COCOSegmentation::Format::kRLE;
//     int64_t counts_size = segm_json["counts"].size();
//     segm->rle.counts.Resize(Shape({counts_size}), DataType::kInt64);
//     FOR_RANGE(int64_t, i, 0, counts_size) {
//       segm->rle.counts.mut_data<int64_t>()[i] = segm_json["counts"][i].get<int64_t>();
//     }
//     segm->rle.height = segm_json["size"][0].get<int32_t>();
//     segm->rle.width = segm_json["size"][1].get<int32_t>();
//   } else if (segm_json.is_array()) {
//     // segmentation is polygon list format
//     int64_t poly_cnt = segm_json.size();
//     segm->format = COCOSegmentation::Format::kPolygonList;
//     segm->polys.elem_cnts.Resize(Shape({poly_cnt}), DataType::kInt32);
//     std::vector<float> elem_vec;
//     FOR_RANGE(int64_t, i, 0, poly_cnt) {
//       int64_t poly_size = segm_json[i].size();
//       CHECK_EQ(poly_size % 2, 0);
//       segm->polys.elem_cnts.mut_data<int32_t>()[i] = poly_size;
//       for (const auto& elem : segm_json[i]) { elem_vec.push_back(elem.get<float>()); }
//     }
//     segm->polys.elems.Resize(Shape({static_cast<int64_t>(elem_vec.size())}), DataType::kFloat);
//     std::copy(elem_vec.begin(), elem_vec.end(), segm->polys.elems.mut_data<float>());
//   } else {
//     UNIMPLEMENTED();
//   }
// }

}  // namespace oneflow
