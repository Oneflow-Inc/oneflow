#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"
#include <json.hpp>

namespace oneflow {

struct COCOSegmentationPolygonList {
  TensorBuffer elems;
  TensorBuffer elem_cnts;
};

struct COCOSegmentationRLE {
  TensorBuffer counts;
  int32_t height;
  int32_t width;
};

struct COCOSegmentation {
  enum class Format { kPolygonList = 0, kRLE };
  union {
    COCOSegmentationPolygonList polys;
    COCOSegmentationRLE rle;
  };
  Format format;

  COCOSegmentation() noexcept {};
  COCOSegmentation(const COCOSegmentation& a) {
    format = a.format;
    if (format == Format::kPolygonList) {
      polys = a.polys;
    } else if (format == Format::kRLE) {
      rle = a.rle;
    }
  };
  COCOSegmentation(COCOSegmentation&& a) {
    format = a.format;
    if (format == Format::kPolygonList) {
      polys = std::move(a.polys);
    } else if (format == Format::kRLE) {
      rle = std::move(a.rle);
    }
  };
  ~COCOSegmentation() {
    if (format == Format::kPolygonList) {
      polys.~COCOSegmentationPolygonList();
    } else if (format == Format::kRLE) {
      rle.~COCOSegmentationRLE();
    }
  }
};

struct COCODataInstance {
  TensorBuffer image;
  int64_t image_id;
  int32_t image_height;
  int32_t image_width;
  TensorBuffer object_bboxes;
  TensorBuffer object_labels;
  std::vector<COCOSegmentation> segmentations;
};

class COCODataset final : public Dataset<COCODataInstance> {
 public:
  using LoadTargetPtr = std::shared_ptr<COCODataInstance>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;
  COCODataset(user_op::KernelInitContext* ctx);
  ~COCODataset() = default;

  LoadTargetPtrList Next() override;
  LoadTargetPtr At(int64_t idx) override;

  int64_t Size() override { return image_ids_.size(); }

  bool EnableRandomAccess() override { return true; }
  bool EnableGetSize() override { return true; }

 private:
  void ReadImage(const nlohmann::json& image_json, TensorBuffer* buffer) const;
  void ReadSegmentation(const nlohmann::json& segm_json, COCOSegmentation* segm) const;
  bool ImageHasValidAnnotations(int64_t image_id) const;

  static constexpr int kMinKeypointsPerImage = 10;
  nlohmann::json annotation_json_;
  std::string image_dir_;
  std::vector<int64_t> image_ids_;
  HashMap<int64_t, const nlohmann::json&> image_id2image_;
  HashMap<int64_t, const nlohmann::json&> anno_id2anno_;
  HashMap<int64_t, std::vector<int64_t>> image_id2anno_id_;
  HashMap<int32_t, int32_t> category_id2contiguous_id_;
  int64_t cur_idx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
