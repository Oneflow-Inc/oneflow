#ifndef ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
#define ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_

#include "oneflow/customized/data/dataset.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {

// struct COCOSegmentationPolygonList {
//   TensorBuffer elems;
//   TensorBuffer elem_cnts;
// };

// struct COCOSegmentationRLE {
//   TensorBuffer counts;
//   int32_t height;
//   int32_t width;
// };

// struct COCOSegmentation {
//   enum class Format { kPolygonList = 0, kRLE };
//   union {
//     COCOSegmentationPolygonList polys;
//     COCOSegmentationRLE rle;
//   };
//   Format format;

//   COCOSegmentation() noexcept {};
//   COCOSegmentation(const COCOSegmentation& a) {
//     format = a.format;
//     if (format == Format::kPolygonList) {
//       polys = a.polys;
//     } else if (format == Format::kRLE) {
//       rle = a.rle;
//     }
//   };
//   COCOSegmentation(COCOSegmentation&& a) {
//     format = a.format;
//     if (format == Format::kPolygonList) {
//       polys = std::move(a.polys);
//     } else if (format == Format::kRLE) {
//       rle = std::move(a.rle);
//     }
//   };
//   ~COCOSegmentation() {
//     if (format == Format::kPolygonList) {
//       polys.~COCOSegmentationPolygonList();
//     } else if (format == Format::kRLE) {
//       rle.~COCOSegmentationRLE();
//     }
//   }
// };

// struct COCODataInstance {
//   TensorBuffer image;
//   int64_t image_id;
//   int32_t image_height;
//   int32_t image_width;
//   TensorBuffer object_bboxes;
//   TensorBuffer object_labels;
//   std::vector<COCOSegmentation> segmentations;
// };

struct COCOImage {
  TensorBuffer data;
  int64_t id;
  int32_t height;
  int32_t width;
};

class COCOMeta;

class COCODataset final : public Dataset<COCOImage> {
 public:
  using LoadTargetPtr = std::shared_ptr<COCOImage>;
  using LoadTargetPtrList = std::vector<LoadTargetPtr>;

  COCODataset(user_op::KernelInitContext* ctx, COCOMeta* meta);
  ~COCODataset() = default;

  LoadTargetPtrList Next() override;
  LoadTargetPtr At(int64_t idx) override;

  int64_t Size() override;

  bool EnableRandomAccess() override { return true; }
  bool EnableGetSize() override { return true; }

 private:
  std::unique_ptr<EmptyTensorManager<COCOImage>> empty_tensor_mgr_;
  const COCOMeta* meta_;
  int64_t cur_idx_;
};

class COCOImageManager final : public EmptyTensorManager<COCOImage> {
 public:
  COCOImageManager(int64_t total_empty_size, int32_t tensor_init_bytes)
      : EmptyTensorManager<COCOImage>(total_empty_size, tensor_init_bytes){};

 protected:
  void PrepareEmpty(COCOImage& image) override {
    image.data.Resize({GetTenosrInitBytes()}, DataType::kChar);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_COCO_DATASET_H_
