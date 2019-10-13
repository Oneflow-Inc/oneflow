#include "oneflow/core/data/data_transform.h"
#include <fenv.h>

extern "C" {
#include "maskApi.h"
}

namespace oneflow {
namespace data {

template<>
struct DataTransformer<DataSourceCase::kImage, TransformCase::kResize> {
  using ImageFieldT = typename DataFieldTrait<DataSourceCase::kImage>::type;
  using ScaleFieldT = typename DataFieldTrait<DataSourceCase::kImageScale>::type;
  using T = typename ScaleFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* image_field = dynamic_cast<ImageFieldT*>(data_inst->GetField<DataSourceCase::kImage>());
    if (!image_field) { return; }

    auto& image_mat = image_field->data();
    int32_t origin_height = image_mat.rows;
    int32_t origin_width = image_mat.cols;
    int32_t target_height = proto.resize().height();
    int32_t target_width = proto.resize().width();
    cv::resize(image_mat, image_mat, cv::Size(target_width, target_height));

    auto* scale_field = data_inst->GetOrCreateField<DataSourceCase::kImageScale>();
    auto& scale_vec = dynamic_cast<ScaleFieldT*>(scale_field)->data();
    scale_vec.push_back(static_cast<T>(target_height) / static_cast<T>(origin_height));
    scale_vec.push_back(static_cast<T>(target_width) / static_cast<T>(origin_width));
  }
};

template<>
struct DataTransformer<DataSourceCase::kObjectBoundingBox, TransformCase::kResize> {
  using BboxFieldT = typename DataFieldTrait<DataSourceCase::kObjectBoundingBox>::type;
  using ScaleFieldT = typename DataFieldTrait<DataSourceCase::kImageScale>::type;
  using T = typename BboxFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* bbox_field =
        dynamic_cast<BboxFieldT*>(data_inst->GetField<DataSourceCase::kObjectBoundingBox>());
    if (!bbox_field) { return; }
    auto* scale_field =
        dynamic_cast<ScaleFieldT*>(data_inst->GetField<DataSourceCase::kImageScale>());
    CHECK_NOTNULL(scale_field);

    auto& bbox_vec = bbox_field->data();
    auto& scale_vec = scale_field->data();
    FOR_RANGE(size_t, i, 0, bbox_vec.size()) { bbox_vec.at(i) *= scale_vec.at(i % 2); }
  }
};

template<>
struct DataTransformer<DataSourceCase::kObjectSegmentation, TransformCase::kResize> {
  using SgemFieldT = typename DataFieldTrait<DataSourceCase::kObjectSegmentation>::type;
  using ScaleFieldT = typename DataFieldTrait<DataSourceCase::kImageScale>::type;
  using T = typename SgemFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* segm_field =
        dynamic_cast<SgemFieldT*>(data_inst->GetField<DataSourceCase::kObjectSegmentation>());
    if (!segm_field) { return; }
    auto* scale_field =
        dynamic_cast<ScaleFieldT*>(data_inst->GetField<DataSourceCase::kImageScale>());
    CHECK_NOTNULL(scale_field);

    auto& scale_vec = scale_field->data();
    T* dptr = segm_field->data();
    auto& last_lod_vec = segm_field->GetLod(segm_field->Levels() - 1);
    for (size_t elems : last_lod_vec) {
      FOR_RANGE(size_t, i, 0, elems) { *dptr *= scale_vec.at(i % 2); }
      dptr += elems;
    }
  }
};

template<>
struct DataTransformer<DataSourceCase::kImage, TransformCase::kTargetResize> {
  using ImageFieldT = typename DataFieldTrait<DataSourceCase::kImage>::type;
  using ScaleFieldT = typename DataFieldTrait<DataSourceCase::kImageScale>::type;
  using ImageSizeFieldT = typename DataFieldTrait<DataSourceCase::kImageSize>::type;
  using T = typename ScaleFieldT::data_type;
  using SizeT = typename ImageSizeFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* image_field = dynamic_cast<ImageFieldT*>(data_inst->GetField<DataSourceCase::kImage>());
    if (!image_field) { return; }
    auto* field = data_inst->GetOrCreateField<DataSourceCase::kImageScale>();
    auto* scale_field = dynamic_cast<ScaleFieldT*>(field);
    CHECK_NOTNULL(scale_field);
    auto* image_size_field =
        dynamic_cast<ImageSizeFieldT*>(data_inst->GetField<DataSourceCase::kImageSize>());

    int32_t target_size = proto.target_resize().target_size();
    int32_t max_size = proto.target_resize().max_size();
    CHECK_GT(target_size, 0);
    CHECK_GE(max_size, target_size);
    auto& image = image_field->data();
    int32_t origin_image_height = image.rows;
    int32_t origin_image_width = image.cols;
    int32_t image_size_min = std::min(origin_image_height, origin_image_width);
    int32_t image_size_max = std::max(origin_image_height, origin_image_width);
    float scale = static_cast<float>(target_size) / image_size_min;
    if (std::round(scale * image_size_max) > max_size) {
      scale = static_cast<float>(max_size) / image_size_max;
    }
    cv::resize(image, image, cv::Size(), scale, scale, cv::INTER_LINEAR);

    int32_t image_height = image.rows;
    int32_t image_width = image.cols;
    CHECK_LE(std::max(image_height, image_width), max_size);
    CHECK(std::max(image_height, image_width) == max_size
          || std::min(image_height, image_width) == target_size);

    if (image_size_field) {
      image_size_field->data().at(0) = static_cast<SizeT>(image_height);
      image_size_field->data().at(1) = static_cast<SizeT>(image_width);
    }

    auto& scale_vec = scale_field->data();
    scale_vec.clear();
    scale_vec.push_back(static_cast<T>(image_height) / static_cast<T>(origin_image_height));
    scale_vec.push_back(static_cast<T>(image_width) / static_cast<T>(origin_image_width));
  }
};

template<>
struct DataTransformer<DataSourceCase::kObjectBoundingBox, TransformCase::kTargetResize> {
  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    DataTransformer<DataSourceCase::kObjectBoundingBox, TransformCase::kResize>::Apply(data_inst,
                                                                                       proto);
  }
};

template<>
struct DataTransformer<DataSourceCase::kObjectSegmentation, TransformCase::kTargetResize> {
  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    DataTransformer<DataSourceCase::kObjectSegmentation, TransformCase::kResize>::Apply(data_inst,
                                                                                        proto);
  }
};

template<>
struct DataTransformer<DataSourceCase::kObjectSegmentation,
                       TransformCase::kSegmentationPolyToMask> {
  using BboxFieldT = typename DataFieldTrait<DataSourceCase::kObjectBoundingBox>::type;
  using SegmPolyFieldT = typename DataFieldTrait<DataSourceCase::kObjectSegmentation>::type;
  using SegmMaskFieldT = typename DataFieldTrait<DataSourceCase::kObjectSegmentationMask>::type;
  using T = typename BboxFieldT::data_type;
  using MaskT = typename SegmMaskFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* bbox =
        dynamic_cast<BboxFieldT*>(data_inst->GetField<DataSourceCase::kObjectBoundingBox>());
    auto* segm_poly =
        dynamic_cast<SegmPolyFieldT*>(data_inst->GetField<DataSourceCase::kObjectSegmentation>());
    if (!bbox || !segm_poly) { return; }
    auto* segm_mask_field = data_inst->GetOrCreateField<DataSourceCase::kObjectSegmentationMask>();
    auto* segm_mask = dynamic_cast<SegmMaskFieldT*>(segm_mask_field);
    CHECK_NOTNULL(segm_mask);
    CHECK_EQ(segm_poly->Levels(), 3);

    int32_t mask_h = proto.segmentation_poly_to_mask().mask_h();
    int32_t mask_w = proto.segmentation_poly_to_mask().mask_w();
    const auto& polys_vec = segm_poly->GetLod(1);
    const auto& points_vec = segm_poly->GetLod(2);

    T* poly_dptr = segm_poly->data();
    segm_mask->data().resize(segm_poly->GetLod(0).at(0) * mask_h * mask_w, 0);
    MaskT* mask_dptr = segm_mask->data().data();
    size_t polys_offset = 0;
    FOR_RANGE(size_t, i, 0, polys_vec.size()) {
      T bbox_x1 = bbox->data().at(i * 4 + 0);
      T bbox_y1 = bbox->data().at(i * 4 + 1);
      T bbox_x2 = bbox->data().at(i * 4 + 2);
      T bbox_y2 = bbox->data().at(i * 4 + 3);
      T scale_w = static_cast<T>(mask_w) / std::max(GetOneVal<T>(), bbox_x2 - bbox_x1);
      T scale_h = static_cast<T>(mask_h) / std::max(GetOneVal<T>(), bbox_y2 - bbox_y1);
      size_t polys = polys_vec.at(i);
      FOR_RANGE(size_t, j, 0, polys) {
        size_t poly_points = points_vec.at(polys_offset + j);
        CHECK_EQ(poly_points % 2, 0);
        // resized points coord into box
        std::vector<double> poly_points_resized(poly_points);
        FOR_RANGE(size_t, k, 0, poly_points) {
          if (k % 2 == 0) {
            poly_points_resized.at(k) = (poly_dptr[k] - bbox_x1) * scale_w;
          } else {
            poly_points_resized.at(k) = (poly_dptr[k] - bbox_y1) * scale_h;
          }
        }
        poly_dptr += poly_points;
        // convert poly points to mask
        std::vector<uint8_t> mask_vec(mask_h * mask_w);
        PolygonXy2ColMajorMask(poly_points_resized.data(), poly_points / 2, mask_h, mask_w,
                               mask_vec.data());
        std::transform(mask_vec.cbegin(), mask_vec.cend(), mask_dptr, mask_dptr,
                       std::bit_or<MaskT>());
      }
      polys_offset += polys;
      mask_dptr += mask_h * mask_w;
    }
  }

  static void PolygonXy2ColMajorMask(const double* point, size_t num_points, size_t mask_h,
                                     size_t mask_w, uint8_t* mask) {
    RLE rle;
    const int fe_excepts = fegetexcept();
    CHECK_NE(fedisableexcept(fe_excepts), -1);
    rleFrPoly(&rle, point, num_points, mask_h, mask_w);
    CHECK_NE(feenableexcept(fe_excepts), -1);
    rleDecode(&rle, mask, 1);
    rleFree(&rle);
  }
};

template<>
void DoDataTransform<TransformCase::kResize>(DataInstance* data_inst,
                                             const DataTransformProto& proto) {
  CHECK(proto.has_resize());
  DataTransformer<DataSourceCase::kImage, TransformCase::kResize>::Apply(data_inst, proto);
  DataTransformer<DataSourceCase::kObjectBoundingBox, TransformCase::kResize>::Apply(data_inst,
                                                                                     proto);
  DataTransformer<DataSourceCase::kObjectSegmentation, TransformCase::kResize>::Apply(data_inst,
                                                                                      proto);
}

template<>
void DoDataTransform<TransformCase::kTargetResize>(DataInstance* data_inst,
                                                   const DataTransformProto& proto) {
  CHECK(proto.has_target_resize());
  DataTransformer<DataSourceCase::kImage, TransformCase::kTargetResize>::Apply(data_inst, proto);
  DataTransformer<DataSourceCase::kObjectBoundingBox, TransformCase::kTargetResize>::Apply(
      data_inst, proto);
  DataTransformer<DataSourceCase::kObjectSegmentation, TransformCase::kTargetResize>::Apply(
      data_inst, proto);
}

template<>
void DoDataTransform<TransformCase::kSegmentationPolyToMask>(DataInstance* data_inst,
                                                             const DataTransformProto& proto) {
  CHECK(proto.has_segmentation_poly_to_mask());
  DataTransformer<DataSourceCase::kObjectSegmentation,
                  TransformCase::kSegmentationPolyToMask>::Apply(data_inst, proto);
}

void DataTransform(DataInstance* data_inst, const DataTransformProto& trans_proto) {
#define MAKE_CASE(trans)                            \
  case trans: {                                     \
    DoDataTransform<trans>(data_inst, trans_proto); \
    break;                                          \
  }

  switch (trans_proto.transform_case()) {
    MAKE_CASE(DataTransformProto::kResize)
    MAKE_CASE(DataTransformProto::kTargetResize)
    MAKE_CASE(DataTransformProto::kSegmentationPolyToMask)
    default: { UNIMPLEMENTED(); }
  }
#undef MAKE_CASE
}

}  // namespace data
}  // namespace oneflow
