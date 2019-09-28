#ifndef ONEFLOW_CORE_DATA_DATA_TRANSFORM_H_
#define ONEFLOW_CORE_DATA_DATA_TRANSFORM_H_

#include "oneflow/core/data/data_instance.h"
#include "oneflow/core/common/util.h"
#include <fenv.h>

extern "C" {
#include "maskApi.h"
}

namespace oneflow {
namespace data {

using TransformCase = DataTransformProto::TransformCase;

template<TransformCase trans>
void DoDataTransform(DataInstance* data_inst, const DataTransformProto& proto);

template<DataSourceCase dsrc, TransformCase trans>
struct DataTransformer;

// Specialized definitions
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
  using T = typename ScaleFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* image_field = dynamic_cast<ImageFieldT*>(data_inst->GetField<DataSourceCase::kImage>());
    if (!image_field) { return; }
    auto* field = data_inst->GetOrCreateField<DataSourceCase::kImageScale>();
    auto* scale_field = dynamic_cast<ScaleFieldT*>(field);
    CHECK_NOTNULL(scale_field);

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

    auto& scale_vec = scale_field->data();
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

    int32_t mask_h = proto.segmentation_poly_to_mask().mask_h();
    int32_t mask_w = proto.segmentation_poly_to_mask().mask_w();
    T bbox_w = bbox->data().at(2) - bbox->data().at(0);
    T bbox_h = bbox->data().at(3) - bbox->data().at(1);
    T scale_w = static_cast<T>(mask_w) / std::max(GetOneVal<T>(), bbox_w);
    T scale_h = static_cast<T>(mask_h) / std::max(GetOneVal<T>(), bbox_h);
    segm_mask->data().resize(segm_poly->GetLod(0).at(0) * mask_h * mask_w, 0);

    std::vector<double> segm_poly_resized;
    T* poly_dptr = segm_poly->data();
    const auto& points_vec = segm_poly->GetLod(segm_poly->Levels() - 1);
    for (size_t points : points_vec) {
      FOR_RANGE(size_t, i, 0, points) {
        if (i % 2 == 0) {
          segm_poly_resized.push_back((poly_dptr[i] - bbox->data().at(0)) * scale_w);
        } else {
          segm_poly_resized.push_back((poly_dptr[i] - bbox->data().at(1)) * scale_h);
        }
      }
      poly_dptr += points;
    }

    double* poly_resized_dptr = segm_poly_resized.data();
    MaskT* mask_dptr = segm_mask->data().data();
    size_t obj_poly_begin = 0;
    const auto& polys_vec = segm_poly->GetLod(1);
    for (size_t polys : polys_vec) {
      FOR_RANGE(size_t, i, 0, polys) {
        std::vector<uint8_t> mask_vec(mask_h * mask_w);
        size_t poly_points = points_vec.at(obj_poly_begin + i);
        CHECK_EQ(poly_points % 2, 0);
        PolygonXy2ColMajorMask(poly_resized_dptr, poly_points / 2, mask_h, mask_w, mask_vec.data());
        std::transform(mask_vec.cbegin(), mask_vec.cend(), mask_dptr, mask_dptr,
                       std::bit_or<MaskT>());
        poly_resized_dptr += poly_points;
      }
      obj_poly_begin += polys;
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

}  // namespace data
}  // namespace oneflow

#define DATA_TRANSFORM_SEQ                                \
  OF_PP_MAKE_TUPLE_SEQ(DataTransformProto::kResize)       \
  OF_PP_MAKE_TUPLE_SEQ(DataTransformProto::kTargetResize) \
  OF_PP_MAKE_TUPLE_SEQ(DataTransformProto::kSegmentationPolyToMask)

#define DATA_FIELD_TRANSFORM_TUPLE_SEQ                                    \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, DATA_SOURCE_SEQ, \
                                   DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ, DATA_TRANSFORM_SEQ)

#endif  // ONEFLOW_CORE_DATA_DATA_TRANSFORM_H_
