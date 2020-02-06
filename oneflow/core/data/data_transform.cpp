#include "oneflow/core/data/data_transform.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
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
    scale_vec.clear();
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
    // image scale (h_scale, w_scale)
    // bbox format (x, y, x, y)
    FOR_RANGE(size_t, i, 0, bbox_vec.size()) { bbox_vec.at(i) *= scale_vec.at((i + 1) % 2); }
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

    // image scale (h_scale, w_scale)
    // segm poly format (x, y, x, y, ...)
    T* dptr = segm_field->data();
    FOR_RANGE(size_t, i, 0, segm_field->total_length()) {
      dptr[i] *= scale_field->data().at((i + 1) % 2);
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
    int32_t image_height = 0;
    int32_t image_width = 0;
    GetSize(target_size, max_size, origin_image_height, origin_image_width, image_height,
            image_width);
    cv::resize(image, image, cv::Size(image_width, image_height), 0, 0, cv::INTER_LINEAR);

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

  static void GetSize(const int32_t target_size, const int32_t max_size, const int32_t o_h,
                      const int32_t o_w, int32_t& h, int32_t& w) {
    // set round to banker's rounding
    int saved_round_way = std::fegetround();
    CHECK_EQ(std::fesetround(FE_TONEAREST), 0);

    float size_min = std::min<float>(o_h, o_w);
    float size_max = std::max<float>(o_h, o_w);
    float size_min_resized = static_cast<float>(target_size);
    float size_max_resized = (size_max / size_min) * size_min_resized;
    if (size_max_resized > max_size) {
      size_max_resized = static_cast<float>(max_size);
      size_min_resized = std::nearbyint(size_max_resized * size_min / size_max);
    }

    if (o_w < o_h) {
      w = static_cast<int32_t>(size_min_resized);
      h = static_cast<int32_t>(size_max_resized);
    } else {
      h = static_cast<int32_t>(size_min_resized);
      w = static_cast<int32_t>(size_max_resized);
    }

    std::fesetround(saved_round_way);
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
  using ImageSizeFieldT = typename DataFieldTrait<DataSourceCase::kImageSize>::type;
  using SegmPolyFieldT = typename DataFieldTrait<DataSourceCase::kObjectSegmentation>::type;
  using SegmMaskFieldT = typename DataFieldTrait<DataSourceCase::kObjectSegmentationMask>::type;
  using T = typename SegmPolyFieldT::data_type;
  using MaskT = typename SegmMaskFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* image_size =
        dynamic_cast<ImageSizeFieldT*>(data_inst->GetField<DataSourceCase::kImageSize>());
    auto* segm_poly =
        dynamic_cast<SegmPolyFieldT*>(data_inst->GetField<DataSourceCase::kObjectSegmentation>());
    auto* segm_mask = dynamic_cast<SegmMaskFieldT*>(
        data_inst->GetField<DataSourceCase::kObjectSegmentationMask>());
    if (!image_size || !segm_poly || !segm_mask) { return; }
    CHECK_EQ(segm_poly->Levels(), 3);

    const auto& polys_vec = segm_poly->GetLod(1);
    const auto& points_vec = segm_poly->GetLod(2);
    const size_t image_height = image_size->data().at(0);
    const size_t image_width = image_size->data().at(1);

    T* poly_dptr = segm_poly->data();
    MaskT* mask_dptr = segm_mask->data();
    size_t polys_offset = 0;
    FOR_RANGE(size_t, i, 0, polys_vec.size()) {
      size_t polys = polys_vec.at(i);
      std::vector<MaskT> mask_merge_vec(image_height * image_width, 0);
      FOR_RANGE(size_t, j, 0, polys) {
        size_t poly_points = points_vec.at(polys_offset + j);
        // convert poly points to mask
        std::vector<uint8_t> mask_vec(mask_merge_vec.size(), 0);
        PolygonXy2ColMajorMask(poly_dptr, poly_points, image_height, image_width, mask_vec.data());
        std::transform(mask_vec.cbegin(), mask_vec.cend(), mask_merge_vec.begin(),
                       mask_merge_vec.begin(), std::bit_or<MaskT>());
        poly_dptr += poly_points * 2;
      }
      // cocoapi output mask is col-major, convert it to row-major
      KernelCtx ctx;
      std::vector<int32_t> perm_vec({1, 0});
      KernelUtil<DeviceType::kCPU, MaskT>::Transpose(
          ctx.device_ctx, 2,
          Shape({static_cast<int64_t>(image_width), static_cast<int64_t>(image_height)}),
          Shape({static_cast<int64_t>(image_height), static_cast<int64_t>(image_width)}),
          PbRf<int32_t>(perm_vec.begin(), perm_vec.end()), image_height * image_width,
          mask_merge_vec.data(), mask_dptr);
      // iter change
      polys_offset += polys;
      mask_dptr += image_height * image_width;
      segm_mask->IncreaseDataLength(image_height * image_width);
      // set lod
      segm_mask->AppendLodLength(1, image_height);
      FOR_RANGE(size_t, i, 0, image_height) { segm_mask->AppendLodLength(2, image_width); }
    }
    segm_mask->AppendLodLength(0, polys_vec.size());
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

// to simplify the implementation of model, pad mask the same way with image
template<>
struct DataTransformer<DataSourceCase::kObjectSegmentation,
                       TransformCase::kSegmentationPolyToAlignedMask> {
  using ImageFieldT = typename DataFieldTrait<DataSourceCase::kImage>::type;
  using SegmPolyFieldT = typename DataFieldTrait<DataSourceCase::kObjectSegmentation>::type;
  using SegmPaddedMaskFieldT =
      typename DataFieldTrait<DataSourceCase::kObjectSegmentationAlignedMask>::type;
  using T = typename SegmPolyFieldT::data_type;
  using MaskT = typename SegmPaddedMaskFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* image = dynamic_cast<ImageFieldT*>(data_inst->GetField<DataSourceCase::kImage>());
    auto* segm_poly =
        dynamic_cast<SegmPolyFieldT*>(data_inst->GetField<DataSourceCase::kObjectSegmentation>());
    auto* segm_mask = dynamic_cast<SegmPaddedMaskFieldT*>(
        data_inst->GetField<DataSourceCase::kObjectSegmentationAlignedMask>());
    if (!image || !segm_poly || !segm_mask) { return; }
    CHECK_EQ(segm_poly->Levels(), 3);

    const auto& polys_vec = segm_poly->GetLod(1);
    const auto& points_vec = segm_poly->GetLod(2);
    const int image_height = image->data().rows;
    const int image_width = image->data().cols;

    T* poly_dptr = segm_poly->data();
    MaskT* mask_dptr = segm_mask->data();
    size_t polys_offset = 0;
    FOR_RANGE(size_t, i, 0, polys_vec.size()) {
      size_t polys = polys_vec.at(i);
      std::vector<MaskT> overlap_mask_vec(image_height * image_width, 0);
      FOR_RANGE(size_t, j, 0, polys) {
        size_t poly_points = points_vec.at(polys_offset + j);
        // convert poly points to mask
        std::vector<uint8_t> mask_vec(overlap_mask_vec.size(), 0);
        DataTransformer<
            DataSourceCase::kObjectSegmentation,
            TransformCase::kSegmentationPolyToMask>::PolygonXy2ColMajorMask(poly_dptr, poly_points,
                                                                            image_height,
                                                                            image_width,
                                                                            mask_vec.data());
        std::transform(mask_vec.cbegin(), mask_vec.cend(), overlap_mask_vec.begin(),
                       overlap_mask_vec.begin(), std::bit_or<MaskT>());
        poly_dptr += poly_points * 2;
      }
      // cocoapi output mask is col-major, convert it to row-major
      KernelCtx ctx;
      std::vector<int32_t> perm_vec({1, 0});
      KernelUtil<DeviceType::kCPU, MaskT>::Transpose(
          ctx.device_ctx, 2,
          Shape({static_cast<int64_t>(image_width), static_cast<int64_t>(image_height)}),
          Shape({static_cast<int64_t>(image_height), static_cast<int64_t>(image_width)}),
          PbRf<int32_t>(perm_vec.begin(), perm_vec.end()), image_height * image_width,
          overlap_mask_vec.data(), mask_dptr);
      // iter change
      polys_offset += polys;
      mask_dptr += overlap_mask_vec.size();
      segm_mask->IncreaseSize(overlap_mask_vec.size());
    }
    segm_mask->SetShape(image_height, image_width);
  }
};

template<>
struct DataTransformer<DataSourceCase::kImage, TransformCase::kImageNormalizeByChannel> {
  using ImageFieldT = typename DataFieldTrait<DataSourceCase::kImage>::type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    auto* image_field = dynamic_cast<ImageFieldT*>(data_inst->GetField<DataSourceCase::kImage>());
    if (!image_field) { return; }

    auto& image_mat = image_field->data();
    CHECK_EQ(image_mat.type(), CV_8UC3);
    int channels = image_mat.channels();
    CHECK_EQ(proto.image_normalize_by_channel().mean_size(), channels);
    CHECK_EQ(proto.image_normalize_by_channel().std_size(), channels);
    int rows = image_mat.rows;
    int cols = image_mat.cols * channels;
    if (image_mat.isContinuous()) {
      cols *= rows;
      rows = 1;
    }

    image_mat.convertTo(image_mat, CV_32F);
    FOR_RANGE(int, i, 0, rows) {
      float* pixel = image_mat.ptr<float>(i);
      FOR_RANGE(int, j, 0, cols) {
        float mean = proto.image_normalize_by_channel().mean(j % channels);
        float std = proto.image_normalize_by_channel().std(j % channels);
        CHECK_GT(std, 0.0f);
        pixel[j] = (pixel[j] - mean) / std;
      }
    }
  }
};

template<>
struct DataTransformer<DataSourceCase::kImage, TransformCase::kImageRandomFlip> {
  using ImageFieldT = typename DataFieldTrait<DataSourceCase::kImage>::type;
  using BboxFieldT = typename DataFieldTrait<DataSourceCase::kObjectBoundingBox>::type;
  using SegmPolyFieldT = typename DataFieldTrait<DataSourceCase::kObjectSegmentation>::type;
  using BBoxT = typename BboxFieldT::data_type;
  using PolyT = typename SegmPolyFieldT::data_type;

  static void Apply(DataInstance* data_inst, const DataTransformProto& proto) {
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_int_distribution<int> dist(0, 99);
    float p = dist(gen) / 100.0f;
    if (p >= proto.image_random_flip().probability()) { return; }

    auto* image_field = dynamic_cast<ImageFieldT*>(data_inst->GetField<DataSourceCase::kImage>());
    if (!image_field) { return; }
    auto& image_mat = image_field->data();
    int flip_code = proto.image_random_flip().flip_code();
    cv::flip(image_mat, image_mat, flip_code);
    int image_height = image_mat.rows;
    int image_width = image_mat.cols;

    auto* bbox_field =
        dynamic_cast<BboxFieldT*>(data_inst->GetField<DataSourceCase::kObjectBoundingBox>());
    if (bbox_field) {
      auto& bbox_vec = bbox_field->data();
      CHECK_EQ(bbox_vec.size() % 4, 0);
      size_t num_bbox = bbox_vec.size() / 4;
      BBoxT* bbox_ptr = bbox_vec.data();
      FOR_RANGE(size_t, i, 0, num_bbox) {
        if (flip_code <= 0) {
          BBoxT ymin = bbox_ptr[1];
          BBoxT ymax = bbox_ptr[3];
          bbox_ptr[1] = image_height - ymax;
          bbox_ptr[3] = image_height - ymin;
        }
        if (flip_code != 0) {
          // why x axis coordinate removes 1 but y axis does not? (follow fb maskrcnn)
          const BBoxT TO_REMOVE = 1;
          BBoxT xmin = bbox_ptr[0];
          BBoxT xmax = bbox_ptr[2];
          bbox_ptr[0] = image_width - xmax - TO_REMOVE;
          bbox_ptr[2] = image_width - xmin - TO_REMOVE;
        }
        bbox_ptr += 4;
      }
    }

    auto* poly_field =
        dynamic_cast<SegmPolyFieldT*>(data_inst->GetField<DataSourceCase::kObjectSegmentation>());
    if (poly_field) {
      PolyT* poly_ptr = poly_field->data();
      FOR_RANGE(size_t, i, 0, poly_field->total_length()) {
        if (i % 2 == 0) {
          if (flip_code != 0) { poly_ptr[i] = image_width - poly_ptr[i]; }
        } else {
          if (flip_code <= 0) { poly_ptr[i] = image_height - poly_ptr[i]; }
        }
      }
    }
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

template<>
void DoDataTransform<TransformCase::kSegmentationPolyToAlignedMask>(
    DataInstance* data_inst, const DataTransformProto& proto) {
  CHECK(proto.has_segmentation_poly_to_aligned_mask());
  DataTransformer<DataSourceCase::kObjectSegmentation,
                  TransformCase::kSegmentationPolyToAlignedMask>::Apply(data_inst, proto);
}

template<>
void DoDataTransform<TransformCase::kImageNormalizeByChannel>(DataInstance* data_inst,
                                                              const DataTransformProto& proto) {
  CHECK(proto.has_image_normalize_by_channel());
  DataTransformer<DataSourceCase::kImage, TransformCase::kImageNormalizeByChannel>::Apply(data_inst,
                                                                                          proto);
}

template<>
void DoDataTransform<TransformCase::kImageRandomFlip>(DataInstance* data_inst,
                                                      const DataTransformProto& proto) {
  CHECK(proto.has_image_random_flip());
  DataTransformer<DataSourceCase::kImage, TransformCase::kImageRandomFlip>::Apply(data_inst, proto);
}

#define DEFINE_BATCH_TRANSFORM_PROCESS(trans)                                                   \
  template<>                                                                                    \
  void DoBatchTransform<trans>(std::shared_ptr<std::vector<DataInstance>> batch_data_inst_ptr,  \
                               const DataTransformProto& proto) {                               \
    for (auto& data_inst : *batch_data_inst_ptr) { DoDataTransform<trans>(&data_inst, proto); } \
  }

DEFINE_BATCH_TRANSFORM_PROCESS(TransformCase::kResize)
DEFINE_BATCH_TRANSFORM_PROCESS(TransformCase::kTargetResize)
DEFINE_BATCH_TRANSFORM_PROCESS(TransformCase::kSegmentationPolyToMask)
DEFINE_BATCH_TRANSFORM_PROCESS(TransformCase::kSegmentationPolyToAlignedMask)
DEFINE_BATCH_TRANSFORM_PROCESS(TransformCase::kImageNormalizeByChannel)
DEFINE_BATCH_TRANSFORM_PROCESS(TransformCase::kImageRandomFlip)

template<>
void DoBatchTransform<TransformCase::kImageAlign>(
    std::shared_ptr<std::vector<DataInstance>> batch_data_inst_ptr,
    const DataTransformProto& proto) {
  CHECK(proto.has_image_align());

  int64_t max_rows = -1;
  int64_t max_cols = -1;
  int64_t channels = -1;
  bool has_image_field = true;

  for (DataInstance& data_inst : *(batch_data_inst_ptr.get())) {
    auto* image_field = dynamic_cast<ImageDataField*>(data_inst.GetField<DataSourceCase::kImage>());
    if (image_field == nullptr) {
      has_image_field = false;
      break;
    }
    auto& image_mat = image_field->data();
    max_rows = std::max<int64_t>(max_rows, image_mat.rows);
    max_cols = std::max<int64_t>(max_cols, image_mat.cols);
    if (channels == -1) {
      channels = image_mat.channels();
    } else {
      CHECK_EQ(channels, image_mat.channels());
    }
  }
  if (!has_image_field) { return; }

  CHECK_GT(max_rows, 0);
  CHECK_GT(max_cols, 0);
  CHECK_GT(channels, 0);
  int alignment = proto.image_align().alignment();
  max_rows = RoundUp(max_rows, alignment);
  max_cols = RoundUp(max_cols, alignment);

  for (DataInstance& data_inst : *batch_data_inst_ptr) {
    auto* image_field = dynamic_cast<ImageDataField*>(data_inst.GetField<DataSourceCase::kImage>());
    CHECK_NOTNULL(image_field);
    auto& image_mat = image_field->data();
    cv::Mat dst = cv::Mat::zeros(cv::Size(max_cols, max_rows), image_mat.type());
    image_mat.copyTo(dst(cv::Rect(0, 0, image_mat.cols, image_mat.rows)));
    image_field->data() = dst;
  }
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
    MAKE_CASE(DataTransformProto::kImageNormalizeByChannel)
    MAKE_CASE(DataTransformProto::kImageRandomFlip)
    default: { UNIMPLEMENTED(); }
  }
#undef MAKE_CASE
}

void BatchTransform(std::shared_ptr<std::vector<DataInstance>> batch_data_inst_ptr,
                    const DataTransformProto& trans_proto) {
#define MAKE_CASE(trans)                                       \
  case trans: {                                                \
    DoBatchTransform<trans>(batch_data_inst_ptr, trans_proto); \
    break;                                                     \
  }

  switch (trans_proto.transform_case()) {
    MAKE_CASE(DataTransformProto::kResize)
    MAKE_CASE(DataTransformProto::kTargetResize)
    MAKE_CASE(DataTransformProto::kSegmentationPolyToMask)
    MAKE_CASE(DataTransformProto::kSegmentationPolyToAlignedMask)
    MAKE_CASE(DataTransformProto::kImageNormalizeByChannel)
    MAKE_CASE(DataTransformProto::kImageAlign)
    MAKE_CASE(DataTransformProto::kImageRandomFlip)
    default: { UNIMPLEMENTED(); }
  }
#undef MAKE_CASE
}

}  // namespace data
}  // namespace oneflow
