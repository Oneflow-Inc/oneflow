#include <opencv2/opencv.hpp>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/persistence/ubf_item.h"

namespace oneflow {

template<>
template<typename src_type, typename T>
void UbfDecoder<DataEncodeType::kNoEncode>::Cast(const UbfItem& ubf_item,
                                                 const Shape& shape,
                                                 T* out_dptr) {
  CHECK(ubf_item.data_encode_type() == DataEncodeType::kNoEncode);
  CHECK(ubf_item.data_type() == GetDataType<src_type>::val);
  CHECK(sizeof(src_type) * shape.Count(1) == ubf_item.value_buffer_len());
  auto data = reinterpret_cast<const src_type*>(ubf_item.value_buffer());
  for (int64_t i = 0; i < shape.Count(1); ++i) { out_dptr[i] = data[i]; }
}

template<>
template<typename T>
void UbfDecoder<DataEncodeType::kNoEncode>::Decode(const UbfItem& ubf_item,
                                                   const Shape& shape,
                                                   T* out_dptr) {
  CHECK(ubf_item.data_encode_type() == DataEncodeType::kNoEncode);
  switch (ubf_item.data_type()) {
#define OFB_DECODE_RAW_DATA_ENTRY(type, type_case) \
  case type_case: return Cast<type, T>(ubf_item, shape, out_dptr);
    OF_PP_FOR_EACH_TUPLE(OFB_DECODE_RAW_DATA_ENTRY, ALL_DATA_TYPE_SEQ)
    default: UNEXPECTED_RUN();
  }
}

template<>
template<typename T>
void UbfDecoder<DataEncodeType::kJpeg>::Decode(const UbfItem& ubf_item,
                                               const Shape& shape,
                                               T* out_dptr) {
  CHECK(ubf_item.data_encode_type() == DataEncodeType::kJpeg);
  int shape_chanels = shape.At(1);
  CHECK(shape_chanels == 3 || shape_chanels == 1);
  int width = shape.At(2);
  int height = shape.At(3);
  cv::Mat img = cv::imdecode(
      cv::_InputArray(ubf_item.value_buffer(), ubf_item.value_buffer_len()),
      (shape_chanels == 3 ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE));
  cv::Mat resized;
  uint64_t size = width * height;
  cv::resize(img, resized, cv::Size(width, height));
  for (int64_t c = 0; c < shape_chanels; ++c) {
    for (int row = 0; row < height; ++row) {
      for (int col = 0; col < width; ++col) {
        out_dptr[c * size + row * width + col] =
            resized.at<cv::Vec3b>(row, col)[c];
      }
    }
  }
}

template<>
template<typename T>
void UbfDecoder<DataEncodeType::kSparse>::Decode(const UbfItem& ubf_item,
                                                 const Shape& shape,
                                                 T* out_dptr) {
  CHECK(ubf_item.data_encode_type() == DataEncodeType::kSparse);
  UNEXPECTED_RUN();
}

namespace {

// only useful for compiling
void SepecializeTemplate() {
#define SPECIALIZE_OFB_DECODE(encode_type, type_pair)                 \
  UbfDecoder<encode_type>::Decode(                                    \
      *static_cast<UbfItem*>(nullptr), *static_cast<Shape*>(nullptr), \
      static_cast<OF_PP_PAIR_FIRST(type_pair)*>(nullptr));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SPECIALIZE_OFB_DECODE, DATA_ENCODE_TYPE_SEQ,
                                   ALL_DATA_TYPE_SEQ)
}

}  // namespace

}  // namespace oneflow
