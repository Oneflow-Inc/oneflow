#include "oneflow/core/persistence/data_encode.h"
#include <opencv2/opencv.hpp>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/persistence/data_set_format.h"

namespace oneflow {

template<>
template<typename T>
void RecordDecoder<DataEncodeType::kNoEncode>::Decode(const Record& record,
                                                      const Shape& shape,
                                                      T* out_dptr) {
}

template<>
template<typename T>
void RecordDecoder<DataEncodeType::kJpeg>::Decode(const Record& record,
                                                  const Shape& shape,
                                                  T* out_dptr) {
  int shape_chanels = shape.At(1);
  int width = shape.At(2);
  int height = shape.At(3);
  const char* buf = record.value_buffer();
  size_t len = record.value_buffer_len();
  cv::Mat img = cv::imdecode(
      cv::_InputArray(buf, len),
      (shape_chanels == 3 ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE));
  cv::Mat resized;
  uint64_t size = width * height;
  cv::resize(img, resized, cv::Size(width, height));
  for (int c = 0; c < shape_chanels; ++c) {
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
void RecordDecoder<DataEncodeType::kSparse>::Decode(const Record& record,
                                                    const Shape& shape,
                                                    T* out_dptr) {
  UNEXPECTED_RUN();
}

namespace {

// only useful for compiling
void SepecializeTemplate() {
#define SPECIALIZE_RECORD_DECODE(encode_type, type_pair)             \
  RecordDecoder<encode_type>::Decode<OF_PP_PAIR_FIRST(type_pair)>(   \
      *static_cast<Record*>(nullptr), *static_cast<Shape*>(nullptr), \
      static_cast<OF_PP_PAIR_FIRST(type_pair)*>(nullptr));
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(SPECIALIZE_RECORD_DECODE,
                                   DATA_ENCODE_TYPE_SEQ, ALL_DATA_TYPE_SEQ)
}

}  // namespace

}  // namespace oneflow
