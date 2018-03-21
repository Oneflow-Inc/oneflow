#include "oneflow/core/record/ofrecord_jpeg_decoder.h"
#include "oneflow/core/record/image.pb.h"

namespace oneflow {

using PreprocessCase = ImagePreprocess::PreprocessCase;

namespace {

void DoPreprocess(std::unique_ptr<uint8_t[]>* image_data, Shape* image_shape,
                  const ImagePreprocess& preprocess_conf) {
  TODO();
}

}  // namespace

template<typename T>
int32_t OFRecordDecoderImpl<EncodeCase::kJpeg, T>::GetColNumOfFeature(
    const Feature& feature, int64_t one_col_elem_num) const {
  // sigle jpeg just has 1 column
  return 0;
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kJpeg, T>::ReadOneCol(
    DeviceCtx* ctx, const Feature& feature, const BlobConf& blob_conf,
    int32_t col_id, T* out_dptr, int64_t one_col_elem_num) const {
  CHECK(feature.has_bytes_list());
  CHECK_EQ(feature.bytes_list().value_size(), 1);
  std::unique_ptr<uint8_t[]> image_data = nullptr;
  Shape image_shape;
  DecodeImage(feature.bytes_list().value(0), &image_data, &image_shape);
  FOR_RANGE(size_t, i, 0, blob_conf.jpeg().preprocess_size()) {
    DoPreprocess(&image_data, &image_shape, blob_conf.jpeg().preprocess(i));
  }
  CHECK_EQ(image_shape, Shape(blob_conf.shape()));
  uint8_t* in_dptr = image_data.get();
  FOR_RANGE(int64_t, i, 0, one_col_elem_num) {
    *(out_dptr++) = static_cast<T>(*(in_dptr++));
  }
}

template<typename T>
void OFRecordDecoderImpl<EncodeCase::kJpeg, T>::DecodeImage(
    const std::string& src_data, std::unique_ptr<uint8_t[]>* image_data,
    Shape* image_shape) const {
  TODO();
}

#define INSTANTIATE_OFRECORD_JPEG_DECODER(type_cpp, type_proto) \
  template class OFRecordDecoderImpl<EncodeCase::kJpeg, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_JPEG_DECODER,
                     ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
