#include "oneflow/core/record/ofrecord_jpeg_encoder.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

template<typename T>
void OFRecordEncoderImpl<EncodeCase::kJpeg, T>::EncodeOneCol(DeviceCtx* ctx, const Blob* in_blob,
                                                             int64_t in_offset, Feature& feature,
                                                             const std::string& field_name,
                                                             int64_t one_col_elem_num) const {
  const T* in_dptr = in_blob->dptr<T>() + in_offset;
  DataType data_type = GetDataType<T>();
  // shape must be N * H * W * C
  const ShapeView& shape = in_blob->shape();
  CHECK(shape.NumAxes() == 4);
  int type = -1;
  if (data_type == DataType::kInt8 && shape.At(3) == 1) {
    type = CV_8SC1;
  } else if (data_type == DataType::kInt8 && shape.At(3) == 3) {
    type = CV_8SC3;
  } else if (data_type == DataType::kInt32 && shape.At(3) == 1) {
    type = CV_32SC1;
  } else if (data_type == DataType::kInt32 && shape.At(3) == 3) {
    type = CV_32SC3;
  } else if (data_type == DataType::kFloat && shape.At(3) == 1) {
    type = CV_32FC1;
  } else if (data_type == DataType::kFloat && shape.At(3) == 3) {
    type = CV_32FC3;
  } else if (data_type == DataType::kDouble && shape.At(3) == 1) {
    type = CV_64FC1;
  } else if (data_type == DataType::kDouble && shape.At(3) == 3) {
    type = CV_64FC3;
  } else {
    UNIMPLEMENTED();
  }
  std::vector<unsigned char> buf;
  cv::Mat img = cv::Mat(shape.At(1), shape.At(2), type, const_cast<T*>(in_dptr)).clone();
  cv::imencode(".jpg", img, buf, {});
  feature.mutable_bytes_list()->add_value(reinterpret_cast<const char*>(buf.data()), buf.size());
}

#define INSTANTIATE_OFRECORD_JPEG_ENCODER(type_cpp, type_proto) \
  template class OFRecordEncoderImpl<EncodeCase::kJpeg, type_cpp>;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_OFRECORD_JPEG_ENCODER, ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace oneflow
