#include "oneflow/core/persistence/ubf_item.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

PersistentOutStream& operator<<(PersistentOutStream& out, const UbfItem& data) {
  out.Write(reinterpret_cast<const char*>(&data),
            Flexible<UbfItem>::SizeOf(data));
  return out;
}

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

std::unique_ptr<UbfItem, decltype(&free)> UbfItem::New(
    const std::string& key, size_t value_buf_len, DataType dtype,
    DataEncodeType detype, const std::function<void(char* buff)>& Fill) {
  size_t value_offset = RoundUpToAlignment(key.size(), 8);
  auto ubf_item = Flexible<UbfItem>::Malloc(value_buf_len + value_offset);
  ubf_item->data_type_ = dtype;
  ubf_item->data_encode_type_ = detype;
  ubf_item->key_len_ = key.size();
  ubf_item->value_offset_ = value_offset;
  memset(ubf_item->data_, 0, value_offset);
  key.copy(ubf_item->mut_key_buffer(), key.size());
  if (value_buf_len) { Fill(const_cast<char*>(ubf_item->mut_value_buffer())); }
  ubf_item->UpdateCheckSum();
  return ubf_item;
}

uint8_t UbfItem::ComputeMetaCheckSum() const {
  uint8_t chk_sum = 0;
  int meta_len = Flexible<UbfItem>::SizeOf(0);
  for (int i = 0; i < meta_len; ++i) {
    chk_sum += reinterpret_cast<const char*>(this)[i];
  }
  return chk_sum;
}

void UbfItem::UpdateMetaCheckSum() {
  uint8_t meta_chk_sum = ComputeMetaCheckSum();
  meta_chk_sum -= meta_check_sum_;
  meta_check_sum_ = -meta_chk_sum;
}

uint8_t UbfItem::ComputeDataCheckSum() const {
  uint8_t data_chk_sum = 0;
  for (int i = 0; i < len_; ++i) { data_chk_sum += data_[i]; }
  return data_chk_sum;
}

void UbfItem::UpdateCheckSum() {
  UpdateDataCheckSum();
  UpdateMetaCheckSum();
}

void UbfItem::UpdateDataCheckSum() {
  uint8_t data_chk_sum = ComputeDataCheckSum();
  data_check_sum_ = -data_chk_sum;
}

std::string UbfItem::GetDataId() const {
  return std::string(key_buffer(), key_buffer_len());
}

template<typename T>
void UbfItem::Decode(const Shape& shape, T* out_dptr) const {
  switch (data_encode_type()) {
#define UBF_ITEM_DECODE_ENTRY(encode_type)                               \
  case DataEncodeType::encode_type:                                      \
    return UbfDecoder<DataEncodeType::encode_type>::Decode(*this, shape, \
                                                           out_dptr);
    OF_PP_FOR_EACH_TUPLE(UBF_ITEM_DECODE_ENTRY, DATA_ENCODE_TYPE_SEQ)
    default: UNEXPECTED_RUN();
  }
}

namespace {

//  it's only usefull for compiling
void SepcializeTemplate() {
#define SPECIALIZE_UBF_ITEM_DECODE(type, type_case)                     \
  static_cast<UbfItem*>(nullptr)->Decode(*static_cast<Shape*>(nullptr), \
                                         static_cast<type*>(nullptr));
  OF_PP_FOR_EACH_TUPLE(SPECIALIZE_UBF_ITEM_DECODE, ALL_DATA_TYPE_SEQ)
}

}  // namespace

}  // namespace oneflow
