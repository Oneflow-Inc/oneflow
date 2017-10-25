#include "oneflow/core/persistence/ubf_item.h"
#include <opencv2/opencv.hpp>

namespace oneflow {

namespace {

template<DataEncodeType encode_type>
class UbfDecoder final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UbfDecoder);
  UbfDecoder() = delete;
  template<typename T>
  static void Decode(const UbfItem& ubf_item, const Shape& shape, T* out_dptr);

 private:
  template<typename src_type, typename T>
  static void Cast(const UbfItem& ubf_item, const Shape& shape, T* out_dptr);
};

template<>
template<typename src_type, typename T>
void UbfDecoder<DataEncodeType::kNoEncode>::Cast(const UbfItem& ubf_item,
                                                 const Shape& shape,
                                                 T* out_dptr) {
  CHECK(ubf_item.data_encode_type() == DataEncodeType::kNoEncode);
  CHECK(ubf_item.data_type() == GetDataType<src_type>::val);
  CHECK(sizeof(src_type) * shape.Count(1) == ubf_item.body_len());
  auto body = reinterpret_cast<const src_type*>(ubf_item.body());
  for (int64_t i = 0; i < shape.Count(1); ++i) { out_dptr[i] = body[i]; }
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
      cv::_InputArray(ubf_item.body(), ubf_item.body_len()),
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

}  // namespace

UbfItem::UbfItem(DataType dtype, DataEncodeType detype,
                 const std::string& data_id, size_t body_len,
                 const std::function<void(char*)>& Fill)
    : desc_(
          of_make_unique<UbfItemDesc>(dtype, detype, data_id.size(), body_len)),
      data_(std::unique_ptr<char[]>(new char[data_id.size() + body_len])) {
  memset(mut_data_id(), 0, desc()->body_offset());
  data_id.copy(mut_data_id(), data_id.size());
  if (body_len) { Fill(mut_body()); }
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

PersistentOutStream& operator<<(PersistentOutStream& out, const UbfItem& data) {
  out.Write(reinterpret_cast<const char*>(data.desc()), sizeof(*data.desc()));
  out.Write(reinterpret_cast<const char*>(data.data()), data.len());
  return out;
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
