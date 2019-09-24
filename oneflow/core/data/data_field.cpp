#include "oneflow/core/data/data_field.h"
#include "oneflow/core/kernel/kernel_util.h"
#include <glog/logging.h>

namespace oneflow {
namespace data {

size_t DataField::ToBuffer(void* buffer, DataType data_type) const {
  return GetDataFieldSerializer(data_source_, data_type)(this, buffer);
}

// void ImageDataField::InferShape(Blob* blob, const ShapeProto& shape,
//                                 const PbRf<int32_t>& flex_axes) const {
//   int64_t image_height = image_mat_.rows;
//   int64_t image_width = image_mat_.cols;
//   int64_t channels = image_mat_.depth();
// }

// void ArrayDataField<T>::InferShape() {}

template<typename K, typename T>
struct DataFieldSerializer<ArrayDataField<K>, T> {
  static size_t Apply(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const ArrayDataField<T>*>(data_field);
    CHECK_NOTNULL(field);
    const auto& vec = field->data();
    CopyElem(vec.data(), buffer, vec.size());
    return vec.size() * sizeof(T);
  }
};

template<typename T>
struct DataFieldSerializer<ImageDataField, T> {
  static size_t Apply(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const ImageDataField*>(data_field);
    CHECK_NOTNULL(field);
    const auto& image_mat = field->data();
    const int64_t total_pixels = image_mat.total() * image_mat.channels();
    if (image_mat.isContinuous()) {
      CopyElem(image_mat.data, buffer, total_pixels);
    } else {
      FOR_RANGE(size_t, i, 0, image_mat.rows) {
        size_t one_row_size = image_mat.cols * image_mat.channels();
        CopyElem(image_mat.ptr<uint8_t>(i), buffer, one_row_size);
        buffer += one_row_size;
      }
    }
    return total_pixels * sizeof(T);
  }
};

template<typename K, typename T>
struct DataFieldSerializer<NdarrayDataField<K>, T> {
  static size_t Apply(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const NdarrayDataField<K>*>(data_field);
    CHECK_NOTNULL(field);
    CopyElem(field->data(), buffer, field->total_length());
    return field->total_length();
  }
};

std::function<size_t(const DataField*, void*)> GetDataFieldSerializer(DataSourceCase dsrc,
                                                                      DataType dtype) {
#define MAKE_ENTRY(dsrc, dtype)                                                                    \
  {GetHashKey(dsrc, dtype), [](const DataField* field, void* buffer) {                             \
     using DataFieldT = typename DataFieldTrait<dsrc>::type;                                       \
     using BufferT = DataTypeToType<dtype>;                                                        \
     return DataFieldSerializer<DataFieldT, BufferT>::Apply(field, static_cast<BufferT*>(buffer)); \
   }},
  static HashMap<std::string, std::function<size_t(const DataField*, void*)>> serializers = {
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, DATA_FIELD_SERIALIZER_TUPLE_SEQ)};
#undef MAKE_ENTRY
  return serializers.at(GetHashKey(dsrc, dtype));
}

std::unique_ptr<DataField> CreateDataFieldFromProto(const DataFieldProto& proto) {
  std::unique_ptr<DataField> data_field_ptr;
  switch (proto.data_source()) {
    case DataSourceCase::kImage: {
      data_field_ptr.reset(new ImageDataField());
      break;
    }
    case DataSourceCase::kLabel: {
      data_field_ptr.reset(new ArrayDataField<int32_t>());
      break;
    }
    case DataSourceCase::kImageScale:
    case DataSourceCase::kObjectBoundingBox: {
      data_field_ptr.reset(new ArrayDataField<float>());
      break;
    }
    case DataSourceCase::kObjectSegmentation: {
      int64_t max_elem_cnt = Shape(proto.shape()).elem_cnt();
      data_field_ptr.reset(new NdarrayDataField<float>(max_elem_cnt));
      break;
    }
    case DataSourceCase::kObjectLabel: {
      data_field_ptr.reset(new ArrayDataField<int32_t>());
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
  data_field_ptr->SetSource(proto.data_source());
  return data_field_ptr;
}

}  // namespace data
}  // namespace oneflow
