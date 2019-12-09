#include "oneflow/core/data/data_field.h"
#include "oneflow/core/kernel/kernel_util.h"
#include <glog/logging.h>

namespace oneflow {

namespace data {

size_t DataField::ToBuffer(void* buffer, DataType data_type) const {
  return GetDataFieldSerializer(data_source_, data_type)(this, buffer);
}

void ImageDataField::InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                                Shape* shape) const {
  CHECK(!var_axis.has_value());
  *shape = Shape(shape_proto);
  CHECK_EQ(shape->NumAxes(), 3);
  shape->Set(0, data_.rows);
  shape->Set(1, data_.cols);
  shape->Set(2, data_.channels());
}

template<typename T>
void TensorDataField<T>::InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                                    Shape* shape) const {
  CHECK(!var_axis.has_value());
  *shape = Shape(shape_proto);
  CHECK_EQ(shape->elem_cnt(), data_.size());
}

template<typename T>
void LoDDataField<T>::InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                                 Shape* shape) const {
  UNIMPLEMENTED();
}

template<typename T>
void TensorListDataField<T>::InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                                        Shape* shape) const {
  CHECK(var_axis.has_value());
  CHECK_EQ(var_axis.value(), 0);
  Shape static_shape(shape_proto);
  DimVector shape_vec = this->shape();
  int64_t length = this->ArrayLength();
  shape_vec.insert(shape_vec.begin(), length);
  *shape = Shape(shape_vec);
  CHECK_GE(static_shape.Count(1), shape->Count(1));
  CHECK_GE(static_shape.At(0), length);
}

template<typename K, typename T>
struct DataFieldSerializer<TensorDataField<K>, T> {
  static size_t Apply(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const TensorDataField<T>*>(data_field);
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

    int channels = image_mat.channels();
    int rows = image_mat.rows;
    int cols = image_mat.cols * channels;
    if (image_mat.isContinuous()) {
      cols *= rows;
      rows = 1;
    }
    FOR_RANGE(int, i, 0, rows) {
      switch (image_mat.depth()) {
        case CV_8U: {
          CopyElem(image_mat.ptr<uint8_t>(i), buffer, cols);
          break;
        }
        case CV_32F: {
          CopyElem(image_mat.ptr<float>(i), buffer, cols);
          break;
        }
        default: { UNIMPLEMENTED(); }
      }
      buffer += cols;
    }
    return rows * cols * sizeof(T);
  }
};

template<typename K, typename T>
struct DataFieldSerializer<LoDDataField<K>, T> {
  static size_t Apply(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const LoDDataField<K>*>(data_field);
    CHECK_NOTNULL(field);
    CopyElem(field->data(), buffer, field->total_length());
    return field->total_length() * sizeof(T);
  }
};

template<typename K, typename T>
struct DataFieldSerializer<TensorListDataField<K>, T> {
  static size_t Apply(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const TensorListDataField<K>*>(data_field);
    CHECK_NOTNULL(field);
    CopyElem(field->data(), buffer, field->size());
    return field->size() * sizeof(T);
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
    case DataSourceCase::kImageId: {
      data_field_ptr.reset(new TensorDataField<int64_t>());
      break;
    }
    case DataSourceCase::kLabel:
    case DataSourceCase::kImageSize: {
      data_field_ptr.reset(new TensorDataField<int32_t>());
      break;
    }
    case DataSourceCase::kImageScale: {
      data_field_ptr.reset(new TensorDataField<float>());
      break;
    }
    case DataSourceCase::kObjectSegmentation: {
      int64_t max_elem_cnt = Shape(proto.shape()).elem_cnt();
      data_field_ptr.reset(new LoDDataField<double>(max_elem_cnt));
      break;
    }
    case DataSourceCase::kObjectLabel: {
      int64_t max_elem_cnt = Shape(proto.shape()).elem_cnt();
      data_field_ptr.reset(new TensorListDataField<int32_t>(max_elem_cnt));
      break;
    }
    case DataSourceCase::kObjectBoundingBox: {
      int64_t max_elem_cnt = Shape(proto.shape()).elem_cnt();
      data_field_ptr.reset(new TensorListDataField<float>(max_elem_cnt));
      break;
    }
    case DataSourceCase::kObjectSegmentationAlignedMask: {
      int64_t max_elem_cnt = Shape(proto.shape()).elem_cnt();
      data_field_ptr.reset(new TensorListDataField<int8_t>(max_elem_cnt));
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
  data_field_ptr->SetSource(proto.data_source());
  return data_field_ptr;
}

#define INSTANTIATE_INFER_SHAPE_FUNC(dtype, dtype_val)                                            \
  template void TensorDataField<dtype>::InferShape(const ShapeProto& shape_proto,                 \
                                                   const OptInt64& var_axis, Shape* shape) const; \
  template void LoDDataField<dtype>::InferShape(const ShapeProto& shape_proto,                    \
                                                const OptInt64& var_axis, Shape* shape) const;    \
  template void TensorListDataField<dtype>::InferShape(                                           \
      const ShapeProto& shape_proto, const OptInt64& var_axis, Shape* shape) const;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_INFER_SHAPE_FUNC, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace data
}  // namespace oneflow
