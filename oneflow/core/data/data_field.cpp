#include "oneflow/core/data/data_field.h"
#include "oneflow/core/kernel/kernel_util.h"
#include <glog/logging.h>

namespace oneflow {
namespace data {

size_t DataField::ToBuffer(void* buffer, DataType data_type) const {
  return GetDataFieldSerializer(data_source_, data_type)(this, buffer);
}

void ImageDataField::InferShape(const ShapeProto& static_shape,
                                const std::vector<int64_t>& var_axes, std::vector<int64_t>* shape,
                                std::vector<std::vector<int64_t>>* lod) const {
  int64_t image_height = data_.rows;
  int64_t image_width = data_.cols;
  int64_t channels = data_.depth();
  if (shape->size() == 0) {
    shape->resize(4, 1);
    shape->at(1) = image_height;
    shape->at(2) = image_width;
    shape->at(3) = channels;
  } else {
    CHECK_EQ(shape->at(1), image_height);
    CHECK_EQ(shape->at(2), image_width);
    CHECK_EQ(shape->at(3), channels);
    shape->at(0) += 1;
  }
}

template<typename T>
void ArrayDataField<T>::InferShape(const ShapeProto& static_shape,
                                   const std::vector<int64_t>& var_axes,
                                   std::vector<int64_t>* shape,
                                   std::vector<std::vector<int64_t>>* lod) const {
  CHECK_LE(var_axes.size(), 1);
  bool has_lod = (var_axes.size() == 1);
  if (has_lod) {
    int64_t need_infer_axis = var_axes.at(0);
    int64_t fixed = 1;
    int64_t pre_infer_dims = 1;
    if (shape->size() == 0) { shape->resize(static_shape.dim_size() - need_infer_axis, 0); }
    FOR_RANGE(int64_t, i, 0, static_shape.dim_size()) {
      if (i < need_infer_axis) { pre_infer_dims *= static_shape.dim(i); }
      if (i != need_infer_axis) { fixed *= static_shape.dim(i); }
    }
    CHECK_EQ(data_.size() % fixed, 0);
    int64_t infered_dims = data_.size() / fixed;
    shape->at(0) += pre_infer_dims * infered_dims;
    FOR_RANGE(int64_t, i, 1, static_shape.dim_size() - need_infer_axis) {
      if (shape->at(i) == 0) {
        shape->at(i) = static_shape.dim(i + need_infer_axis);
      } else {
        CHECK_EQ(shape->at(i), static_shape.dim(i + need_infer_axis));
      }
    }

    if (lod->size() == 0) {
      lod->resize(need_infer_axis + 2, {});
      lod->at(0).push_back(0);
    }
    lod->at(0).at(0) += 1;
    lod->at(1).push_back(static_shape.dim(0));
    FOR_RANGE(int64_t, i, 1, need_infer_axis + 1) {
      FOR_RANGE(int64_t, j, 0, static_shape.dim(i - 1)) {
        if (i == need_infer_axis) {
          lod->at(i + 1).push_back(infered_dims);
        } else {
          lod->at(i + 1).push_back(static_shape.dim(i));
        }
      }
    }
  } else {
    if (shape->size() == 0) { shape->resize(static_shape.dim_size() + 1, 0); }
    shape->at(0) += 1;
    FOR_RANGE(int64_t, i, 0, static_shape.dim_size()) {
      if (shape->at(i + 1) == 0) {
        shape->at(i + 1) = static_shape.dim(i);
      } else {
        CHECK_EQ(shape->at(i + 1), static_shape.dim(i));
      }
    }
  }
}

template<typename T>
void NdarrayDataField<T>::InferShape(const ShapeProto& static_shape,
                                     const std::vector<int64_t>& var_axes,
                                     std::vector<int64_t>* shape,
                                     std::vector<std::vector<int64_t>>* lod) const {
  TODO();
}

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

#define INSTANTIATE_INFER_SHAPE_FUNC(dtype, dtype_val)                                            \
  template void ArrayDataField<dtype>::InferShape(const ShapeProto&, const std::vector<int64_t>&, \
                                                  std::vector<int64_t>*,                          \
                                                  std::vector<std::vector<int64_t>>*) const;      \
  template void NdarrayDataField<dtype>::InferShape(                                              \
      const ShapeProto&, const std::vector<int64_t>&, std::vector<int64_t>*,                      \
      std::vector<std::vector<int64_t>>*) const;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_INFER_SHAPE_FUNC, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace data
}  // namespace oneflow
