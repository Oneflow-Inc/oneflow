#include "oneflow/core/data/data_field.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/register/lod_view.h"
#include <glog/logging.h>

namespace oneflow {

namespace data {

namespace {

void BuildLodTree(const int64_t level, const int64_t max_level, const int64_t elem_cnt,
                  const Shape& shape, LoDTree* node, int64_t& offset) {
  if (level < max_level) {
    FOR_RANGE(int64_t, i, 0, shape.At(level)) {
      auto* child = node->mutable_children()->Add();
      BuildLodTree(level + 1, max_level, elem_cnt, shape, child, offset);
    }
    TreeLoDHelper::UpdateInnerNode(node);
  } else {
    int64_t n = shape.Count(0, max_level);
    int64_t m = shape.Count(max_level + 1);
    int64_t fixed = n * m;
    CHECK_EQ(elem_cnt % fixed, 0);
    int64_t length = elem_cnt / fixed;
    node->set_length(length);
    node->set_offset(offset);
    offset += length * m;
  }
}

}  // namespace

size_t DataField::ToBuffer(void* buffer, DataType data_type) const {
  return GetDataFieldSerializer(data_source_, data_type)(this, buffer);
}

void ImageDataField::InferShape(const ShapeProto& shape_proto, const PbRf<int>& var_axes,
                                Shape* shape, LoDTree* lod_tree) const {
  CHECK_LE(var_axes.size(), 0);
  CHECK(lod_tree == nullptr);

  *shape = Shape(shape_proto);
  CHECK_EQ(shape->NumAxes(), 3);
  shape->Set(0, data_.rows);
  shape->Set(1, data_.cols);
  shape->Set(2, data_.channels());
}

template<typename T>
void ArrayDataField<T>::InferShape(const ShapeProto& shape_proto, const PbRf<int>& var_axes,
                                   Shape* shape, LoDTree* lod_tree) const {
  if (lod_tree) {
    CHECK_EQ(var_axes.size(), 1);
    int64_t lod_axis = var_axes.Get(0);
    Shape static_shape(shape_proto);

    *shape = Shape::Ones(shape_proto.dim_size() - lod_axis);
    FOR_RANGE(int64_t, i, lod_axis + 1, shape_proto.dim_size()) {
      shape->Set(i - lod_axis, shape_proto.dim(i));
    }
    int64_t fixed = static_shape.elem_cnt() / static_shape.At(lod_axis);
    CHECK_EQ(data_.size() % fixed, 0);
    int64_t var_len = data_.size() / fixed;
    shape->Set(0, var_len * static_shape.Count(0, lod_axis));

    int64_t offset = 0;
    BuildLodTree(0, lod_axis, data_.size(), static_shape, lod_tree, offset);
  } else {
    CHECK_EQ(var_axes.size(), 0);
    *shape = Shape(shape_proto);
    CHECK_EQ(shape->elem_cnt(), data_.size());
  }
}

template<typename T>
void NdarrayDataField<T>::InferShape(const ShapeProto& shape_proto, const PbRf<int>& var_axes,
                                     Shape* shape, LoDTree* lod_tree) const {
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
    case DataSourceCase::kLabel:
    case DataSourceCase::kObjectLabel:
    case DataSourceCase::kImageSize: {
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
    case DataSourceCase::kObjectSegmentationMask: {
      data_field_ptr.reset(new ArrayDataField<int8_t>());
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
  data_field_ptr->SetSource(proto.data_source());
  return data_field_ptr;
}

#define INSTANTIATE_INFER_SHAPE_FUNC(dtype, dtype_val)                                           \
  template void ArrayDataField<dtype>::InferShape(                                               \
      const ShapeProto& shape_proto, const PbRf<int>& var_axes, Shape* shape, LoDTree* lod_tree) \
      const;                                                                                     \
  template void NdarrayDataField<dtype>::InferShape(                                             \
      const ShapeProto& shape_proto, const PbRf<int>& var_axes, Shape* shape, LoDTree* lod_tree) \
      const;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_INFER_SHAPE_FUNC, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace data
}  // namespace oneflow
