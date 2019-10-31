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

void BuildLodTreeFromShape(const int64_t max_level, const int64_t elem_cnt, const Shape& shape,
                           LoDTree* lod_tree) {
  int64_t offset = 0;
  BuildLodTree(0, max_level, elem_cnt, shape, lod_tree, offset);
  TreeLoDHelper::UpdateInnerNode(lod_tree);
}

// TODO: Implementation of BuildLodTreeFromNestedVector is similar to
// codes in TreeLoDView::Init(), consider make a general indenpendent impl
void BuildLodTreeFromNestedVector(const std::vector<std::vector<size_t>>& lod_nested_vec,
                                  LoDTree* lod_tree) {
  int64_t offset = 0;
  FOR_RANGE(size_t, level, 0, lod_nested_vec.size()) {
    std::vector<LoDTree*> cur_level_subtrees;
    TreeLoDHelper::FindLevelMutNodes(level, lod_tree, &cur_level_subtrees);
    CHECK_EQ(lod_nested_vec.at(level).size(), cur_level_subtrees.size());

    FOR_RANGE(int64_t, i, 0, cur_level_subtrees.size()) {
      int64_t length = lod_nested_vec.at(level).at(i);
      LoDTree* sub_tree = cur_level_subtrees.at(i);
      if (level == lod_nested_vec.size() - 1) {
        sub_tree->set_offset(offset);
        sub_tree->set_length(length);
        offset += length;
      } else {
        FOR_RANGE(int64_t, _, 0, length) { sub_tree->mutable_children()->Add(); }
      }
    }
  }
  TreeLoDHelper::UpdateInnerNode(lod_tree);
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
    CHECK_NOTNULL(lod_tree);
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

    BuildLodTreeFromShape(lod_axis, data_.size(), static_shape, lod_tree);
  } else {
    CHECK_EQ(var_axes.size(), 0);
    *shape = Shape(shape_proto);
    CHECK_EQ(shape->elem_cnt(), data_.size());
  }
}

template<typename T>
void NdarrayDataField<T>::InferShape(const ShapeProto& shape_proto, const PbRf<int>& var_axes,
                                     Shape* shape, LoDTree* lod_tree) const {
  CHECK_NOTNULL(lod_tree);
  CHECK_EQ(var_axes.size(), lod_len_.size());
  CHECK_LE(lod_len_.size(), shape_proto.dim_size());

  const auto& last_level_length_vec = lod_len_.at(lod_len_.size() - 1);
  size_t total_length = std::accumulate(last_level_length_vec.begin(), last_level_length_vec.end(),
                                        0, std::plus<size_t>());
  DimVector shape_vec;
  shape_vec.push_back(total_length);
  int last_var_axis = var_axes.Get(var_axes.size() - 1);
  FOR_RANGE(int, i, last_var_axis + 1, shape_proto.dim_size()) {
    shape_vec.push_back(shape_proto.dim(i));
  }
  *shape = Shape(shape_vec);
  BuildLodTreeFromNestedVector(lod_len_, lod_tree);
}

template<typename T>
void TensorArrayDataField<T>::InferShape(const ShapeProto& shape_proto, const PbRf<int>& var_axes,
                                         Shape* shape, LoDTree* lod_tree) const {
  CHECK_EQ(var_axes.size(), 1);
  CHECK_NOTNULL(lod_tree);
  int var_axis = var_axes.Get(0);
  CHECK_EQ(var_axis, 0);

  Shape static_shape(shape_proto);
  DimVector shape_vec = this->shape();
  int64_t length = this->ArrayLength();
  shape_vec.insert(shape_vec.begin(), length);
  *shape = Shape(shape_vec);
  CHECK_GE(static_shape.Count(var_axis + 1), shape->Count(1));
  CHECK_GE(static_shape.At(var_axis), length);
  lod_tree->set_length(length);
  lod_tree->set_offset(0);
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
struct DataFieldSerializer<NdarrayDataField<K>, T> {
  static size_t Apply(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const NdarrayDataField<K>*>(data_field);
    CHECK_NOTNULL(field);
    CopyElem(field->data(), buffer, field->total_length());
    return field->total_length() * sizeof(T);
  }
};

template<typename K, typename T>
struct DataFieldSerializer<TensorArrayDataField<K>, T> {
  static size_t Apply(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const TensorArrayDataField<K>*>(data_field);
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
      data_field_ptr.reset(new ArrayDataField<int64_t>());
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
      data_field_ptr.reset(new NdarrayDataField<double>(max_elem_cnt));
      break;
    }
    case DataSourceCase::kObjectSegmentationMask: {
      int64_t max_elem_cnt = Shape(proto.shape()).elem_cnt();
      data_field_ptr.reset(new NdarrayDataField<int8_t>(max_elem_cnt));
      break;
    }
    case DataSourceCase::kObjectSegmentationAlignedMask: {
      int64_t max_elem_cnt = Shape(proto.shape()).elem_cnt();
      data_field_ptr.reset(new TensorArrayDataField<int8_t>(max_elem_cnt));
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
      const;                                                                                     \
  template void TensorArrayDataField<dtype>::InferShape(                                         \
      const ShapeProto& shape_proto, const PbRf<int>& var_axes, Shape* shape, LoDTree* lod_tree) \
      const;

OF_PP_FOR_EACH_TUPLE(INSTANTIATE_INFER_SHAPE_FUNC, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace data
}  // namespace oneflow
