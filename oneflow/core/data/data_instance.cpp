#include "oneflow/core/data/data_instance.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/kernel/kernel_util.h"
#include <glog/logging.h>

namespace oneflow {
namespace data {

template<typename T>
struct DataFieldSerializer<ImageDataField, T> {
  static size_t Serialize(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const ImageDataField*>(data_field);
    CHECK_NOTNULL(field);
    const auto& image_mat = field->image_mat();
    if (image_mat.isContinuous()) {
      CopyElem(image_mat.data, buffer, field->ElemCnt());
    } else {
      FOR_RANGE(size_t, i, 0, image_mat.rows) {
        size_t one_row_size = image_mat.cols * image_mat.channels();
        CopyElem(image_mat.ptr<uint8_t>(i), buffer, one_row_size);
        buffer += one_row_size;
      }
    }
    return field->ElemCnt() * sizeof(T);
  }
};

template<typename T>
struct DataFieldSerializer<ArrayDataField<T>, T> {
  static size_t Serialize(const DataField* data_field, T* buffer) {
    const auto* field = dynamic_cast<const ArrayDataField<T>*>(data_field);
    CHECK_NOTNULL(field);
    const auto& vec = field->array();
    CopyElem(vec.data(), buffer, vec.size());
    return vec.size() * sizeof(T);
  }
};

template<typename Pb>
struct DataFieldSerializer<PbListDataField<Pb>, char> {
  static size_t Serialize(const DataField* data_field, char* buffer) {
    const auto* field = dynamic_cast<const PbListDataField<Pb>*>(data_field);
    CHECK_NOTNULL(field);
    const auto& pb_list = field->pb_list();
    size_t pb_msg_max_size = field->pb_max_size();
    size_t total_size = 0;
    for (const PbMessage& pb_msg : pb_list) {
      size_t pb_msg_size = pb_msg.ByteSizeLong();
      CHECK_LE(pb_msg_size, pb_msg_max_size);
      pb_msg.SerializeToArray(buffer, pb_msg_size);
      buffer += pb_msg_max_size;
      total_size += pb_msg_max_size;
    }
    return total_size;
  }
};

template<>
struct DataFieldTransformer<ImageDataField, DataCase::kImage, DataTransformProto::kResize> {
  static void Transform(DataInstance* data_inst, const std::string field_name,
                        const DataTransformProto& proto) {
    CHECK(proto.has_resize());
    auto* data_field = dynamic_cast<ImageDataField*>(data_inst->GetField(field_name));
    auto& img_mat = data_field->image_mat();
    int32_t target_height = proto.resize().height();
    int32_t target_width = proto.resize().width();
    cv::Mat target_mat;
    cv::resize(img_mat, target_mat, cv::Size(target_width, target_height));
    std::vector<float> scale(2);
    scale.at(0) = target_mat.rows * 1.0f / img_mat.rows;
    scale.at(1) = target_mat.cols * 1.0f / img_mat.cols;
    data_inst->AddField("scale", MakeDataField<ArrayDataField<float>>(std::move(scale)));
  }
};

template<>
struct DataFieldTransformer<ImageDataField, DataCase::kImage, DataTransformProto::kTargetResize> {
  static void Transform(DataInstance* data_inst, const std::string field_name,
                        const DataTransformProto& proto) {
    TODO();
  }
};

template<typename T>
struct DataFieldTransformer<ArrayDataField<T>, DataCase::kBoundingBox,
                            DataTransformProto::kResize> {
  static void Transform(DataInstance* data_inst, const std::string field_name,
                        const DataTransformProto& proto) {
    // CHECK(proto.has_resize());
    auto* bbox_data_field = dynamic_cast<ArrayDataField<T>*>(data_inst->GetField(field_name));
    auto* scale_data_field = dynamic_cast<ArrayDataField<float>*>(data_inst->GetField("scale"));
    CHECK_NOTNULL(bbox_data_field);
    CHECK_NOTNULL(scale_data_field);
    auto& bbox_array = bbox_data_field->array();
    auto& scale_array = scale_data_field->array();
    FOR_RANGE(size_t, i, 0, bbox_array.size()) { bbox_array.at(i) *= scale_array.at(i % 2); }
  }
};

template<typename T>
struct DataFieldTransformer<ArrayDataField<T>, DataCase::kBoundingBox,
                            DataTransformProto::kTargetResize> {
  static void Transform(DataInstance* data_inst, const std::string field_name,
                        const DataTransformProto& proto) {
    DataFieldTransformer<ArrayDataField<T>, DataCase::kBoundingBox,
                         DataTransformProto::kResize>::Transform(data_inst, field_name, proto);
  }
};

template<typename Pb>
struct DataFieldTransformer<PbListDataField<Pb>, DataCase::kSegmentation,
                            DataTransformProto::kResize> {
  static void Transform(DataInstance* data_inst, const std::string field_name,
                        const DataTransformProto& proto) {
    auto* data_field = dynamic_cast<PbListDataField<Pb>*>(data_inst->GetField(field_name));
    auto* scale_data_field = dynamic_cast<ArrayDataField<float>*>(data_inst->GetField("scale"));
    CHECK_NOTNULL(data_field);
    CHECK_NOTNULL(scale_data_field);
    auto& scale_array = scale_data_field->array();
    for (PolygonList& polygon_list : data_field->pb_list()) {
      for (FloatList& polygon : *polygon_list.mutable_polygons()) {
        FOR_RANGE(size_t, i, 0, polygon.value_size()) {
          polygon.set_value(i, polygon.value(i) * scale_array.at(i % 2));
        }
      }
    }
  }
};

template<typename Pb>
struct DataFieldTransformer<PbListDataField<Pb>, DataCase::kSegmentation,
                            DataTransformProto::kTargetResize> {
  static void Transform(DataInstance* data_inst, const std::string field_name,
                        const DataTransformProto& proto) {
    DataFieldTransformer<PbListDataField<Pb>, DataCase::kSegmentation,
                         DataTransformProto::kTargetResize>::Transform(data_inst, field_name,
                                                                       proto);
  }
};

size_t DataField::ToBuffer(void* buffer, const DataFieldProto& proto) const {
  return GetDataFieldSerializer(proto)(this, buffer);
}

void ImageDataField::FromFeature(const Feature& feature, const DataFieldProto& proto) {
  CHECK(proto.data_codec().has_image());
  CHECK(feature.has_bytes_list());
  CHECK_EQ(feature.bytes_list().value_size(), 1);
  const std::string& bytes = feature.bytes_list().value(0);
  cv::_InputArray bytes_array(bytes.data(), bytes.size());
  image_mat_ = cv::imdecode(bytes_array, cv::IMREAD_ANYCOLOR);
}

template<typename T>
void ArrayDataField<T>::FromFeature(const Feature& feature, const DataFieldProto& proto) {
  CHECK(proto.data_codec().has_raw());
  if (feature.has_bytes_list()) {
    auto data_type = GetDataType<T>::value;
    CHECK(data_type == DataType::kChar || data_type == DataType::kInt8
          || data_type == DataType::kUInt8);
    CHECK_EQ(feature.bytes_list().value_size(), 1);
    const std::string& bytes = feature.bytes_list().value(0);
    std::transform(bytes.begin(), bytes.end(), array_.begin(),
                   [](unsigned char c) { return static_cast<T>(c); });
  }
#define DEFINE_ELIF(PbT, CppT)                                   \
  else if (feature.has_##PbT##_list()) {                         \
    const auto& values = feature.PbT##_list().value();           \
    std::transform(values.begin(), values.end(), array_.begin(), \
                   [](CppT v) { return static_cast<T>(v); });    \
  }
  DEFINE_ELIF(int32, int32_t)
  DEFINE_ELIF(float, float)
  DEFINE_ELIF(double, double)
#undef DEFINE_ELIF
  else {
    UNIMPLEMENTED();
  }
}

template<typename Pb>
void PbListDataField<Pb>::FromFeature(const Feature& feature, const DataFieldProto& proto) {
  CHECK(proto.data_codec().has_bytes_list());
  CHECK(feature.has_bytes_list());
  pb_max_size_ = proto.data_codec().bytes_list().max_byte_size();
  for (const std::string& bytes : feature.bytes_list().value()) {
    CHECK_LE(bytes.size(), pb_max_size_);
    pb_list_.emplace_back(Pb());
    CHECK(pb_list_.back().ParseFromString(bytes));
  }
}

void DataInstance::Init(const DataInstanceProto& data_inst_proto, const OFRecord& record) {
  for (const auto& pair : record.feature()) {
    const std::string& field_name = pair.first;
    const Feature& feature = pair.second;
    CHECK(data_inst_proto.fields().find(field_name) != data_inst_proto.fields().end());
    const DataFieldProto& data_field_proto = data_inst_proto.fields().at(field_name);
    CHECK(
        fields_.emplace(field_name, CreateDataFieldFromFeature(feature, data_field_proto)).second);
  }
}

bool DataInstance::AddField(const std::string& field_name,
                            std::unique_ptr<DataField>&& data_field_ptr) {
  return fields_.emplace(field_name, std::forward<std::unique_ptr<DataField>>(data_field_ptr))
      .second;
}

bool DataInstance::SetField(const std::string& field_name,
                            std::unique_ptr<DataField>&& data_field_ptr) {
  auto it = fields_.find(field_name);
  if (it == fields_.end()) { return false; }
  it->second.swap(data_field_ptr);
  return true;
}

void DataInstance::Transform(const DataInstanceProto& proto,
                             const DataTransformProto& trans_proto) {
  for (auto& pair : fields_) {
    const std::string& field_name = pair.first;
    const DataFieldProto& field_proto = proto.fields().at(field_name);
    GetDataFieldTransformer(field_proto, trans_proto)(this, field_name, trans_proto);
  }
}

// std::vector<std::string> DataInstance::Fields() const {
//   std::vector<std::string> ret;
//   for (const auto& pair : fields_) { ret.emplace_back(pair.first); }
//   return ret;
// }

std::unique_ptr<DataField> CreateDataFieldFromFeature(const Feature& feature,
                                                      const DataFieldProto& proto) {
#define MAKE_ENTRY(dcodec, dtype, dcase)                                     \
  {GetHashKey(dcodec, dtype, dcase), []() -> DataField* {                    \
     using DataFieldT = typename DataFieldTrait<dcodec, dtype, dcase>::type; \
     return new DataFieldT();                                                \
   }},
  static const HashMap<std::string, std::function<DataField*()>> creators = {
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, DATA_FIELD_TUPLE_SEQ)};
#undef MAKE_ENTRY
  std::unique_ptr<DataField> data_field_ptr(creators.at(
      GetHashKey(proto.data_codec().codec_case(), proto.data_type(), proto.data_case()))());
  data_field_ptr->FromFeature(feature, proto);
  return data_field_ptr;
}

std::function<size_t(const DataField*, void*)> GetDataFieldSerializer(const DataFieldProto& proto) {
#define MAKE_ENTRY(dcodec, dtype, dcase)                                                        \
  {GetHashKey(dcodec, dtype, dcase), [](const DataField* field, void* buffer) {                 \
     using BufferT = DataTypeToType<dtype>;                                                     \
     using DataFieldT = typename DataFieldTrait<dcodec, dtype, dcase>::type;                    \
     return DataFieldSerializer<DataFieldT, BufferT>::Serialize(field,                          \
                                                                static_cast<BufferT*>(buffer)); \
   }},
  static HashMap<std::string, std::function<size_t(const DataField*, void*)>> serializers = {
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, DATA_FIELD_TUPLE_SEQ)};
#undef MAKE_ENTRY
  return serializers.at(
      GetHashKey(proto.data_codec().codec_case(), proto.data_type(), proto.data_case()));
}

std::function<void(DataInstance*, const std::string&, const DataTransformProto&)>
GetDataFieldTransformer(const DataFieldProto& field_proto, const DataTransformProto& trans_proto) {
#define MAKE_ENTRY(dcodec, dtype, dcase, tcase)                                                  \
  {GetHashKey(dcodec, dtype, dcase, tcase),                                                      \
   [](DataInstance* data_inst, const std::string& field_name, const DataTransformProto& proto) { \
     using DataFieldT = typename DataFieldTrait<dcodec, dtype, dcase>::type;                     \
     DataFieldTransformer<DataFieldT, dcase, tcase>::Transform(data_inst, field_name, proto);    \
   }},
  static HashMap<std::string,
                 std::function<void(DataInstance*, const std::string&, const DataTransformProto&)>>
      transformers = {OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, DATA_FIELD_TRANSFORM_TUPLE_SEQ)};
#undef MAKE_ENTRY
  std::string key = GetHashKey(field_proto.data_codec().codec_case(), field_proto.data_type(),
                               field_proto.data_case(), trans_proto.transform_case());
  if (transformers.find(key) != transformers.end()) { return transformers.at(key); }
  return [](DataInstance*, const std::string&, const DataTransformProto&) {};
}

// REGISTER_DATA_FIELD(EncodeConf::kJpeg, ImageDataField);
// REGISTER_TEMPLATE_DATA_FIELD(EncodeConf::kRaw, RawDataField);

}  // namespace data
}  // namespace oneflow
