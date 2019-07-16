#ifndef ONEFLOW_CORE_DATASET_DATA_INSTANCE_H_
#define ONEFLOW_CORE_DATASET_DATA_INSTANCE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/record/record.pb.h"
#include "oneflow/core/data/data.pb.h"
#include "oneflow/core/record/coco.pb.h"
#include <opencv2/opencv.hpp>

namespace oneflow {
namespace data {

struct DataField {
  OF_DISALLOW_COPY_AND_MOVE(DataField);
  DataField() = default;
  virtual ~DataField() = default;
  size_t ToBuffer(void* buffer, const DataFieldProto& proto) const;
  virtual void FromFeature(const Feature& feature, const DataFieldProto& proto) = 0;
  virtual size_t ElemCnt() const = 0;
};

template<typename T>
class ArrayDataField : public DataField {
 public:
  ArrayDataField() = default;
  explicit ArrayDataField(std::vector<T>&& vec) : array_(vec) {}
  void FromFeature(const Feature& feature, const DataFieldProto& proto) override;
  size_t ElemCnt() const override { return array_.size(); }
  const std::vector<T>& array() const { return array_; }
  std::vector<T>& array() { return array_; }

 private:
  std::vector<T> array_;
};

class ImageDataField : public DataField {
 public:
  void FromFeature(const Feature& feature, const DataFieldProto& proto) override;
  size_t ElemCnt() const override { return image_mat_.total() * image_mat_.channels(); }
  const cv::Mat& image_mat() const { return image_mat_; }
  cv::Mat& image_mat() { return image_mat_; }

 private:
  cv::Mat image_mat_;
};

template<typename Pb>
class PbListDataField : public DataField {
 public:
  void FromFeature(const Feature& feature, const DataFieldProto& proto) override;
  size_t ElemCnt() const override { return pb_list_.size(); }
  const std::vector<Pb>& pb_list() const { return pb_list_; }
  std::vector<Pb>& pb_list() { return pb_list_; }
  const std::vector<size_t>& pb_sizes() const { return pb_sizes_; }
  size_t pb_max_size() const { return pb_max_size_; }

 private:
  std::vector<Pb> pb_list_;
  std::vector<size_t> pb_sizes_;
  size_t pb_max_size_;
};

class DataInstance {
 public:
  DataInstance() = default;
  void Init(const DataInstanceProto& proto, const OFRecord& record);
  bool HasField(const std::string& field_name) const;
  const DataField* GetField(const std::string& field_name) const;
  DataField* GetField(const std::string& field_name);
  bool AddField(const std::string& field_name, std::unique_ptr<DataField>&& data_field_ptr);
  bool SetField(const std::string& field_name, std::unique_ptr<DataField>&& data_field_ptr);
  void Transform(const DataInstanceProto& proto, const DataTransformProto& trans_proto);

 private:
  HashMap<std::string, std::unique_ptr<DataField>> fields_;
};

inline bool DataInstance::HasField(const std::string& field_name) const {
  return fields_.find(field_name) != fields_.end();
}

inline const DataField* DataInstance::GetField(const std::string& field_name) const {
  return fields_.at(field_name).get();
}

inline DataField* DataInstance::GetField(const std::string& field_name) {
  return fields_.at(field_name).get();
}

template<DataCodec::CodecCase codec, DataType dtype, DataCase dcase>
struct DataFieldTrait;

template<DataType dtype, DataCase dcase>
struct DataFieldTrait<DataCodec::kImage, dtype, dcase> {
  using type = ImageDataField;
};

template<DataType dtype, DataCase dcase>
struct DataFieldTrait<DataCodec::kRaw, dtype, dcase> {
  using type = ArrayDataField<DataTypeToType<dtype>>;
};

template<DataType dtype>
struct DataFieldTrait<DataCodec::kBytesList, dtype, DataCase::kSegmentation> {
  using type = PbListDataField<PolygonList>;
};

template<typename DataFieldT>
struct IsArrayDataField : std::false_type {};

template<typename T>
struct IsArrayDataField<ArrayDataField<T>> : std::true_type {};

template<typename D, typename T>
struct DataFieldSerializer {
  static size_t Serialize(const DataField* data_field, T* buffer);
};

template<typename DataFieldT, DataCase DCase, DataTransformProto::TransformCase TCase>
struct DataFieldTransformer {
  static void Transform(DataInstance* data_inst, const std::string field_name,
                        const DataTransformProto& proto);
};

std::unique_ptr<DataField> CreateDataFieldFromFeature(const Feature& feature,
                                                      const DataFieldProto& proto);
std::function<size_t(const DataField*, void*)> GetDataFieldSerializer(const DataFieldProto& proto);
std::function<void(DataInstance*, const std::string&, const DataTransformProto&)>
GetDataFieldTransformer(const DataFieldProto& field_proto, const DataTransformProto& trans_proto);

template<typename T, typename... Args>
std::unique_ptr<DataField> MakeDataField(Args&&... args) {
  std::unique_ptr<DataField> ret = std::make_unique<T>(std::forward<Args>(args)...);
  return ret;
}

}  // namespace data
}  // namespace oneflow

#define DATA_CODEC_SEQ                    \
  OF_PP_MAKE_TUPLE_SEQ(DataCodec::kImage) \
  OF_PP_MAKE_TUPLE_SEQ(DataCodec::kRaw)   \
  OF_PP_MAKE_TUPLE_SEQ(DataCodec::kBytesList)

#define DATA_CASE_SEQ                           \
  OF_PP_MAKE_TUPLE_SEQ(DataCase::kImage)        \
  OF_PP_MAKE_TUPLE_SEQ(DataCase::kLabel)        \
  OF_PP_MAKE_TUPLE_SEQ(DataCase::kBoundingBox)  \
  OF_PP_MAKE_TUPLE_SEQ(DataCase::kSegmentation) \
  OF_PP_MAKE_TUPLE_SEQ(DataCase::kObjectLabel)

#define DATA_TRANSFORM_CASE_SEQ                     \
  OF_PP_MAKE_TUPLE_SEQ(DataTransformProto::kResize) \
  OF_PP_MAKE_TUPLE_SEQ(DataTransformProto::kTargetResize)

#define EXTRACT_DATA_TYPE(type, type_val) OF_PP_MAKE_TUPLE_SEQ(type_val)

#define DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ \
  OF_PP_FOR_EACH_TUPLE(EXTRACT_DATA_TYPE, ARITHMETIC_DATA_TYPE_SEQ)

#define IMAGE_DATA_FIELD_TUPLE_SEQ                                            \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, (DataCodec::kImage), \
                                   DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ, (DataCase::kImage))

#define DATA_FIELD_POD_DATA_TYPE_SEQ OF_PP_FOR_EACH_TUPLE(EXTRACT_DATA_TYPE, POD_DATA_TYPE_SEQ)

#define ARRAY_DATA_FIELD_TUPLE_SEQ                                          \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, (DataCodec::kRaw), \
                                   DATA_FIELD_POD_DATA_TYPE_SEQ, DATA_CASE_SEQ)

#define PB_LIST_DATA_FIELD_TUPLE_SEQ \
  OF_PP_MAKE_TUPLE_SEQ(DataCodec::kBytesList, DataType::kChar, DataCase::kSegmentation)

#define DATA_FIELD_TUPLE_SEQ \
  IMAGE_DATA_FIELD_TUPLE_SEQ \
  ARRAY_DATA_FIELD_TUPLE_SEQ \
  PB_LIST_DATA_FIELD_TUPLE_SEQ

#define IMAGE_DATA_FIELD_TRANSFORM_TUPLE_SEQ                                                \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, (DataCodec::kImage),               \
                                   DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ, (DataCase::kImage), \
                                   DATA_TRANSFORM_CASE_SEQ)

/*
#define LABEL_DATA_FIELD_TRANSFORM_TUPLE_SEQ \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, (DataCodec::kRaw), \
                                   DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ, \
                                   (DataCase::kLabel DataCase::kObjectLabel), \
                                   DATA_TRANSFORM_CASE_SEQ)
*/

#define BBOX_DATA_FIELD_TRANSFORM_TUPLE_SEQ                                                       \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, (DataCodec::kRaw),                       \
                                   DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ, (DataCase::kBoundingBox), \
                                   DATA_TRANSFORM_CASE_SEQ)

#define SEGMENTATION_DATA_FIELD_TRANSFORM_TUPLE_SEQ                               \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, (DataCodec::kBytesList), \
                                   (DataType::kChar), (DataCase::kSegmentation),  \
                                   DATA_TRANSFORM_CASE_SEQ)

#define DATA_FIELD_TRANSFORM_TUPLE_SEQ \
  IMAGE_DATA_FIELD_TRANSFORM_TUPLE_SEQ \
  BBOX_DATA_FIELD_TRANSFORM_TUPLE_SEQ  \
  SEGMENTATION_DATA_FIELD_TRANSFORM_TUPLE_SEQ

/*
#define REGISTER_DATA_FIELD(k, derived_data_field_class) \
  REGISTER_CLASS_WITH_ARGS(k, DataField, derived_data_field_class, const DataFieldProto&)

#define REGISTER_DATA_FIELD_CREATOR(k, creator) \
  REGISTER_CLASS_CREATOR(k, DataField, creator, const DataFieldProto&);

#define MAKE_GENERIC_DATA_FIELD_CREATOR_ENTRY(dcodec, dtype, dcase)          \
  {GetHashKey(dcodec, dtype, dcase), [](const DataFieldProto& proto) {       \
     using DataFieldT = typename DataFieldTrait<dcodec, dtype, dcase>::type; \
     return new DataFieldT(proto);                                           \
   }},

#define REGISTER_GENERIC_DATA_FIELD_CREATOR(k, data_field_class, type_pair_seq)               \
  namespace {                                                                                 \
  DataField* OF_PP_CAT(CreateDataField, __LINE__)(const DataFieldProto& proto) {              \
    static const HashMap<std::string, std::function<DataField*(const DataFieldProto& proto)>> \
        creators = {OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_GENERIC_DATA_FIELD_CREATOR_ENTRY,   \
                                                     (data_field_class), type_pair_seq)};     \
    return creators.at(proto.data_codec().codec_case(), proto.data_type(),                    \
                       proto.data_case())(proto);                                             \
  }                                                                                           \
  REGISTER_DATA_FIELD_CREATOR(k, OF_PP_CAT(CreateDataField, __LINE__));                       \
  }

#define EXPLICIT_INSTANTIATE_TEMPLATE_DATA_FIELD(data_field_class, t) \
  template class data_field_class<t>;

#define REGISTER_TEMPLATE_DATA_FIELD(k, data_field_class)                                        \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(EXPLICIT_INSTANTIATE_TEMPLATE_DATA_FIELD, (data_field_class), \
                                   ARITHMETIC_DATA_TYPE_SEQ);                                    \
  REGISTER_GENERIC_DATA_FIELD_CREATOR(k, data_field_class, ARITHMETIC_DATA_TYPE_SEQ);
*/

#endif  // ONEFLOW_CORE_DATASET_DATA_INSTANCE_H_
