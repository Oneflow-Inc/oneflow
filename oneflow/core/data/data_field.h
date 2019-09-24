#ifndef ONEFLOW_CORE_DATA_DATA_FIELD_H_
#define ONEFLOW_CORE_DATA_DATA_FIELD_H_

#include "oneflow/core/data/data.pb.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include <opencv2/opencv.hpp>

namespace oneflow {
namespace data {

class DataInstance;

class DataField {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataField);
  DataField() = default;
  virtual ~DataField() = default;
  size_t ToBuffer(void* buffer, DataType data_type) const;
  void SetSource(DataSourceCase dsrc) { data_source_ = dsrc; }
  DataSourceCase Source() const { return data_source_; }

 private:
  DataSourceCase data_source_;
};

template<typename T>
class ArrayDataField : public DataField {
 public:
  typedef T data_type;

  const std::vector<T>& data() const { return data_; }
  std::vector<T>& data() { return data_; }

 private:
  std::vector<T> data_;
};

class ImageDataField : public DataField {
 public:
  const cv::Mat& data() const { return data_; }
  cv::Mat& data() { return data_; }

 private:
  cv::Mat data_;
};

template<typename T>
class NdarrayDataField : public DataField {
 public:
  typedef T data_type;

  NdarrayDataField(size_t max_length) : data_(new T[max_length]), total_length_(0) {}

  void PushBack(T val) {
    CHECK_LT(total_length_, sizeof(*data_.get()) / sizeof(T));
    data_.get()[total_length_] = val;
    total_length_ += 1;
  }

  void AppendLodLength(size_t lvl, size_t len) {
    if (lod_len_.size() <= lvl) { lod_len_.resize(lvl + 1); }
    lod_len_.at(lvl).push_back(len);
  }

  void IncreaseLodLength(size_t lvl, size_t len) {
    if (lod_len_.size() <= lvl) { lod_len_.resize(lvl + 1); }
    if (lod_len_.at(lvl).empty()) { lod_len_.at(lvl).push_back(0); }
    lod_len_.at(lvl).back() += len;
  }

  int Levels() const { return lod_len_.size(); }
  std::vector<size_t>& GetLod(int lvl) { return lod_len_.at(lvl); }
  const std::vector<size_t>& GetLod(int lvl) const { return lod_len_.at(lvl); }

  T* data() { return data_.get(); }
  const T* data() const { return data_.get(); }
  size_t total_length() const { return total_length_; }

 private:
  std::unique_ptr<T[]> data_;
  size_t total_length_;
  std::vector<std::vector<size_t>> lod_len_;
};

template<DataSourceCase dsrc>
struct DataFieldTrait;

template<>
struct DataFieldTrait<DataSourceCase::kImage> {
  typedef ImageDataField type;
};

template<>
struct DataFieldTrait<DataSourceCase::kLabel> {
  typedef ArrayDataField<int32_t> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kObjectBoundingBox> {
  typedef ArrayDataField<float> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kObjectSegmentation> {
  typedef NdarrayDataField<float> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kObjectLabel> {
  typedef ArrayDataField<int32_t> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kImageScale> {
  typedef ArrayDataField<float> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kImageSize> {
  typedef ArrayDataField<int32_t> type;
};

template<typename DataFieldT, typename T>
struct DataFieldSerializer;

std::function<size_t(const DataField*, void*)> GetDataFieldSerializer(DataSourceCase dsrc,
                                                                      DataType dtype);
std::unique_ptr<DataField> CreateDataFieldFromProto(const DataFieldProto& proto);

template<typename T, typename... Args>
std::unique_ptr<DataField> MakeDataField(Args&&... args) {
  std::unique_ptr<DataField> ret = std::make_unique<T>(std::forward<Args>(args)...);
  return ret;
}

}  // namespace data
}  // namespace oneflow

#define DATA_SOURCE_SEQ                                     \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kImage)              \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kLabel)              \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kObjectBoundingBox)  \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kObjectSegmentation) \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kObjectLabel)        \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kImageScale)         \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kImageSize)

#define EXTRACT_DATA_TYPE(type, type_val) OF_PP_MAKE_TUPLE_SEQ(type_val)

#define DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ \
  OF_PP_FOR_EACH_TUPLE(EXTRACT_DATA_TYPE, ARITHMETIC_DATA_TYPE_SEQ)

#define DATA_FIELD_SERIALIZER_TUPLE_SEQ                                   \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, DATA_SOURCE_SEQ, \
                                   DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ)

#endif  // ONEFLOW_CORE_DATA_DATA_FIELD_H_
