#ifndef ONEFLOW_CORE_DATA_DATA_FIELD_H_
#define ONEFLOW_CORE_DATA_DATA_FIELD_H_

#include "oneflow/core/data/data.pb.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/shape.h"
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
  virtual void InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                          Shape* shape) const = 0;

 private:
  DataSourceCase data_source_;
};

template<typename T>
class TensorDataField : public DataField {
 public:
  typedef T data_type;

  const std::vector<T>& data() const { return data_; }
  std::vector<T>& data() { return data_; }
  void InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                  Shape* shape) const override;

 private:
  std::vector<T> data_;
};

class ImageDataField : public DataField {
 public:
  const cv::Mat& data() const { return data_; }
  cv::Mat& data() { return data_; }
  void InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                  Shape* shape) const override;

 private:
  cv::Mat data_;
};

template<typename T>
class LoDDataField : public DataField {
 public:
  typedef T data_type;

  LoDDataField(size_t max_length)
      : data_(new T[max_length]), capacity_(max_length), total_length_(0) {
    std::memset(data_.get(), 0, capacity_ * sizeof(T));
  }

  void PushBack(T val) {
    CHECK_LT(total_length_, capacity_);
    data_.get()[total_length_] = val;
    total_length_ += 1;
  }

  void IncreaseDataLength(size_t len) {
    total_length_ += len;
    CHECK_LT(total_length_, capacity_);
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
  void InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                  Shape* shape) const override;

  T* data() { return data_.get(); }
  const T* data() const { return data_.get(); }
  size_t total_length() const { return total_length_; }

 private:
  std::unique_ptr<T[]> data_;
  size_t capacity_;
  size_t total_length_;
  std::vector<std::vector<size_t>> lod_len_;
};

template<typename T>
class TensorListDataField : public DataField {
 public:
  typedef T data_type;

  TensorListDataField(size_t max_size) : data_(new T[max_size]), capacity_(max_size), size_(0) {
    std::memset(data_.get(), 0, capacity_ * sizeof(T));
  }

  void IncreaseSize(size_t size) {
    size_ += size;
    CHECK_LT(size_, capacity_);
  }
  void PushBack(T val) {
    data()[size()] = val;
    IncreaseSize(1);
  }
  template<typename... I>
  auto SetShape(I... dims)
      -> std::enable_if_t<std::conjunction<std::is_convertible<int, I>...>::value> {
    tensor_shape_ = {dims...};
  }
  size_t ArrayLength() const {
    int64_t tensor_elem_cnt =
        std::accumulate(tensor_shape_.begin(), tensor_shape_.end(), 1, std::multiplies<int64_t>());
    CHECK_EQ(size_ % tensor_elem_cnt, 0);
    return size_ / static_cast<size_t>(tensor_elem_cnt);
  }
  void InferShape(const ShapeProto& shape_proto, const OptInt64& var_axis,
                  Shape* shape) const override;

  T* data() { return data_.get(); }
  const T* data() const { return data_.get(); }
  DimVector& shape() { return tensor_shape_; }
  const DimVector& shape() const { return tensor_shape_; }
  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

 private:
  std::unique_ptr<T[]> data_;
  size_t capacity_;
  size_t size_;
  DimVector tensor_shape_;
};

template<DataSourceCase dsrc>
struct DataFieldTrait;

template<>
struct DataFieldTrait<DataSourceCase::kImage> {
  typedef ImageDataField type;
};

template<>
struct DataFieldTrait<DataSourceCase::kLabel> {
  typedef TensorDataField<int32_t> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kImageId> {
  typedef TensorDataField<int64_t> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kImageSize> {
  typedef TensorDataField<int32_t> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kImageScale> {
  typedef TensorDataField<float> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kObjectSegmentation> {
  typedef LoDDataField<double> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kObjectLabel> {
  typedef TensorListDataField<int32_t> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kObjectBoundingBox> {
  typedef TensorListDataField<float> type;
};

template<>
struct DataFieldTrait<DataSourceCase::kObjectSegmentationAlignedMask> {
  typedef TensorListDataField<int8_t> type;
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

#define DATA_SOURCE_SEQ                                                \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kImage)                         \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kLabel)                         \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kObjectBoundingBox)             \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kObjectSegmentation)            \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kObjectSegmentationAlignedMask) \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kObjectLabel)                   \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kImageScale)                    \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kImageSize)                     \
  OF_PP_MAKE_TUPLE_SEQ(DataSourceCase::kImageId)

#define EXTRACT_DATA_TYPE(type, type_val) OF_PP_MAKE_TUPLE_SEQ(type_val)

#define DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ \
  OF_PP_FOR_EACH_TUPLE(EXTRACT_DATA_TYPE, ARITHMETIC_DATA_TYPE_SEQ)

#define DATA_FIELD_SERIALIZER_TUPLE_SEQ                                   \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_MAKE_TUPLE_SEQ, DATA_SOURCE_SEQ, \
                                   DATA_FIELD_ARITHMETIC_DATA_TYPE_SEQ)

#endif  // ONEFLOW_CORE_DATA_DATA_FIELD_H_
