#ifndef ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_
#define ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/memory_allocator.h"

namespace oneflow {

inline void CheckTensorBufferDataType(DataType val) {
  CHECK(val != DataType::kTensorBuffer && val != DataType::kOFRecord)
      << "TensorBuffer only support POD as internal data type.";
}

class TensorBuffer {
 public:
  OF_DISALLOW_COPY_AND_MOVE(TensorBuffer);
  TensorBuffer()
      : shape_(Shape()), data_(nullptr), data_type_(DataType::kInvalidDataType), num_bytes_(0) {}
  virtual ~TensorBuffer() = default;

  const Shape& shape() const { return shape_; }

  DataType data_type() const { return data_type_; }

  void set_data_type(DataType val) {
    CheckTensorBufferDataType(val);
    if (data_type_ == val) { return; }
    if (val == DataType::kInvalidDataType) {
      data_type_ = val;
      return;
    } else {
      Resize(shape_, val);
    }
  }

  template<typename T = void>
  inline T* mut_data() {
    if (data_ == nullptr) { return nullptr; }
    CheckDataType<T>(data_type_);
    return static_cast<T*>(data_.get());
  }

  template<typename T = void>
  inline const T* data() const {
    if (data_ == nullptr) { return nullptr; }
    CheckDataType<T>(data_type_);
    return static_cast<const T*>(data_.get());
  }

  void reset() {
    shape_ = Shape();
    data_.reset();
    data_type_ = DataType::kInvalidDataType;
    num_bytes_ = 0;
  }

  void reserve(size_t new_num_bytes) {
    if (new_num_bytes <= num_bytes_) { return; }
    data_.reset();
    data_.reset(MemoryAllocatorImpl::AllocateUnPinnedHostMem(new_num_bytes),
                MemoryAllocatorImpl::DeallocateUnPinnedHostMem);
    num_bytes_ = new_num_bytes;
  }

  int64_t elem_cnt() const { return shape_.elem_cnt(); }

  size_t nbytes() const { return elem_cnt() * GetSizeOfDataType(data_type_); }

  size_t capacity() const { return num_bytes_; }

  void Resize(const Shape& new_shape) { Resize(new_shape, data_type_); }

  void Resize(const Shape& new_shape, DataType new_type) {
    int64_t elem_cnt = new_shape.elem_cnt();
    if (new_type == DataType::kInvalidDataType || elem_cnt == 0) { return; }
    CheckTensorBufferDataType(new_type);

    data_type_ = new_type;
    shape_ = new_shape;

    size_t new_num_bytes = elem_cnt * GetSizeOfDataType(new_type);
    new_num_bytes = RoundUp(new_num_bytes, kTensorBufferAlignedSize);
    if (new_num_bytes > num_bytes_) {
      new_num_bytes =
          std::max(new_num_bytes, RoundUp(num_bytes_ * growth_factor_, kTensorBufferAlignedSize));
      reserve(new_num_bytes);
    } else if (new_num_bytes < num_bytes_ * shrink_threshold_) {
      data_.reset();
      num_bytes_ = 0;
      reserve(new_num_bytes);
    }
  }

  void CopyFrom(const TensorBuffer& src) {
    if (&src == this) { return; }
    Resize(src.shape(), src.data_type());
    memcpy(mut_data(), src.data(), nbytes());
  }

 private:
  // TODO(chengcheng)
  static double growth_factor_;
  static double shrink_threshold_;
  static constexpr size_t kTensorBufferAlignedSize = 1024;

  Shape shape_;
  std::shared_ptr<void> data_;
  DataType data_type_;
  size_t num_bytes_;
};

#define BUFFER_DATA_TYPE_SEQ OF_PP_MAKE_TUPLE_SEQ(TensorBuffer, DataType::kTensorBuffer)

template<>
struct GetDataType<TensorBuffer> : std::integral_constant<DataType, DataType::kTensorBuffer> {};
inline TensorBuffer GetTypeByDataType(std::integral_constant<DataType, DataType::kTensorBuffer>) {
  return {};
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_
