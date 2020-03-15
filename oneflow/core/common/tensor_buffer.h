#ifndef ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_
#define ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/memory/dynamic_memory_allocator.h"

namespace oneflow {

class TensorBuffer {
 public:
  TensorBuffer()
      : shape_(Shape()), data_(nullptr), data_type_(DataType::kInvalidDataType), num_bytes_(0) {
    mem_case_.mutable_host_mem();
  };
  virtual ~TensorBuffer() = default;

  const Shape& shape() const { return shape_; }

  DataType data_type() const { return data_type_; }

  void set_data_type(DataType val) {
    CHECK(val != DataType::kTensorBuffer) << "TensorBuffer cannot store datatype as itself";
    if (data_type_ == val) { return; }
    data_type_ = val;
    if (val == DataType::kInvalidDataType) { return; }
    size_t new_num_bytes = elem_cnt() * GetSizeOfDataType(val);
    if (new_num_bytes > num_bytes_) { reserve(new_num_bytes); }
  }

  template<typename T = void>
  inline T* mut_data() {
    if (data_ == nullptr) { return nullptr; }
    CheckDataType<T>(data_type_);
    return static_cast<T*>(data_.get());
  }

  template<typename T = void>
  inline const T* data() {
    if (data_ == nullptr) { return nullptr; }
    CheckDataType<T>(data_type_);
    return static_cast<const T*>(data_.get());
  }

  void reset() {
    shape_ = Shape();
    data_.reset();
    data_type_ = DataType::kInvalidDataType;
    num_bytes_ = 0;
    mem_case_.mutable_host_mem();
  }

  void reserve(size_t new_num_bytes) {
    if (new_num_bytes <= num_bytes_) { return; }

    data_.reset();
    data_.reset(DynamicMemoryAllocator::New(mem_case_, new_num_bytes),
                std::bind(DynamicMemoryAllocator::Delete, std::placeholders::_1, mem_case_));
    num_bytes_ = new_num_bytes;
  }

  /**
   * @brief Returns the size in elements of the underlying data
   */
  int64_t elem_cnt() const { return shape_.elem_cnt(); }

  /**
   * @brief Returns the size in bytes of the underlying data
   */
  size_t nbytes() const { return elem_cnt() * GetSizeOfDataType(data_type_); }

  /**
   * @brief Returns the real size of the allocation
   */
  size_t capacity() const { return num_bytes_; }

  const MemoryCase& mem_case() const { return mem_case_; }

  void set_mem_case(const MemoryCase& new_mem_case) {
    if (mem_case_ == new_mem_case) { return; }
    // set new mem case will reset() original data
    DataType original_data_type = data_type_;
    reset();
    mem_case_ = new_mem_case;
    data_type_ = original_data_type;
  }

  void Resize(const Shape& new_shape) { Resize(new_shape, data_type_); }

  void Resize(const Shape& new_shape, DataType new_type) {
    int64_t elem_cnt = new_shape.elem_cnt();
    if (new_type == DataType::kInvalidDataType || elem_cnt == 0) { return; }

    data_type_ = new_type;
    shape_ = new_shape;

    size_t new_num_bytes = elem_cnt * GetSizeOfDataType(new_type);
    if (new_num_bytes > num_bytes_) {
      new_num_bytes =
          std::max(new_num_bytes, RoundUp(num_bytes_ * growth_factor_, kTensorBufferAlignedSize));
      reserve(new_num_bytes);
    } else if (!MemoryCaseUtil::IsPinnedMemoryCase(mem_case_)
               && RoundUp(new_num_bytes, kTensorBufferAlignedSize)
                      < num_bytes_ * shrink_threshold_) {
      data_.reset();
      num_bytes_ = 0;
      reserve(RoundUp(new_num_bytes, kTensorBufferAlignedSize));
    }
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
  MemoryCase mem_case_;
};

double TensorBuffer::growth_factor_ = 1.0;
double TensorBuffer::shrink_threshold_ = 0.9;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_TENSOR_BUFFER_H_
