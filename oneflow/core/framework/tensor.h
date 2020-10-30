/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/memory/memory_case.pb.h"
namespace oneflow {

class Blob;
class BlobAccessChecker;

namespace user_op {

/**
 * @brief A Tensor object contains the data needed for caculation and maintains the pointer
 * to the buffer inside it. We can use the method shape to get tenosr's shape, use the 
 * dptr or mut_dptr to get the pointer to the buffer. 
 *
 */
class Tensor final {
 public:
  Tensor(Blob*);
  ~Tensor() = default;

  Tensor(const Tensor& rhs) { this->CopyWithoutData(rhs); }
  Tensor(Tensor&& rhs) { *this = std::move(rhs); }
  void CopyWithoutData(const Tensor& rhs);
  Tensor& operator=(Tensor&& rhs);

  /**
   * @brief Get the shape of current tensor.
   * 
   * @return const ShapeView& 
   */
  const ShapeView& shape() const { return shape_; }
  
  /**
   * @brief Get the mutable shape of current tensor.
   * 
   * @return MutShapeView* 
   */
  MutShapeView* mut_shape() {
    this->header_access_check();
    return mut_shape_.get();
  }

  DataType data_type() const { return data_type_; }

  /**
   * @brief Return the memory case (host or device memory)
   * 
   * @return const MemoryCase& 
   */
  const MemoryCase& mem_case() const { return *mem_case_; }

  /**
   * @brief Return the pointer to the data buffer.
   * 
   * @tparam T What type the pointer should converted to
   * @return const T* 
   */
  template<typename T = void>
  const T* dptr() const {
    CheckDataType<T>();
    return static_cast<const T*>(dptr_);
  }

  /**
   * @brief Similar to dptr(), except that it returns a pointer to mutable buffer.
   * 
   * @tparam T 
   * @return T* 
   */
  template<typename T = void>
  T* mut_dptr() {
    this->body_access_check();
    CheckDataType<T>();
    return static_cast<T*>(dptr_);
  }

 private:
  template<typename T>
  void CheckDataType() const {
    LOG_IF(FATAL, (std::is_same<T, void>::value == false && std::is_same<T, char>::value == false
                   && data_type_ != DataType::kChar && data_type_ != GetDataType<T>::value))
        << "tensor data_type mismatched. value: " << DataType_Name(data_type_)
        << ", template T:" << DataType_Name(GetDataType<T>::value);
  }

  void header_access_check();
  void body_access_check();

  void* dptr_;
  ShapeView shape_;
  std::unique_ptr<MutShapeView> mut_shape_;
  DataType data_type_;
  const MemoryCase* mem_case_;
  const BlobAccessChecker* blob_access_checker_;
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_H_
