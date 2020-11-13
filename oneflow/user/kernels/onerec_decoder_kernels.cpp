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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/user/kernels/example_generated.h"

namespace oneflow {

namespace {

int64_t GetBatchSizeInBytes(int64_t batch_size, const Shape& shape, const DataType data_type) {
  return batch_size * shape.elem_cnt() * GetSizeOfDataType(data_type);
}

void GetTensorsFromRecords(const TensorBuffer* records, const int64_t record_num,
                           const std::string& key,
                           std::vector<const onerec::example::Tensor*>* tensors) {
  tensors->resize(record_num);
  for (int32_t i = 0; i < record_num; ++i) {
    const TensorBuffer* cur_record = records + i;
    const auto buffer = reinterpret_cast<const uint8_t*>(cur_record->data());

    const onerec::example::Example* example = onerec::example::GetExample(buffer);
    const auto* features = example->features();
    CHECK_NOTNULL(features);
    const onerec::example::Feature* feature = features->LookupByKey(key.c_str());
    CHECK_NOTNULL(feature);
    const onerec::example::Tensor* tensor = feature->tensor();
    CHECK_NOTNULL(tensor);
    (*tensors)[i] = tensor;
  }
}

void GetTensorDimsWithoutReshape(const std::vector<const onerec::example::Tensor*>& tensors,
                                 const int32_t num_axes,
                                 std::vector<std::vector<int32_t>>* tensor_dims) {
  tensor_dims->resize(num_axes);
  for (int32_t d = 0; d < num_axes; ++d) { (*tensor_dims)[d].resize(tensors.size()); }
  for (int32_t j = 0; j < tensors.size(); ++j) {
    const flatbuffers::Vector<int32_t>* shape_vec = tensors.at(j)->shape();
    CHECK_NOTNULL(shape_vec);
    CHECK_EQ(shape_vec->size(), num_axes);
    for (int32_t d = 0; d < num_axes; ++d) { (*tensor_dims)[d][j] = shape_vec->Get(d); }
  }
}

void GetTensorDimsWithReshape(const std::vector<const onerec::example::Tensor*>& tensors,
                              const Shape& new_shape,
                              std::vector<std::vector<int32_t>>* tensor_dims) {
  tensor_dims->resize(new_shape.NumAxes());
  for (int32_t d = 0; d < new_shape.NumAxes(); ++d) { (*tensor_dims)[d].resize(tensors.size()); }
  bool has_free_dim = false;
  int32_t free_dim_idx = -1;
  int32_t elem_cnt_without_free_dim = 1;
  for (int32_t d = 0; d < new_shape.NumAxes(); ++d) {
    const int32_t dim_size = new_shape.At(d);
    if (dim_size > 0) {
      elem_cnt_without_free_dim *= dim_size;
    } else if (dim_size == -1) {
      CHECK(!has_free_dim);
      has_free_dim = true;
      free_dim_idx = d;
    } else {
      UNIMPLEMENTED();
    }
  }
  for (int32_t j = 0; j < tensors.size(); ++j) {
    const flatbuffers::Vector<int32_t>* shape_vec = tensors.at(j)->shape();
    CHECK_NOTNULL(shape_vec);
    int32_t elem_cnt = 1;
    for (int32_t d = 0; d < shape_vec->size(); ++d) { elem_cnt *= shape_vec->Get(d); }
    if (has_free_dim) {
      CHECK_EQ(elem_cnt % elem_cnt_without_free_dim, 0);
    } else {
      CHECK_EQ(elem_cnt, elem_cnt_without_free_dim);
    }
    for (int32_t d = 0; d < new_shape.NumAxes(); ++d) {
      if (d == free_dim_idx) {
        (*tensor_dims)[d][j] = elem_cnt / elem_cnt_without_free_dim;
      } else {
        (*tensor_dims)[d][j] = new_shape.At(d);
      }
    }
  }
}

template<typename L, bool dim0_padding>
inline void CopyTensorValues(void* dst, const onerec::example::Tensor* tensor, int32_t elem_cnt,
                             int32_t elem_size) {
  const L* list = tensor->data_as<L>();
  CHECK_NOTNULL(list);
  const auto* values = list->values();
  CHECK_NOTNULL(values);
  const int32_t values_size = values->size();
  if (dim0_padding) {
    CHECK_LE(values_size, elem_cnt);
  } else {
    CHECK_EQ(values_size, elem_cnt);
  }
  const auto* src = values->data();
  using elem_type =
      typename std::remove_const<typename std::remove_pointer<decltype(src)>::type>::type;
  CHECK_EQ(sizeof(elem_type), elem_size);
  std::copy(src, src + values_size, reinterpret_cast<elem_type*>(dst));
  if (dim0_padding) {
    std::memset(reinterpret_cast<elem_type*>(dst) + values_size, 0,
                (elem_cnt - values_size) * sizeof(elem_type));
  }
}

template<typename L, bool dim0_padding>
void BatchCopyTensorValues(char* dst_ptr,
                           const std::vector<const onerec::example::Tensor*>& tensors,
                           int32_t elem_cnt, int32_t elem_size) {
  const int32_t size = elem_cnt * elem_size;
  for (int32_t i = 0; i < tensors.size(); ++i) {
    const auto* tensor = tensors.at(i);
    CopyTensorValues<L, dim0_padding>(dst_ptr, tensor, elem_cnt, elem_size);
    dst_ptr += size;
  }
}

template<bool dim0_padding>
void CopyTensorsToBuffer(const std::vector<const onerec::example::Tensor*>& tensors,
                         DataType data_type, const Shape& shape, char* dst_ptr) {
  const int32_t elem_cnt = shape.elem_cnt();
  const int32_t elem_size = GetSizeOfDataType(data_type);
  if (data_type == DataType::kInt8) {
    BatchCopyTensorValues<onerec::example::Int8List, dim0_padding>(dst_ptr, tensors, elem_cnt,
                                                                   elem_size);
  } else if (data_type == DataType::kInt32) {
    BatchCopyTensorValues<onerec::example::Int32List, dim0_padding>(dst_ptr, tensors, elem_cnt,
                                                                    elem_size);
  } else if (data_type == DataType::kInt64) {
    BatchCopyTensorValues<onerec::example::Int64List, dim0_padding>(dst_ptr, tensors, elem_cnt,
                                                                    elem_size);
  } else if (data_type == DataType::kFloat) {
    BatchCopyTensorValues<onerec::example::Float32List, dim0_padding>(dst_ptr, tensors, elem_cnt,
                                                                      elem_size);
  } else if (data_type == DataType::kDouble) {
    BatchCopyTensorValues<onerec::example::Float64List, dim0_padding>(dst_ptr, tensors, elem_cnt,
                                                                      elem_size);
  } else {
    UNIMPLEMENTED();
  }
}

void DecodeField(const TensorBuffer* records, const int64_t record_num, const std::string key,
                 const DataType data_type, const Shape& static_shape, const bool is_dynamic,
                 const bool has_reshape, const Shape& reshape, const bool has_batch_padding,
                 const Shape& batch_padding, user_op::Tensor* out_blob) {
  const int32_t batch_size = record_num;
  char* out_ptr = out_blob->mut_dptr<char>();
  const int64_t out_bytes = out_blob->shape().elem_cnt() * GetSizeOfDataType(data_type);
  std::vector<const onerec::example::Tensor*> tensors;
  GetTensorsFromRecords(records, record_num, key, &tensors);
  std::vector<std::vector<int32_t>> tensor_dims;
  if (has_reshape) {
    CHECK_EQ(reshape.NumAxes(), static_shape.NumAxes());
    GetTensorDimsWithReshape(tensors, reshape, &tensor_dims);
  } else {
    GetTensorDimsWithoutReshape(tensors, static_shape.NumAxes(), &tensor_dims);
  }
  DimVector instance_dim_vec;
  if (has_batch_padding) {
    CHECK_EQ(batch_padding.NumAxes(), static_shape.NumAxes());
    for (int32_t d = 1; d < batch_padding.NumAxes(); ++d) {
      if (batch_padding.At(d) != 0) { UNIMPLEMENTED(); }
    }
    const int32_t dim0_padding_method = batch_padding.At(0);
    int32_t padded_dim0_size = 0;
    if (dim0_padding_method == 0) {
      padded_dim0_size = tensor_dims.at(0).front();
      CHECK(std::all_of(tensor_dims.at(0).cbegin() + 1, tensor_dims.at(0).cend(),
                        [&](const int32_t v) { return v == padded_dim0_size; }));
    } else if (dim0_padding_method == -1) {
      padded_dim0_size = *std::max_element(tensor_dims.at(0).cbegin(), tensor_dims.at(0).cend());
    } else if (dim0_padding_method > 0) {
      padded_dim0_size = dim0_padding_method;
      CHECK(std::all_of(tensor_dims.at(0).cbegin(), tensor_dims.at(0).cend(),
                        [&](const int32_t v) { return v <= padded_dim0_size; }));
    } else {
      UNIMPLEMENTED();
    }
    instance_dim_vec.push_back(padded_dim0_size);
  } else {
    const int32_t dim0_size = tensor_dims.at(0).front();
    CHECK(std::all_of(tensor_dims.at(0).cbegin() + 1, tensor_dims.at(0).cend(),
                      [&](const int32_t v) { return v == dim0_size; }));
    instance_dim_vec.push_back(dim0_size);
  }
  for (int32_t d = 1; d < static_shape.NumAxes(); ++d) {
    const int32_t dim_size = tensor_dims.at(d).front();
    CHECK(std::all_of(tensor_dims.at(d).cbegin() + 1, tensor_dims.at(d).cend(),
                      [&](const int32_t v) { return v == dim_size; }));
    instance_dim_vec.push_back(dim_size);
  }
  const Shape instance_shape = Shape(instance_dim_vec);
  if (is_dynamic) {
    CHECK_LE(instance_shape.elem_cnt(), static_shape.elem_cnt());
    out_blob->mut_shape()->Set(0, record_num);
    for (int64_t d = 0; d < instance_shape.NumAxes(); ++d) {
      out_blob->mut_shape()->Set(d + 1, instance_shape.At(d));
    }
  } else {
    CHECK(instance_shape == static_shape);
    CHECK_EQ(out_blob->shape().At(0), record_num);
    for (int64_t d = 0; d < instance_shape.NumAxes(); ++d) {
      CHECK_EQ(out_blob->shape().At(d + 1), instance_shape.At(d));
    }
  }
  const int64_t buffer_size = GetBatchSizeInBytes(batch_size, instance_shape, data_type);
  CHECK_LE(buffer_size, out_bytes);
  if (has_batch_padding) {
    CopyTensorsToBuffer<true>(tensors, data_type, instance_shape, out_ptr);
  } else {
    CopyTensorsToBuffer<false>(tensors, data_type, instance_shape, out_ptr);
  }
}

}  // namespace

template<typename T>
class OneRecDecoderKernel final : public user_op::OpKernel {
 public:
  OneRecDecoderKernel() = default;
  ~OneRecDecoderKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t record_num = in_blob->shape().At(0);
    CHECK(record_num > 0);
    const TensorBuffer* records = in_blob->dptr<TensorBuffer>();

    const std::string key = ctx->Attr<std::string>("key");
    const DataType data_type = ctx->Attr<DataType>("data_type");
    const Shape& static_shape = ctx->Attr<Shape>("static_shape");
    const bool is_dynamic = ctx->Attr<bool>("is_dynamic");
    const bool has_reshape = ctx->Attr<bool>("has_reshape");
    const Shape& reshape = ctx->Attr<Shape>("reshape");
    const bool has_batch_padding = ctx->Attr<bool>("has_batch_padding");
    const Shape& batch_padding = ctx->Attr<Shape>("batch_padding");

    DecodeField(records, record_num, key, data_type, static_shape, is_dynamic, has_reshape, reshape,
                has_batch_padding, batch_padding, out_blob);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ONEREC_DECODER_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("onerec_decoder")                                            \
      .SetCreateFn<OneRecDecoderKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                           \
                       & (user_op::HobDataType("in", 0) == DataType::kTensorBuffer) \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_ONEREC_DECODER_KERNEL(char)
REGISTER_ONEREC_DECODER_KERNEL(float)
REGISTER_ONEREC_DECODER_KERNEL(double)
REGISTER_ONEREC_DECODER_KERNEL(int8_t)
REGISTER_ONEREC_DECODER_KERNEL(int32_t)
REGISTER_ONEREC_DECODER_KERNEL(int64_t)
REGISTER_ONEREC_DECODER_KERNEL(uint8_t)

}  // namespace oneflow
