#include "oneflow/core/record/onerec_decoder.h"
#include "oneflow/core/record/onerec_reader.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace {

int64_t GetBatchSizeInBytes(int64_t batch_size, const Shape& shape, const DataType data_type) {
  return batch_size * shape.elem_cnt() * GetSizeOfDataType(data_type);
}

void GetTensorsFromRecords(const std::vector<OneRecExampleWrapper>& records, const std::string& key,
                           std::vector<const onerec::example::Tensor*>* tensors) {
  tensors->resize(records.size());
  for (int32_t i = 0; i < records.size(); ++i) {
    const onerec::example::Example* example = records.at(i).GetExample();
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
                              const ShapeProto& new_shape,
                              std::vector<std::vector<int32_t>>* tensor_dims) {
  tensor_dims->resize(new_shape.dim_size());
  for (int32_t d = 0; d < new_shape.dim_size(); ++d) { (*tensor_dims)[d].resize(tensors.size()); }
  bool has_free_dim = false;
  int32_t free_dim_idx = -1;
  int32_t elem_cnt_without_free_dim = 1;
  for (int32_t d = 0; d < new_shape.dim_size(); ++d) {
    const int32_t dim_size = new_shape.dim(d);
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
    for (int32_t d = 0; d < new_shape.dim_size(); ++d) {
      if (d == free_dim_idx) {
        (*tensor_dims)[d][j] = elem_cnt / elem_cnt_without_free_dim;
      } else {
        (*tensor_dims)[d][j] = new_shape.dim(d);
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
void BatchCopyTensorValues(void* dst, const std::vector<const onerec::example::Tensor*>& tensors,
                           int32_t elem_cnt, int32_t elem_size) {
  char* dst_ptr = reinterpret_cast<char*>(dst);
  const int32_t size = elem_cnt * elem_size;
  for (int32_t i = 0; i < tensors.size(); ++i) {
    const auto* tensor = tensors.at(i);
    CopyTensorValues<L, dim0_padding>(dst_ptr, tensor, elem_cnt, elem_size);
    dst_ptr += size;
  }
}

template<bool dim0_padding>
void CopyTensorsToBuffer(const std::vector<const onerec::example::Tensor*>& tensors,
                         DataType data_type, const Shape& shape, std::vector<char>* buf) {
  const int32_t elem_cnt = shape.elem_cnt();
  const int32_t elem_size = GetSizeOfDataType(data_type);
  char* dst_ptr = buf->data();
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

void DecodeField(const std::vector<OneRecExampleWrapper>& records,
                 const DecodeOneRecFieldConf& field, Shape* instance_shape,
                 std::vector<char>* buf) {
  const int32_t batch_size = records.size();
  std::vector<const onerec::example::Tensor*> tensors;
  GetTensorsFromRecords(records, field.key(), &tensors);
  std::vector<std::vector<int32_t>> tensor_dims;
  if (field.has_reshape()) {
    CHECK_EQ(field.reshape().dim_size(), field.static_shape().dim_size());
    GetTensorDimsWithReshape(tensors, field.reshape(), &tensor_dims);
  } else {
    GetTensorDimsWithoutReshape(tensors, field.static_shape().dim_size(), &tensor_dims);
  }
  DimVector instance_dim_vec;
  if (field.has_batch_padding()) {
    CHECK_EQ(field.batch_padding().dim_size(), field.static_shape().dim_size());
    for (int32_t d = 1; d < field.batch_padding().dim_size(); ++d) {
      if (field.batch_padding().dim(d) != 0) { UNIMPLEMENTED(); }
    }
    const int32_t dim0_padding_method = field.batch_padding().dim(0);
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
  for (int32_t d = 1; d < field.static_shape().dim_size(); ++d) {
    const int32_t dim_size = tensor_dims.at(d).front();
    CHECK(std::all_of(tensor_dims.at(d).cbegin() + 1, tensor_dims.at(d).cend(),
                      [&](const int32_t v) { return v == dim_size; }));
    instance_dim_vec.push_back(dim_size);
  }
  const Shape static_shape(field.static_shape());
  *instance_shape = Shape(instance_dim_vec);
  if (field.is_dynamic()) {
    CHECK_LE(instance_shape->elem_cnt(), static_shape.elem_cnt());
  } else {
    CHECK(*instance_shape == static_shape);
  }
  const int64_t buffer_size = GetBatchSizeInBytes(batch_size, *instance_shape, field.data_type());
  CHECK_LE(buffer_size, buf->size());
  if (field.has_batch_padding()) {
    CopyTensorsToBuffer<true>(tensors, field.data_type(), *instance_shape, buf);
  } else {
    CopyTensorsToBuffer<false>(tensors, field.data_type(), *instance_shape, buf);
  }
}

}  // namespace

OneRecDecoder::OneRecDecoder(std::unique_ptr<BufferedBatchedOneRecReader>&& reader,
                             int32_t batch_size, std::vector<DecodeOneRecFieldConf> fields)
    : batch_size_(batch_size), reader_(std::move(reader)), field_vec_(std::move(fields)) {
  for (auto& field : field_vec_) {
    buffer_vec_.emplace_back(
        GetBatchSizeInBytes(batch_size_, Shape(field.static_shape()), field.data_type()));
  }
  decode_thread_ = std::thread([&]() {
    std::vector<OneRecExampleWrapper> records;
    records.reserve(batch_size_);
    while (true) {
      reader_->Read(&records);
      CHECK_EQ(records.size(), batch_size_);
      std::unique_lock<std::mutex> lock(mutex_);
      cond_.wait(lock, [this]() { return closed_ || !filled_; });
      if (closed_) { break; }
      FOR_RANGE(int32_t, i, 0, field_vec_.size()) {
        const DecodeOneRecFieldConf& field = field_vec_.at(i);
        instance_shape_vec_.emplace_back();
        DecodeField(records, field, &instance_shape_vec_.back(), &buffer_vec_.at(i));
      }
      filled_ = true;
      cond_.notify_all();
      records.clear();
    }
  });
}

OneRecDecoder::~OneRecDecoder() {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    closed_ = true;
    cond_.notify_all();
  }
  decode_thread_.join();
  reader_.reset();
}

bool OneRecDecoder::GetBatch(const std::vector<Blob*>& blobs) {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return closed_ || filled_; });
  if (closed_) { return false; }
  CHECK_EQ(blobs.size(), field_vec_.size());
  FOR_RANGE(int32_t, i, 0, field_vec_.size()) {
    const DecodeOneRecFieldConf& field = field_vec_.at(i);
    Blob* blob = blobs.at(i);
    const Shape& instance_shape = instance_shape_vec_.at(i);
    if (field.is_dynamic()) {
      CHECK_LE(batch_size_ * instance_shape.elem_cnt(), blob->static_shape().elem_cnt());
      CHECK(blob->blob_desc().is_dynamic());
      blob->mut_shape_view()->Set(0, batch_size_);
      for (int64_t d = 0; d < instance_shape.NumAxes(); ++d) {
        blob->mut_shape_view()->Set(d + 1, instance_shape.At(d));
      }
    } else {
      CHECK(!blob->blob_desc().is_dynamic());
      CHECK_EQ(blob->shape().At(0), batch_size_);
      for (int64_t d = 0; d < instance_shape.NumAxes(); ++d) {
        CHECK_EQ(blob->shape().At(d + 1), instance_shape.At(d));
      }
    }
    std::memcpy(blob->mut_dptr(), buffer_vec_.at(i).data(),
                GetBatchSizeInBytes(batch_size_, instance_shape, field.data_type()));
  }
  instance_shape_vec_.clear();
  filled_ = false;
  cond_.notify_all();
  return true;
}

}  // namespace oneflow
