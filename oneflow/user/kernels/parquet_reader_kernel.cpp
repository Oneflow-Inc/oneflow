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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/buffer.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/user/data/parquet_util.h"

#include "parquet/api/reader.h"
#include "arrow/filesystem/localfs.h"

namespace oneflow {

namespace {

template<typename PType>
Maybe<bool> ReadColumnValuesImpl(parquet::ColumnReader* col_reader, size_t footprint,
                                 std::shared_ptr<TensorBuffer>& sample) {
  using T = typename PType::c_type;
  thread_local std::vector<T> values;
  thread_local std::vector<int16_t> rep_levels;
  values.reserve(footprint);
  rep_levels.reserve(footprint);

  // the first value must be the begin of LIST which repetition level should be 0
  // since a LIST is a complete sample
  CHECK_EQ_OR_RETURN(rep_levels.size(), values.size());
  CHECK_EQ_OR_RETURN(rep_levels.front(), 0);
  size_t row_read = 0;
  size_t row_end = 0;

  auto FindRowEnd = [&](size_t start) {
    for (size_t i = start; i < rep_levels.size(); ++i) {
      if (i > 0 && rep_levels[i] == 0) {
        row_end = i;
        row_read += 1;
        break;
      }
    }
  };
  FindRowEnd(0);

  auto* typed_col_reader = static_cast<parquet::TypedColumnReader<PType>*>(col_reader);
  int64_t values_read = 0;
  while (typed_col_reader->HasNext() && row_read == 0) {
    size_t cursor = values.size();
    values.resize(values.size() + footprint);
    rep_levels.resize(rep_levels.size() + footprint);
    auto levels_read = typed_col_reader->ReadBatch(
        footprint, /* def_levels */ nullptr, &rep_levels[cursor], &values[cursor], &values_read);
    CHECK_EQ_OR_RETURN(values_read, levels_read);
    CHECK_LE_OR_RETURN(values_read, footprint);
    if (values_read < footprint) {
      size_t remain = footprint - values_read;
      values.resize(values.size() - remain);
      rep_levels.resize(rep_levels.size() - remain);
    }
    FindRowEnd(cursor);
  }

  // all values can't fill a complete sample
  if (row_read == 0) { return false; }

  CHECK_OR_RETURN(!sample);
  sample.reset(new TensorBuffer());
  sample->Resize(Shape({static_cast<int64_t>(row_end)}), GetDataType<T>::value);
  std::copy(values.begin(), values.begin() + row_end, sample->mut_data<T>());
  values.erase(values.begin(), values.begin() + row_end);
  rep_levels.erase(rep_levels.begin(), rep_levels.begin() + row_end);
  return true;
}

template<typename PType>
Maybe<bool> ReadColumnFixedLengthValuesImpl(parquet::ColumnReader* col_reader, size_t batch_size,
                                            size_t sample_size,
                                            std::shared_ptr<TensorBuffer>& batch) {
  using T = typename PType::c_type;
  thread_local std::vector<T> values;
  thread_local std::vector<int16_t> rep_levels;
  size_t values_to_read = batch_size * sample_size;
  values.reserve(values_to_read);
  rep_levels.reserve(values_to_read);

  CHECK_EQ_OR_RETURN(values.size(), rep_levels.size());
  if (!values.empty()) { values_to_read -= values.size(); }
  size_t cursor = values.size();
  values.resize(values.size() + values_to_read);
  rep_levels.resize(rep_levels.size() + values_to_read);

  auto* typed_col_reader = static_cast<parquet::TypedColumnReader<PType>*>(col_reader);
  int64_t values_read = 0;
  int64_t levels_read = typed_col_reader->ReadBatch(
      values_to_read, /* def_levels */ nullptr, &rep_levels[cursor], &values[cursor], &values_read);
  CHECK_EQ_OR_RETURN(values_read, levels_read);
  CHECK_LE_OR_RETURN(values_read, values_to_read);

  // not enough batch, return false
  if (values_read < values_to_read) {
    size_t remain = values_to_read - values_read;
    values.resize(values.size() - remain);
    rep_levels.resize(rep_levels.size() - remain);
    return false;
  }

  CHECK_EQ_OR_RETURN(values.size() % sample_size, 0);
  // verify repetition levels
  for (size_t i = 0; i < rep_levels.size(); ++i) {
    // repetition level of the first value of LIST must be 0
    // repetition levels of other values of LIST must be 1
    if (i % sample_size == 0) {
      CHECK_EQ_OR_RETURN(rep_levels[i], 0)
          << "the number of values in LIST is greater than sample size " << sample_size
          << " at column " << typed_col_reader->descr()->path()->ToDotString();
    } else {
      CHECK_EQ_OR_RETURN(rep_levels[i], 1)
          << "the number of values in LIST is less than sample size " << sample_size
          << " at column " << typed_col_reader->descr()->path()->ToDotString();
    }
  }

  CHECK_OR_RETURN(!batch);
  batch.reset(new TensorBuffer());
  batch->Resize(Shape({static_cast<int64_t>(values.size())}), GetDataType<T>::value);
  std::copy(values.begin(), values.end(), batch->mut_data<T>());
  values.resize(0);
  rep_levels.resize(0);
  return true;
}

Maybe<bool> ReadColumnValues(parquet::ColumnReader* col_reader, size_t footprint,
                             std::shared_ptr<TensorBuffer>& sample) {
  switch (col_reader->type()) {
#define CASE_ENTRY(type)                                              \
  case type: {                                                        \
    using PhyT = parquet::PhysicalType<type>;                         \
    return ReadColumnValuesImpl<PhyT>(col_reader, footprint, sample); \
  }

    CASE_ENTRY(parquet::Type::INT32)
    CASE_ENTRY(parquet::Type::INT64)
    CASE_ENTRY(parquet::Type::FLOAT)
    CASE_ENTRY(parquet::Type::DOUBLE)
    // TODO(zwx): to support BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY

#undef CASE_ENTRY
    default: {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
}

Maybe<bool> ReadColumnFixedLengthValues(parquet::ColumnReader* col_reader, size_t batch_size,
                                        size_t sample_size, std::shared_ptr<TensorBuffer>& batch) {
  switch (col_reader->type()) {
#define CASE_ENTRY(type)                                                                      \
  case type: {                                                                                \
    using PhyT = parquet::PhysicalType<type>;                                                 \
    return ReadColumnFixedLengthValuesImpl<PhyT>(col_reader, batch_size, sample_size, batch); \
  }

    CASE_ENTRY(parquet::Type::INT32)
    CASE_ENTRY(parquet::Type::INT64)
    CASE_ENTRY(parquet::Type::FLOAT)
    CASE_ENTRY(parquet::Type::DOUBLE)
    // TODO(zwx): to support BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY

#undef CASE_ENTRY
    default: {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
}

DataType ParquetTypeToDataType(parquet::Type::type type) {
  switch (type) {
    case parquet::Type::INT32: {
      return DataType::kInt32;
    }
    case parquet::Type::INT64: {
      return DataType::kInt64;
    }
    case parquet::Type::FLOAT: {
      return DataType::kFloat;
    }
    case parquet::Type::DOUBLE: {
      return DataType::kDouble;
    }
    default: {
      // BOOLEAN
      // INT96
      // FLOAT
      // DOUBLE
      // BYTE_ARRAY
      // FIXED_LEN_BYTE_ARRAY
      return DataType::kInvalidDataType;
    }
  }
}

}  // namespace

namespace data {

class ParquetReader final : public user_op::OpKernelState {
 public:
  ParquetReader() : is_closed_(false){};
  OF_DISALLOW_COPY_AND_MOVE(ParquetReader);
  ~ParquetReader() { Close(); };

  Maybe<void> Init(user_op::KernelInitContext* ctx);
  void Close();

  Maybe<void> GetColumnBatch(int col, TensorBuffer* buffer, size_t batch_size);
  Maybe<void> GetColumnBatch(int col, user_op::Tensor* tensor);

 private:
  Maybe<void> InitParallelInfo(user_op::KernelInitContext* ctx);
  Maybe<void> InitParquetFiles(user_op::KernelInitContext* ctx);
  Maybe<void> InitSchema(user_op::KernelInitContext* ctx);
  Maybe<void> InitWorkers(user_op::KernelInitContext* ctx);
  Maybe<bool> Partition(std::queue<int>* part_indices, size_t total_size, bool do_part);

  bool IsClosed() { return is_closed_.load(); }
  size_t NumColumns() { return schema_.col_descs.size(); }

  Maybe<bool> ReadColumn(int col);
  Maybe<void> NextRowGroup();
  Maybe<void> NextParquetFile();

  Maybe<bool> Cache(int col, std::shared_ptr<TensorBuffer>& sample_or_batch);
  Maybe<bool> ShuffleAndCache(int col, std::shared_ptr<TensorBuffer>& sample_or_batch);
  Maybe<bool> ShuffleByBatchAndCache(int col, std::shared_ptr<TensorBuffer>& sample);
  Maybe<bool> ShuffleAndBatchCache(int col, std::shared_ptr<TensorBuffer>& sample);

  enum ShuffleMode { kNoShuffle = 0, kShuffleSample, kShuffleBatch };
  enum CacheMethod {
    kDirectlyCache = 0,
    kShuffleAndCache,
    kShuffleByBatchAndCache,
    kShuffleAndBatchCache
  };

  ShuffleMode GetShuffleMode();
  CacheMethod GetChechMethod(int col, ShuffleMode shuffle_mode);

  int64_t rank_;
  size_t world_size_;
  bool shuffle_;
  bool completely_shuffle_;
  int64_t seed_;
  std::mt19937 rng_;
  bool use_mmap_;
  size_t read_footprint_;
  size_t batch_size_;
  size_t shuffle_buffer_size_;
  ShuffleMode shuffle_mode_;
  ParquetColumnSchema schema_;

  std::string base_path_;
  std::shared_ptr<arrow::fs::LocalFileSystem> local_fs_;
  std::vector<std::string> parquet_files_;
  std::shared_ptr<parquet::FileMetaData> file_meta_;

  std::unique_ptr<parquet::ParquetFileReader> file_reader_;
  std::shared_ptr<parquet::RowGroupReader> row_group_reader_;

  std::queue<int> row_group_part_indices_;
  std::queue<int> parquet_file_part_indices_;

  std::vector<std::thread> workers_;
  std::vector<std::unique_ptr<Buffer<std::shared_ptr<TensorBuffer>>>> prefetch_buffers_;
  std::vector<std::vector<std::shared_ptr<TensorBuffer>>> shuffle_buffers_;
  std::vector<std::mt19937> col_gens_;
  std::vector<CacheMethod> cache_methods_;

  std::mutex mutex_;
  std::condition_variable cond_;
  std::atomic_bool is_closed_;
  int read_keys_;
  bool read_done_;
};

void ParquetReader::Close() {
  if (!is_closed_.load()) {
    is_closed_.store(true);
    for (auto& buffer : prefetch_buffers_) { buffer->Close(); }
    for (auto& worker : workers_) {
      if (worker.joinable()) { worker.join(); }
    }
  }
}

Maybe<void> ParquetReader::GetColumnBatch(int col, TensorBuffer* buffer, size_t batch_size) {
  CHECK_GE_OR_RETURN(col, 0);
  CHECK_LT_OR_RETURN(col, NumColumns());
  CHECK_EQ_OR_RETURN(batch_size, batch_size_);
  std::vector<std::shared_ptr<TensorBuffer>> batch(batch_size);
  if (prefetch_buffers_[col]->PullMany(batch.data(), batch.size())
      != BufferStatus::kBufferStatusSuccess) {
    return Maybe<void>::Ok();
  }
  for (auto& sample : batch) {
    buffer->Swap(sample.get());
    buffer++;
  }
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::GetColumnBatch(int col, user_op::Tensor* tensor) {
  CHECK_GE_OR_RETURN(col, 0);
  CHECK_LT_OR_RETURN(col, NumColumns());
  CHECK_GT_OR_RETURN(tensor->shape().NumAxes(), 0);
  CHECK_EQ_OR_RETURN(tensor->shape().At(0), batch_size_);

  std::shared_ptr<TensorBuffer> batch;
  if (prefetch_buffers_[col]->Pull(&batch) != BufferStatus::kBufferStatusSuccess) {
    return Maybe<void>::Ok();
  }
  CHECK_EQ_OR_RETURN(batch->data_type(), tensor->data_type())
      << "The data type of batch read from parquet dismatch the data type of tensor";
  CHECK_EQ_OR_RETURN(batch->elem_cnt(), tensor->shape().elem_cnt())
      << "The shape of batch read from parquet dismatch the shape of tensor: "
      << batch->shape().ToString() << " vs. " << tensor->shape().ToString()
      << " (tensor shape include batch dimension)";
  memcpy(tensor->mut_dptr(), batch->mut_data(), batch->nbytes());
  return Maybe<void>::Ok();
}

Maybe<bool> ParquetReader::ReadColumn(int col) {
  // when all col readers finish reading, it's the last finishing reader's due
  // to move to next row group
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (--read_keys_ == 0) { read_done_ = true; }
    if (row_group_reader_) { row_group_reader_.reset(); }
    cond_.wait(lock, [this]() { return read_done_ || IsClosed(); });
    if (!row_group_reader_) { JUST(NextRowGroup()); }
    if (++read_keys_ == workers_.size()) { read_done_ = false; }
  }
  cond_.notify_all();

  const auto& col_desc = schema_.col_descs[col];
  int col_id = col_desc.col_id;
  CHECK_GE_OR_RETURN(col_id, 0);
  CHECK_LT_OR_RETURN(col_id, row_group_reader_->metadata()->num_columns());
  auto col_reader = row_group_reader_->Column(col_id);
  while (col_reader->HasNext()) {
    // read from parquet column chunk to TensorBuffer
    std::shared_ptr<TensorBuffer> sample_or_batch;
    bool read_done = false;
    if (col_desc.is_variadic) {
      read_done = JUST(ReadColumnValues(col_reader.get(), read_footprint_, sample_or_batch));
    } else {
      size_t bz = completely_shuffle_ ? 1 : batch_size_;
      read_done = JUST(ReadColumnFixedLengthValues(col_reader.get(), bz, col_desc.shape.elem_cnt(),
                                                   sample_or_batch));
    }
    if (!read_done) { continue; }
    if (!JUST(Cache(col, sample_or_batch))) { return false; }
  }
  return true;
}

ParquetReader::ShuffleMode ParquetReader::GetShuffleMode() {
  if (!shuffle_) { return kNoShuffle; }
  if (completely_shuffle_) { return kShuffleSample; }

  int64_t prefer_shuffle_batch = 0;
  int64_t prefer_shuffle_sample = 0;
  for (int col = 0; col < NumColumns(); ++col) {
    const auto& col_desc = schema_.col_descs[col];
    size_t bytes_size = col_desc.shape.elem_cnt() * GetSizeOfDataType(col_desc.dtype);
    if (col_desc.is_variadic) {
      prefer_shuffle_sample += bytes_size;
    } else {
      prefer_shuffle_batch += bytes_size;
    }
  }
  if (prefer_shuffle_sample > prefer_shuffle_batch) { return kShuffleSample; }
  return kShuffleBatch;
}

ParquetReader::CacheMethod ParquetReader::GetChechMethod(int col, ShuffleMode shuffle_mode) {
  if (shuffle_mode == kNoShuffle) { return kDirectlyCache; }
  const auto& col_desc = schema_.col_descs[col];
  if (col_desc.is_variadic) {
    if (shuffle_mode == kShuffleSample) {
      return kShuffleAndCache;
    } else if (shuffle_mode == kShuffleBatch) {
      return kShuffleByBatchAndCache;
    }
  } else {
    if (shuffle_mode == kShuffleSample) {
      return kShuffleAndBatchCache;
    } else if (shuffle_mode == kShuffleBatch) {
      return kShuffleAndCache;
    }
  }
  return kDirectlyCache;
}

Maybe<bool> ParquetReader::Cache(int col, std::shared_ptr<TensorBuffer>& sample_or_batch) {
  switch (cache_methods_[col]) {
    case kDirectlyCache: {
      if (prefetch_buffers_[col]->Push(sample_or_batch) != BufferStatus::kBufferStatusSuccess) {
        return false;
      }
      return true;
    }
    case kShuffleAndCache: {
      return ShuffleAndCache(col, sample_or_batch);
    }
    case kShuffleByBatchAndCache: {
      return ShuffleByBatchAndCache(col, sample_or_batch);
    }
    case kShuffleAndBatchCache: {
      return ShuffleAndBatchCache(col, sample_or_batch);
    }
    default: {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
}

Maybe<bool> ParquetReader::ShuffleAndCache(int col,
                                           std::shared_ptr<TensorBuffer>& sample_or_batch) {
  // set shuffle buffer size to be batch size times since the item saved in buffer is a sample
  size_t buffer_size = 0;
  if (shuffle_mode_ == kShuffleBatch) {
    buffer_size = shuffle_buffer_size_;
  } else if (shuffle_mode_ == kShuffleSample) {
    buffer_size = shuffle_buffer_size_ * batch_size_;
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }

  auto& shuffle_buffer = shuffle_buffers_[col];
  if (shuffle_buffer.size() < buffer_size) {
    shuffle_buffer.push_back(std::move(sample_or_batch));
    return true;
  }

  std::uniform_int_distribution<size_t> dis(0, shuffle_buffer.size() - 1);
  size_t offset = dis(col_gens_[col]);
  std::swap(shuffle_buffer[offset], sample_or_batch);
  if (prefetch_buffers_[col]->Push(sample_or_batch) != BufferStatus::kBufferStatusSuccess) {
    return false;
  }
  return true;
}

Maybe<bool> ParquetReader::ShuffleByBatchAndCache(int col, std::shared_ptr<TensorBuffer>& sample) {
  auto& shuffle_buffer = shuffle_buffers_[col];
  if (shuffle_buffer.size() < (shuffle_buffer_size_ + 1) * batch_size_) {
    shuffle_buffer.push_back(std::move(sample));
    return true;
  }

  std::uniform_int_distribution<size_t> dis(0, shuffle_buffer_size_ - 1);
  size_t offset = dis(col_gens_[col]) * batch_size_;
  for (size_t i = offset; i < offset + batch_size_; ++i) {
    std::swap(shuffle_buffer[i], sample);
    if (prefetch_buffers_[col]->Push(sample) != BufferStatus::kBufferStatusSuccess) {
      return false;
    }
  }
  return true;
}

Maybe<bool> ParquetReader::ShuffleAndBatchCache(int col, std::shared_ptr<TensorBuffer>& sample) {
  const size_t buffer_size = shuffle_buffer_size_ * batch_size_;
  auto& shuffle_buffer = shuffle_buffers_[col];
  if (shuffle_buffer.size() < buffer_size) {
    shuffle_buffer.push_back(std::move(sample));
    return true;
  }

  if (shuffle_buffer.size() < buffer_size + batch_size_) {
    std::uniform_int_distribution<size_t> dis(0, buffer_size - 1);
    size_t offset = dis(col_gens_[col]);
    std::swap(shuffle_buffer[offset], sample);
    shuffle_buffer.push_back(std::move(sample));
    if (shuffle_buffer.size() != buffer_size + batch_size_) { return true; }
  }

  const auto& col_desc = schema_.col_descs[col];
  int64_t sample_size = col_desc.shape.elem_cnt();
  auto batch = std::make_shared<TensorBuffer>();
  batch->Resize(Shape({static_cast<int64_t>(batch_size_), sample_size}), col_desc.dtype);
  char* dptr = static_cast<char*>(batch->mut_data());
  size_t total_bytes = 0;
  for (auto it = shuffle_buffer.begin() + buffer_size; it != shuffle_buffer.end(); ++it) {
    const auto& sample_ = *it;
    CHECK_OR_RETURN(sample_);
    CHECK_EQ_OR_RETURN(sample_->data_type(), batch->data_type());
    CHECK_EQ_OR_RETURN(sample_->elem_cnt(), sample_size);
    memcpy(dptr, sample_->data(), sample_->nbytes());
    dptr += sample_->nbytes();
    total_bytes += sample_->nbytes();
  }
  CHECK_EQ_OR_RETURN(total_bytes, batch->nbytes());
  shuffle_buffer.resize(buffer_size);
  if (prefetch_buffers_[col]->Push(batch) != BufferStatus::kBufferStatusSuccess) { return false; }
  return true;
}

Maybe<void> ParquetReader::NextRowGroup() {
  CHECK_OR_RETURN(!row_group_reader_);
  if (row_group_part_indices_.empty()) {
    JUST(NextParquetFile());
    int num_row_groups = file_reader_->metadata()->num_row_groups();
    CHECK_OR_RETURN(
        JUST(Partition(&row_group_part_indices_, num_row_groups, parquet_files_.size() == 1)))
        << "There are only " << num_row_groups << " row groups in parquet file of "
        << parquet_files_[0] << " that can't be partitioned by " << world_size_;
  }
  LOG(INFO) << "ParquetReader::NextRowGroup: " << row_group_part_indices_.front();
  row_group_reader_ = file_reader_->RowGroup(row_group_part_indices_.front());
  row_group_part_indices_.pop();
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::NextParquetFile() {
  CHECK_GT_OR_RETURN(parquet_files_.size(), 0);
  if (parquet_files_.size() == 1) {
    if (!file_reader_) {
      file_reader_ = parquet::ParquetFileReader::OpenFile(parquet_files_[0], use_mmap_);
    }
    return Maybe<void>::Ok();
  }

  if (parquet_file_part_indices_.empty()) {
    CHECK_OR_RETURN(JUST(
        Partition(&parquet_file_part_indices_, parquet_files_.size(), parquet_files_.size() > 1)))
        << "There are only " << parquet_files_.size() << " parquet files in " << base_path_
        << " that can't be partitioned by " << world_size_;
  }

  int file_index = parquet_file_part_indices_.front();
  parquet_file_part_indices_.pop();
  file_reader_ = parquet::ParquetFileReader::OpenFile(parquet_files_[file_index], use_mmap_);
  CHECK_OR_RETURN(file_reader_->metadata()->schema()->Equals(*file_meta_->schema()))
      << "The schema of " << parquet_files_[file_index] << " dismatch";
  LOG(INFO) << "ParquetReader::NextParquetFile: " << parquet_files_[file_index];
  return Maybe<void>::Ok();
}

Maybe<bool> ParquetReader::Partition(std::queue<int>* part_indices, size_t total_size,
                                     bool need_do_part) {
  std::vector<int> indices;
  indices.resize(total_size, 0);
  std::iota(indices.begin(), indices.end(), 0);
  if (shuffle_) { std::shuffle(indices.begin(), indices.end(), rng_); }
  if (need_do_part && world_size_ > 1) {
    if (total_size < world_size_) { return false; }
    size_t part_size = total_size / world_size_;
    for (size_t i = rank_ * part_size; i < (rank_ + 1) * part_size; ++i) {
      part_indices->push(indices.at(i));
    }
  } else {
    for (int index : indices) { part_indices->push(index); }
  }
  return true;
}

Maybe<void> ParquetReader::InitWorkers(user_op::KernelInitContext* ctx) {
  size_t num_cols = NumColumns();
  read_keys_ = num_cols;
  read_done_ = false;

  workers_.reserve(num_cols);
  prefetch_buffers_.reserve(num_cols);
  shuffle_buffers_.reserve(num_cols);

  size_t prefetch_buffer_size = ctx->Attr<int64_t>("prefetch_buffer_size");
  shuffle_buffer_size_ = ctx->Attr<int64_t>("shuffle_buffer_size");
  shuffle_mode_ = GetShuffleMode();
  for (int col = 0; col < num_cols; ++col) {
    const auto& col_desc = schema_.col_descs[col];
    // init prefetch buffer
    if (col_desc.is_variadic) { prefetch_buffer_size *= batch_size_; }
    prefetch_buffers_.emplace_back(
        std::make_unique<Buffer<std::shared_ptr<TensorBuffer>>>(prefetch_buffer_size));
    // init shuffle buffer
    shuffle_buffers_.emplace_back();
    // init read workers
    workers_.emplace_back(std::thread([this, col] {
      while (!IsClosed() && CHECK_JUST(ReadColumn(col))) {}
    }));
    // init generators
    int64_t sub_seed = seed_ + rank_ + 1;
    col_gens_.emplace_back(sub_seed);
    // init cache method
    cache_methods_.emplace_back(GetChechMethod(col, shuffle_mode_));
  }
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::InitParquetFiles(user_op::KernelInitContext* ctx) {
  // Find all parquet files
  base_path_ = ctx->Attr<std::string>("path");
  local_fs_ = std::make_shared<arrow::fs::LocalFileSystem>();
  auto result = local_fs_->GetFileInfo(base_path_);
  CHECK_OR_RETURN(result.ok()) << "ParquetReader can't open " << base_path_;
  auto file_info = result.ValueOrDie();
  if (file_info.IsDirectory()) {
    arrow::fs::FileSelector sel;
    sel.base_dir = file_info.path();
    auto result = local_fs_->GetFileInfo(sel);
    CHECK_OR_RETURN(result.ok()) << base_path_ << " is empty";
    for (auto& sub_file_info : result.ValueOrDie()) {
      if (sub_file_info.extension() == "parquet") {
        parquet_files_.push_back(sub_file_info.path());
      }
    }
  } else if (file_info.IsFile()) {
    CHECK_EQ_OR_RETURN(file_info.extension(), "parquet")
        << base_path_ << "is not a file with suffix .parquet";
    parquet_files_.push_back(file_info.path());
  }
  CHECK_GT_OR_RETURN(parquet_files_.size(), 0) << base_path_ << " contains no *.parquet file";
  std::sort(parquet_files_.begin(), parquet_files_.end());
  // parquet file metadata
  auto first_parquet_file_reader =
      parquet::ParquetFileReader::OpenFile(parquet_files_[0], use_mmap_);
  CHECK_OR_RETURN(first_parquet_file_reader) << "Can't open parquet file " << parquet_files_[0];
  file_meta_ = first_parquet_file_reader->metadata();
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::InitParallelInfo(user_op::KernelInitContext* ctx) {
  auto nd_sbp_str_vec = ctx->Attr<std::vector<std::string>>("nd_sbp");
  if (nd_sbp_str_vec.empty() && JUST(IsMultiClient())) {
    world_size_ = GlobalProcessCtx::WorldSize();
    rank_ = GlobalProcessCtx::Rank();
  } else {
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    CHECK_EQ_OR_RETURN(hierarchy.NumAxes(), nd_sbp_str_vec.size());
    rank_ = 0;
    world_size_ = 1;

    using index_helper_t = NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE>;
    index_helper_t index_helper(hierarchy.dim_vec().data(), hierarchy.NumAxes());
    int64_t nd_index[SHAPE_MAX_AXIS_SIZE] = {0};
    index_helper.OffsetToNdIndex(ctx->parallel_ctx().parallel_id(), nd_index);

    for (int i = hierarchy.NumAxes() - 1; i >= 0; --i) {
      cfg::SbpParallel sbp;
      CHECK_OR_RETURN(ParseSbpParallelFromString(nd_sbp_str_vec[i], &sbp));
      if (sbp.has_split_parallel()) {
        rank_ += nd_index[i] * world_size_;
        world_size_ *= hierarchy.At(i);
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::InitSchema(user_op::KernelInitContext* ctx) {
  const std::string& schema_json_str = ctx->Attr<std::string>("schema_json_str");
  ParseParquetColumnSchemaFromJson(&schema_, schema_json_str);
  int output_index = 0;
  HashSet<int> spec_cols;
  const size_t num_cols = file_meta_->schema()->num_columns();
  for (auto& out_desc : schema_.col_descs) {
    // Find column descriptor by col_id or col_name
    if (out_desc.col_name.empty()) {
      if (out_desc.col_id < 0 || out_desc.col_id >= num_cols) {
        return Error::RuntimeError() << "The schema of output " << output_index
                                     << " has invalid col_id: " << out_desc.col_id
                                     << ", that must be set in range [0, " << num_cols << ").";
      }
    } else {
      bool col_name_found = false;
      for (int col = 0; col < num_cols; ++col) {
        auto* col_desc = file_meta_->schema()->Column(col);
        CHECK_NOTNULL_OR_RETURN(col_desc);
        const auto& col_path = col_desc->path()->ToDotString();
        auto first_name = col_path.substr(0, col_path.find('.'));
        if (out_desc.col_name == first_name) {
          if (out_desc.col_id != -1 && out_desc.col_id != col) {
            LOG(WARNING) << "col_id " << out_desc.col_id << " doesn't match col_name "
                         << out_desc.col_name << " which is corresponds to col_id " << col;
          }
          out_desc.col_id = col;
          col_name_found = true;
        }
      }
      CHECK_OR_RETURN(col_name_found)
          << "col_name " << out_desc.col_name << " is not found in parquet file schema";
    }
    CHECK_OR_RETURN((spec_cols.insert(out_desc.col_id)).second)
        << "Duplicated column with col_id: " << out_desc.col_id
        << ", col_name: " << out_desc.col_name;
    const auto* col_desc = file_meta_->schema()->Column(out_desc.col_id);
    CHECK_NOTNULL_OR_RETURN(col_desc);
    // check def and rep level
    if (!((col_desc->max_definition_level() == 1 && col_desc->max_repetition_level() == 0)
          || (col_desc->max_repetition_level() == 1 && col_desc->name() == "element"))) {
      return Error::RuntimeError() << "It's only supported for now that read values with primitive "
                                      "type or LIST type from column";
    }
    // check data type
    DataType data_type = ParquetTypeToDataType(col_desc->physical_type());
    CHECK_NE_OR_RETURN(data_type, DataType::kInvalidDataType)
        << "Unsupported parquet physical type " << col_desc->physical_type() << " for column "
        << out_desc.col_id;
    CHECK_EQ_OR_RETURN(data_type, out_desc.dtype)
        << "The configured column dtype dismatch the type of column in schema";
    // check batch_size
    const auto& out_shape = ctx->TensorDesc4ArgNameAndIndex("out", output_index)->shape();
    CHECK_GE_OR_RETURN(out_shape.NumAxes(), 1);
    size_t cur_out_batch_size = out_shape.At(0);
    if (output_index == 0) {
      batch_size_ = cur_out_batch_size;
    } else {
      CHECK_EQ_OR_RETURN(batch_size_, cur_out_batch_size)
          << "The " << output_index << "th output has different batch size: " << cur_out_batch_size;
    }
    output_index++;
  }
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::Init(user_op::KernelInitContext* ctx) {
  shuffle_ = ctx->Attr<bool>("shuffle");
  completely_shuffle_ = ctx->Attr<bool>("completely_shuffle") && shuffle_;
  seed_ = ctx->Attr<int64_t>("random_seed");
  rng_ = std::mt19937(seed_);
  use_mmap_ = ctx->Attr<bool>("use_mmap");
  read_footprint_ = ctx->Attr<int64_t>("read_footprint");
  JUST(InitParallelInfo(ctx));
  JUST(InitParquetFiles(ctx));
  JUST(InitSchema(ctx));
  JUST(InitWorkers(ctx));
  return Maybe<void>::Ok();
}

}  // namespace data

class ParquetReaderKernel final : public user_op::OpKernel {
 public:
  ParquetReaderKernel() = default;
  ~ParquetReaderKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    auto parquet_reader = std::make_shared<data::ParquetReader>();
    CHECK_JUST(parquet_reader->Init(ctx));
    return parquet_reader;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    auto* parquet_reader = static_cast<data::ParquetReader*>(state);
    size_t output_size = ctx->output_size("out");
    MultiThreadLoop(output_size, [&](size_t i) {
      user_op::Tensor* out_i = ctx->Tensor4ArgNameAndIndex("out", i);
      if (out_i->data_type() == DataType::kTensorBuffer) {
        TensorBuffer* out_buffer = out_i->mut_dptr<TensorBuffer>();
        size_t batch_size = out_i->shape().At(0);
        CHECK_JUST(parquet_reader->GetColumnBatch(i, out_buffer, batch_size));
      } else {
        CHECK_JUST(parquet_reader->GetColumnBatch(i, out_i));
      }
    });
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("parquet_reader")
    .SetCreateFn<ParquetReaderKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU));

}  // namespace oneflow
