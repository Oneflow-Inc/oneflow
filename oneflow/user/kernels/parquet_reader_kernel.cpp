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
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/tensor_buffer.h"
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

#include "parquet/api/reader.h"
#include "parquet/column_reader.h"
#include "parquet/file_reader.h"
#include "parquet/types.h"
#include "arrow/filesystem/localfs.h"

namespace oneflow {

namespace {

template<typename DType>
Maybe<TensorBuffer> ReadColumnValuesImpl(parquet::ColumnReader* col_reader) {
  using T = typename DType::c_type;
  thread_local std::vector<T> values;
  auto* typed_col_reader = static_cast<parquet::TypedColumnReader<DType>*>(col_reader);
  int64_t values_read = 0;
  int16_t def_level = 0;
  int16_t rep_level = 0;
  auto buffer = std::make_shared<TensorBuffer>();
  while (typed_col_reader->HasNext()) {
    T value;
    auto levels_read = typed_col_reader->ReadBatch(1, &def_level, &rep_level, &value, &values_read);
    CHECK_EQ_OR_RETURN(levels_read, 1);
    CHECK_EQ_OR_RETURN(values_read, 1);
    if (rep_level == 0) {
      buffer->Resize(Shape({static_cast<int64_t>(values.size())}), GetDataType<T>::value);
      std::copy(values.begin(), values.end(), buffer->mut_data<T>());
      values.resize(0);
      values.push_back(std::move(value));
      break;
    } else if (rep_level == 1) {
      values.push_back(std::move(value));
    } else {
      UNIMPLEMENTED_THEN_RETURN();
    }
  }
  return buffer;
}

Maybe<TensorBuffer> ReadColumnValues(parquet::ColumnReader* col_reader) {
  switch (col_reader->type()) {
#define CASE_ENTRY(type)                           \
  case type: {                                     \
    using PhyT = parquet::PhysicalType<type>;      \
    return ReadColumnValuesImpl<PhyT>(col_reader); \
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

template<typename T>
class RandomShuffleBuffer final {
 public:
  enum Status { kSuccess = 0, kClosed, kEmpty };

  RandomShuffleBuffer(size_t max_size, bool shuffle, int64_t seed)
      : max_size_(max_size), shuffle_(shuffle), rng_(seed), is_closed_(false) {}
  OF_DISALLOW_COPY_AND_MOVE(RandomShuffleBuffer);
  ~RandomShuffleBuffer() = default;

  bool IsClosed() { return is_closed_; }
  bool IsFull() { return queue_.size() >= max_size_; }
  bool IsEmpty() { return queue_.empty(); }

  template<typename U>
  Status Push(U&& item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]() { return !IsFull() || IsClosed(); });
    if (is_closed_) { return kClosed; }

    queue_.push_back(std::move(item));
    if (shuffle_ && queue_.size() > 2) {
      std::uniform_int_distribution<size_t> dis(0, queue_.size() - 1);
      size_t offset = dis(rng_);
      std::swap(queue_[offset], queue_.back());
    }

    cond_.notify_one();
    return kSuccess;
  }

  Status Pull(T* item) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this]() { return !IsEmpty() || IsClosed(); });
    if (IsEmpty()) { return kEmpty; }
    *item = std::move(queue_.front());
    queue_.pop_front();
    cond_.notify_one();
    return kSuccess;
  }

  Status PullMany(std::vector<T>* item_vec, size_t size) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this, size]() { return queue_.size() >= size || IsClosed(); });
    if (IsEmpty()) { return kEmpty; }
    while (!queue_.empty()) {
      item_vec->push_back(std::move(queue_.front()));
      queue_.pop_front();
    }
    cond_.notify_one();
    return kSuccess;
  }

  void Close() {
    std::unique_lock<std::mutex> lock(mutex_);
    is_closed_ = true;
    cond_.notify_all();
  }

 private:
  std::deque<T> queue_;
  size_t max_size_;
  bool shuffle_;
  std::mt19937 rng_;
  bool is_closed_;
  std::mutex mutex_;
  std::condition_variable cond_;
};

}  // namespace

namespace data {

class ParquetReader final : public user_op::OpKernelState {
 public:
  using BufferType = RandomShuffleBuffer<std::shared_ptr<TensorBuffer>>;

  ParquetReader() : is_closed_(false){};
  OF_DISALLOW_COPY_AND_MOVE(ParquetReader);
  ~ParquetReader() { Close(); };

  Maybe<void> Init(user_op::KernelInitContext* ctx);
  void Close();

  Maybe<void> GetColumnBatch(int col, TensorBuffer* buffer, size_t batch_size);
  Maybe<void> GetColumnBatch(int col, user_op::Tensor* tensor);

 private:
  Maybe<void> InitParallelInfo(user_op::KernelInitContext* ctx);
  Maybe<void> InitParquetFiles(const std::string& path);
  Maybe<void> InitWorkers(size_t buffer_size);

  bool IsClosed() { return is_closed_.load(); }
  bool AllColumnsRanOut();
  size_t NumColumns() { return num_columns_; }

  Maybe<bool> ReadColumn(int col);
  Maybe<void> NextRowGroup();
  Maybe<void> NextParquetFile();

  int64_t rank_;
  size_t world_size_;
  bool shuffle_;
  int64_t seed_;
  std::mt19937 rng_;
  bool use_mmap_;
  size_t num_columns_;

  std::string base_path_;
  std::shared_ptr<arrow::fs::LocalFileSystem> local_fs_;
  std::vector<std::string> parquet_files_;

  std::unique_ptr<parquet::ParquetFileReader> file_reader_;
  std::shared_ptr<parquet::RowGroupReader> row_group_reader_;

  std::vector<bool> columns_ran_out_;
  std::queue<int> row_group_part_indices_;
  std::queue<int> parquet_file_part_indices_;

  std::vector<std::thread> workers_;
  std::vector<std::unique_ptr<BufferType>> buffers_;
  std::mutex mutex_;
  std::condition_variable cond_;
  std::atomic_bool is_closed_;
};

void ParquetReader::Close() {
  if (!is_closed_.load()) {
    is_closed_.store(true);
    for (auto& buffer : buffers_) { buffer->Close(); }
    for (auto& worker : workers_) { worker.join(); }
  }
}

Maybe<void> ParquetReader::GetColumnBatch(int col, TensorBuffer* buffer, size_t batch_size) {
  std::vector<std::shared_ptr<TensorBuffer>> batch;
  batch.reserve(batch_size);
  buffers_[col]->PullMany(&batch, batch_size);
  CHECK_EQ(batch.size(), batch_size);
  for (auto& sample : batch) {
    buffer->Swap(sample.get());
    buffer++;
  }
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::GetColumnBatch(int col, user_op::Tensor* tensor) {
  CHECK_GT_OR_RETURN(tensor->shape().NumAxes(), 0);
  size_t batch_size = tensor->shape().At(0);
  std::vector<std::shared_ptr<TensorBuffer>> batch;
  batch.reserve(batch_size);
  buffers_[col]->PullMany(&batch, batch_size);
  CHECK_EQ_OR_RETURN(batch.size(), batch_size);
  char* dptr = static_cast<char*>(tensor->mut_dptr());
  DataType dtype = tensor->data_type();
  int64_t sample_size = tensor->shape().Count(1);
  for (auto& sample : batch) {
    CHECK_EQ_OR_RETURN(sample->data_type(), dtype);
    CHECK_EQ_OR_RETURN(sample->elem_cnt(), sample_size);
    memcpy(dptr, sample->mut_data(), sample->nbytes());
    dptr += sample->nbytes();
  }
  return Maybe<void>::Ok();
}

bool ParquetReader::AllColumnsRanOut() {
  return std::all_of(columns_ran_out_.begin(), columns_ran_out_.end(),
                     [](const bool& v) { return v; });
}

Maybe<bool> ParquetReader::ReadColumn(int col) {
  {
    std::unique_lock<std::mutex> lock(mutex_);
    columns_ran_out_[col] = true;
    row_group_reader_.reset();
    cond_.wait(lock, [this]() { return AllColumnsRanOut() || IsClosed(); });
    if (!row_group_reader_) {
      JUST(NextRowGroup());
      cond_.notify_all();
    }
    columns_ran_out_[col] = false;
  }
  auto col_reader = row_group_reader_->Column(col);
  while (col_reader->HasNext()) {
    auto sample = JUST(ReadColumnValues(col_reader.get()));
    if (buffers_[col]->Push(std::move(sample)) != BufferType::kSuccess) { return false; }
  }
  return true;
}

Maybe<void> ParquetReader::NextRowGroup() {
  CHECK_OR_RETURN(!row_group_reader_);
  if (row_group_part_indices_.empty()) {
    JUST(NextParquetFile());
    std::vector<int> row_group_indices;
    row_group_indices.resize(file_reader_->metadata()->num_row_groups(), 0);
    std::iota(row_group_indices.begin(), row_group_indices.end(), 0);
    if (shuffle_) { std::shuffle(row_group_indices.begin(), row_group_indices.end(), rng_); }
    if (parquet_files_.size() == 1 && world_size_ > 1) {
      // need partition row groups
      CHECK_GE_OR_RETURN(row_group_indices.size(), world_size_)
          << "There are only " << row_group_indices.size() << " row groups in parquet file of "
          << parquet_files_[0] << " that can't be partitioned by " << world_size_;
      size_t part_size = row_group_indices.size() / world_size_;
      for (size_t i = rank_ * part_size; i < (rank_ + 1) * part_size; ++i) {
        row_group_part_indices_.push(row_group_indices[i]);
      }
    } else {
      for (int index : row_group_indices) { row_group_part_indices_.push(index); }
    }
  }
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
    std::vector<int> parquet_file_indices;
    parquet_file_indices.resize(parquet_files_.size(), 0);
    if (shuffle_) { std::shuffle(parquet_file_indices.begin(), parquet_file_indices.end(), rng_); }
    if (world_size_ > 1) {
      CHECK_GE_OR_RETURN(parquet_files_.size(), world_size_)
          << "There are only " << parquet_files_.size() << " parquet files in " << base_path_
          << " that can't be partitioned by " << world_size_;
      size_t part_size = parquet_file_indices.size() / world_size_;
      for (size_t i = rank_ * part_size; i < (rank_ + 1) * part_size; ++i) {
        parquet_file_part_indices_.push(parquet_file_indices[i]);
      }
    } else {
      for (int index : parquet_file_indices) { parquet_file_part_indices_.push(index); }
    }
  }

  int file_index = parquet_file_part_indices_.front();
  parquet_file_part_indices_.pop();
  file_reader_ = parquet::ParquetFileReader::OpenFile(parquet_files_[file_index], use_mmap_);
  CHECK_EQ_OR_RETURN(file_reader_->metadata()->num_columns(), NumColumns())
      << parquet_files_[file_index] << " has different number of columns.";
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::InitWorkers(size_t buffer_size) {
  buffers_.reserve(NumColumns());
  workers_.reserve(NumColumns());
  for (int i = 0; i < NumColumns(); ++i) {
    buffers_.emplace_back(std::make_unique<BufferType>(buffer_size, shuffle_, seed_ + i + 1));
  }
  columns_ran_out_.resize(NumColumns(), false);
  for (int col = 0; col < NumColumns(); ++col) {
    workers_.emplace_back(std::thread([this, col] {
      while (!IsClosed() && CHECK_JUST(ReadColumn(col))) {}
    }));
  }
  return Maybe<void>::Ok();
}

Maybe<void> ParquetReader::InitParquetFiles(const std::string& path) {
  // Find all parquet files
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
  // parquet file metadata
  auto first_parquet_file_reader =
      parquet::ParquetFileReader::OpenFile(parquet_files_[0], use_mmap_);
  CHECK_OR_RETURN(first_parquet_file_reader) << "Can't open parquet file " << parquet_files_[0];
  num_columns_ = first_parquet_file_reader->metadata()->num_columns();
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

Maybe<void> ParquetReader::Init(user_op::KernelInitContext* ctx) {
  shuffle_ = ctx->Attr<bool>("shuffle");
  seed_ = ctx->Attr<int64_t>("seed");
  rng_ = std::mt19937(seed_);
  use_mmap_ = ctx->Attr<bool>("use_mmap");
  JUST(InitParallelInfo(ctx));
  JUST(InitParquetFiles(ctx->Attr<std::string>("path")));
  JUST(InitWorkers(ctx->Attr<int64_t>("preload_buffer_size")));
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
    parquet_reader->Init(ctx);
    return parquet_reader;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache* cache) const override {
    auto* parquet_reader = static_cast<data::ParquetReader*>(state);
    size_t output_size = ctx->output_size("out");
    // NOTE(zwx): Could use MultiThreadLoop to process output tensors parallelly
    for (size_t i = 0; i < output_size; ++i) {
      user_op::Tensor* out_i = ctx->Tensor4ArgNameAndIndex("out", i);
      if (out_i->data_type() == DataType::kTensorBuffer) {
        TensorBuffer* out_buffer = out_i->mut_dptr<TensorBuffer>();
        size_t batch_size = out_i->shape().At(0);
        CHECK_JUST(parquet_reader->GetColumnBatch(i, out_buffer, batch_size));
        // TODO(zwx): get tensor shape from column schema and resize tensor buffer
        // out_buffer->Resize(shape);
        UNIMPLEMENTED();
      } else {
        CHECK_JUST(parquet_reader->GetColumnBatch(i, out_i));
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("parquet_reader")
    .SetCreateFn<ParquetReaderKernel>()
    .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU));

}  // namespace oneflow
