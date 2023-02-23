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
#include "oneflow/core/common/buffer.h"
#include "oneflow/core/embedding/posix_file.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/channel.h"

namespace oneflow {

namespace {

struct Block {
  size_t file_index;
  size_t offset_in_file;
};

struct BatchReaderRequest {
  std::shared_ptr<std::vector<size_t>> blocks;
  void* buffer{};
};

class BatchReader {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchReader);
  BatchReader(std::vector<std::unique_ptr<embedding::PosixFile>>&& files,
              std::vector<Block>&& blocks, size_t block_size_bytes, size_t num_workers)
      : head_(0),
        tail_(0),
        files_(std::move(files)),
        blocks_(blocks),
        block_size_bytes_(block_size_bytes),
        num_workers_(num_workers) {
    for (size_t i = 0; i < num_workers_; ++i) {
      Worker worker;
      auto* sq = new Channel<BatchReaderRequest>();
      auto* cq = new Channel<BatchReaderRequest>();
      worker.sq.reset(sq);
      worker.cq.reset(cq);
      worker.thread = std::thread([sq, cq, this]() {
        while (true) {
          BatchReaderRequest request;
          auto status = sq->Receive(&request);
          if (status == kChannelStatusErrorClosed) { break; }
          CHECK_EQ(status, kChannelStatusSuccess) << "channel error";
          size_t buffer_offset = 0;
          for (size_t i = 0; i < request.blocks->size(); ++i) {
            size_t block_index = request.blocks->at(i);
            const Block& block = blocks_[block_index];
            size_t remaining = block_size_bytes_;
            size_t file_index = block.file_index;
            size_t file_offset = block.offset_in_file;
            while (remaining != 0) {
              const size_t bytes_to_read =
                  std::min(remaining, files_.at(file_index)->Size() - file_offset);
              PCHECK(pread(files_[file_index]->fd(),
                           reinterpret_cast<unsigned char*>(request.buffer) + buffer_offset,
                           bytes_to_read, file_offset)
                     == bytes_to_read)
                  << "file read error";
              remaining -= bytes_to_read;
              buffer_offset += bytes_to_read;
              if (remaining != 0) {
                file_index = (file_index + 1) % files_.size();
                file_offset = 0;
              }
            }
          }
          CHECK(cq->Send(std::move(request)) == kChannelStatusSuccess) << "channel error";
        }
      });
      workers_.emplace_back(std::move(worker));
    }
  }
  ~BatchReader() {
    for (auto& work : workers_) { work.Close(); }
  }

  void SubmitRequest(BatchReaderRequest&& request) {
    size_t worker_id = head_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
    workers_.at(worker_id).sq->Send(std::move(request));
  }
  void WaitCompleted(BatchReaderRequest* request) {
    size_t worker_id = tail_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
    workers_.at(worker_id).cq->Receive(request);
  }

 private:
  struct Worker {
    std::thread thread;
    std::unique_ptr<Channel<BatchReaderRequest>> sq;
    std::unique_ptr<Channel<BatchReaderRequest>> cq;
    void Close() {
      sq->Close();
      cq->Close();
      thread.join();
    }
  };
  std::atomic<size_t> head_;
  std::atomic<size_t> tail_;
  std::vector<Worker> workers_;
  std::vector<std::unique_ptr<embedding::PosixFile>> files_;
  std::vector<Block> blocks_;
  size_t block_size_bytes_;
  size_t num_workers_;
};

size_t GetNumShards(const Shape& hierarchy, const NdSbp& nd_sbp) {
  size_t num_shards = 1;
  FOR_RANGE(size_t, i, 0, nd_sbp.sbp_parallel_size()) {
    const auto& sbp_parallel = nd_sbp.sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      num_shards *= hierarchy.At(sbp_parallel.split_parallel().axis());
    }
  }
  return num_shards;
}

size_t GetShardIndex(const Shape& hierarchy, const NdSbp& nd_sbp, size_t rank) {
  using index_helper_t = NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE>;
  size_t ndim = hierarchy.NumAxes();
  CHECK_GT(ndim, 0) << "wrong hierarchy";
  CHECK_LE(ndim, SHAPE_MAX_AXIS_SIZE) << "wrong hierarchy";
  index_helper_t index_helper(hierarchy.dim_vec().data(), ndim);
  int64_t nd_index[SHAPE_MAX_AXIS_SIZE] = {0};
  index_helper.OffsetToNdIndex(rank, nd_index);
  size_t stride = 1;
  size_t index = 0;
  for (int i = ndim - 1; i >= 0; --i) {
    const auto& sbp_parallel = nd_sbp.sbp_parallel(i);
    if (sbp_parallel.has_split_parallel()) {
      index += nd_index[i] * stride;
      stride *= hierarchy.At(i);
    }
  }
  return index;
}

class BatchGenerator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchGenerator);
  BatchGenerator() = default;
  virtual ~BatchGenerator() = default;

  virtual void Next(size_t* blocks) = 0;
};

class SequentialBatchGenerator : public BatchGenerator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SequentialBatchGenerator);
  SequentialBatchGenerator(size_t shard_index, size_t num_shards, size_t num_batches,
                           size_t num_blocks_per_batch)
      : shard_index_(shard_index),
        num_shards_(num_shards),
        num_batches_(num_batches),
        num_blocks_per_batch_(num_blocks_per_batch),
        num_blocks_per_local_batch_(num_blocks_per_batch_ / num_shards_),
        next_batch_index_(0) {}
  ~SequentialBatchGenerator() override = default;

  void Next(size_t* blocks) override {
    const size_t batch_index = next_batch_index_;
    next_batch_index_ = (batch_index + 1) % num_batches_;
    for (size_t i = 0; i < num_blocks_per_local_batch_; ++i) {
      blocks[i] =
          batch_index * num_blocks_per_batch_ + shard_index_ * num_blocks_per_local_batch_ + i;
    }
  }

 private:
  size_t shard_index_;
  size_t num_shards_;
  size_t num_batches_;
  size_t num_blocks_per_batch_;
  size_t num_blocks_per_local_batch_;
  size_t next_batch_index_;
};

class RandomShuffleBatchGenerator : public BatchGenerator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RandomShuffleBatchGenerator);
  RandomShuffleBatchGenerator(size_t shard_index, size_t num_shards, size_t num_batches,
                              size_t num_blocks_per_batch, std::mt19937_64 generator)
      : shard_index_(shard_index),
        num_shards_(num_shards),
        num_batches_(num_batches),
        num_blocks_per_batch_(num_blocks_per_batch),
        num_blocks_per_local_batch_(num_blocks_per_batch_ / num_shards_),
        current_batch_pos_(0),
        generator_(generator) {
    batches_.resize(num_batches_);
    std::iota(batches_.begin(), batches_.end(), 0);
  }
  ~RandomShuffleBatchGenerator() override = default;

  void Next(size_t* blocks) override {
    size_t target_batch_pos =
        generator_() % (batches_.size() - current_batch_pos_) + current_batch_pos_;
    if (target_batch_pos != current_batch_pos_) {
      std::swap(batches_[target_batch_pos], batches_[current_batch_pos_]);
    }
    const size_t batch_index = batches_[current_batch_pos_];
    for (size_t i = 0; i < num_blocks_per_local_batch_; ++i) {
      blocks[i] =
          batch_index * num_blocks_per_batch_ + shard_index_ * num_blocks_per_local_batch_ + i;
    }
    current_batch_pos_ = (current_batch_pos_ + 1) % batches_.size();
    if (current_batch_pos_ == 0) { shard_index_ = (shard_index_ + 1) % num_shards_; }
  }

 private:
  size_t shard_index_;
  size_t num_shards_;
  size_t num_batches_;
  size_t num_blocks_per_batch_;
  size_t num_blocks_per_local_batch_;
  std::vector<size_t> batches_;
  size_t current_batch_pos_;
  std::mt19937_64 generator_;
};

class RawReaderKernelState final : public user_op::OpKernelState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RawReaderKernelState);
  explicit RawReaderKernelState(user_op::KernelInitContext* ctx) {
    const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex("out", 0);
    num_shards_ = GetNumShards(*ctx->parallel_desc().hierarchy(), nd_sbp);
    shard_index_ =
        GetShardIndex(*ctx->parallel_desc().hierarchy(), nd_sbp, ctx->parallel_ctx().parallel_id());
    batch_size_ = ctx->Attr<int64_t>("batch_size");
    CHECK_EQ(batch_size_ % num_shards_, 0) << "batch_size must be a multiple of num_shards";
    local_batch_size_ = batch_size_ / num_shards_;
    random_shuffle_ = ctx->Attr<bool>("random_shuffle");
    block_size_ = ctx->Attr<int64_t>("shuffle_block_size");
    if (block_size_ <= 0 || !random_shuffle_) { block_size_ = local_batch_size_; }
    CHECK_EQ(batch_size_ % block_size_, 0) << "batch_size must be a multiple of block_size";
    if (block_size_ > local_batch_size_) { block_size_ = local_batch_size_; }
    const std::vector<std::string>& filenames = ctx->Attr<std::vector<std::string>>("files");
    const Shape& instance_shape = ctx->Attr<Shape>("shape");
    const size_t elem_cnt = instance_shape.elem_cnt();
    CHECK_GT(elem_cnt, 0) << "instance size must be greater than 0";
    DimVector dim_vec;
    dim_vec.push_back(local_batch_size_);
    for (int64_t i = 0; i < instance_shape.NumAxes(); ++i) {
      dim_vec.push_back(instance_shape.At(i));
    }
    out_shape_ = Shape(dim_vec);
    data_type_ = ctx->Attr<DataType>("data_type");
    instance_size_ = ctx->Attr<Shape>("shape").elem_cnt() * GetSizeOfDataType(data_type_);
    CHECK_GT(batch_size_, 0) << "batch size must be greater than 0";
    size_t num_instances = 0;
    std::vector<std::unique_ptr<embedding::PosixFile>> files;
    int flags = O_RDONLY;
    if (ParseBooleanFromEnv("ONEFLOW_RAW_READER_FORCE_DIRECT_IO", false)) { flags |= O_DIRECT; }
    for (const auto& filename : filenames) {
      std::unique_ptr<embedding::PosixFile> file(new embedding::PosixFile(filename, flags, 0644));
      if (file->Size() == 0) { continue; }
      CHECK_EQ(file->Size() % instance_size_, 0) << "file_size must be a multiple of instance_size";
      num_instances += file->Size() / instance_size_;
      files.emplace_back(std::move(file));
    }
    if ((flags & O_DIRECT) != 0) {
      num_batches_ = num_instances / batch_size_;
    } else {
      num_batches_ = RoundUp(num_instances, batch_size_) / batch_size_;
    }
    block_size_bytes_ = block_size_ * instance_size_;
    local_batch_size_bytes_ = local_batch_size_ * instance_size_;
    num_blocks_per_local_batch_ = local_batch_size_ / block_size_;
    const size_t num_blocks = num_batches_ * (batch_size_ / block_size_);
    size_t file_index = 0;
    size_t offset_in_file = 0;
    std::vector<Block> blocks;
    for (size_t i = 0; i < num_blocks; ++i) {
      blocks.emplace_back(Block{file_index, offset_in_file});
      size_t remaining = block_size_bytes_;
      while (remaining != 0) {
        if (files[file_index]->Size() - offset_in_file >= remaining) {
          offset_in_file += remaining;
          if (offset_in_file == files[file_index]->Size()) { offset_in_file = 0; }
          remaining = 0;
        } else {
          remaining -= (files[file_index]->Size() - offset_in_file);
          offset_in_file = 0;
          file_index = (file_index + 1) % files.size();
        }
      }
    }
    if (random_shuffle_) {
      std::mt19937_64 generator;
      generator.seed(ctx->Attr<int64_t>("seed"));
      std::shuffle(blocks.begin(), blocks.end(), generator);
      batch_generator_.reset(new RandomShuffleBatchGenerator(
          shard_index_, num_shards_, num_batches_, batch_size_ / block_size_, generator));
    } else {
      batch_generator_.reset(new SequentialBatchGenerator(shard_index_, num_shards_, num_batches_,
                                                          batch_size_ / block_size_));
    }
    const size_t num_workers = ParseIntegerFromEnv("ONEFLOW_RAW_READER_NUM_WORKERS", 1);
    batch_reader_.reset(
        new BatchReader(std::move(files), std::move(blocks), block_size_bytes_, num_workers));
    prefetching_qd_ = ParseIntegerFromEnv("ONEFLOW_RAW_READER_PREFETCHING_QUEUE_DEPTH", 256);
    for (size_t i = 0; i < prefetching_qd_; ++i) {
      BatchReaderRequest request;
      request.blocks = std::make_shared<std::vector<size_t>>();
      if (ctx->device_type() == DeviceType::kCPU) {
        request.buffer = aligned_alloc(4096, RoundUp(local_batch_size_bytes_, 4096));  // NOLINT
      } else if (ctx->device_type() == DeviceType::kCUDA) {
#ifdef WITH_CUDA
        int dev = 0;
        OF_CUDA_CHECK(cudaGetDevice(&dev));
        OF_CUDA_CHECK(NumaAwareCudaMallocHost(dev, &request.buffer, local_batch_size_bytes_));
#else
        UNIMPLEMENTED();
#endif
      } else {
        UNIMPLEMENTED();
      }
      request.blocks = std::make_shared<std::vector<size_t>>(local_batch_size_ / block_size_);
      batch_generator_->Next(request.blocks->data());
      batch_reader_->SubmitRequest(std::move(request));
    }
    device_type_ = ctx->device_type();
  }

  ~RawReaderKernelState() {
    for (size_t i = 0; i < prefetching_qd_; ++i) {
      BatchReaderRequest request;
      batch_reader_->WaitCompleted(&request);
      if (device_type_ == DeviceType::kCPU) {
        free(request.buffer);  // NOLINT
      } else if (device_type_ == DeviceType::kCUDA) {
#ifdef WITH_CUDA
        OF_CUDA_CHECK(cudaFreeHost(request.buffer));
#else
        UNIMPLEMENTED();
#endif
      } else {
        UNIMPLEMENTED();
      }
    }
  }

  void Next(user_op::KernelComputeContext* ctx) {
    auto* tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(tensor->data_type(), data_type_) << "data type mismatch";
    CHECK(tensor->shape_view() == ShapeView(out_shape_)) << "shape mismatch";
    BatchReaderRequest request;
    batch_reader_->WaitCompleted(&request);
    if (ctx->stream()->device_type() == DeviceType::kCPU) {
      std::memcpy(tensor->mut_dptr<char>(), request.buffer, local_batch_size_bytes_);
    } else if (ctx->stream()->device_type() == DeviceType::kCUDA) {
#ifdef WITH_CUDA
      OF_CUDA_CHECK(cudaMemcpyAsync(tensor->mut_dptr<char>(), request.buffer,
                                    local_batch_size_bytes_, cudaMemcpyDefault,
                                    ctx->stream()->As<ep::CudaStream>()->cuda_stream()));
#else
      UNIMPLEMENTED();
#endif
    } else {
      UNIMPLEMENTED() << "only support CPU or CUDA";
    }
    CHECK_JUST(ctx->stream()->Sync());
    CHECK(request.blocks) << "blocks is NULL";
    CHECK_EQ(request.blocks->size(), num_blocks_per_local_batch_) << "blocks size mismatch";
    batch_generator_->Next(request.blocks->data());
    batch_reader_->SubmitRequest(std::move(request));
  }

 private:
  size_t instance_size_;
  size_t batch_size_;
  size_t local_batch_size_;
  size_t num_batches_;
  size_t num_shards_;
  size_t shard_index_;
  size_t block_size_;
  size_t block_size_bytes_;
  size_t num_blocks_per_local_batch_;
  size_t local_batch_size_bytes_;
  bool random_shuffle_;
  Shape out_shape_;
  DataType data_type_;
  std::unique_ptr<BatchGenerator> batch_generator_;
  std::unique_ptr<BatchReader> batch_reader_;
  DeviceType device_type_;
  size_t prefetching_qd_;
};

}  // namespace

class RawReaderKernel final : public user_op::OpKernel {
 public:
  RawReaderKernel() = default;
  ~RawReaderKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    std::shared_ptr<RawReaderKernelState> state(new RawReaderKernelState(ctx));
    return state;
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state,
               const user_op::OpKernelCache*) const override {
    auto* reader = CHECK_NOTNULL(dynamic_cast<RawReaderKernelState*>(state));
    reader->Next(ctx);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("raw_reader").SetCreateFn<RawReaderKernel>();

}  // namespace oneflow
