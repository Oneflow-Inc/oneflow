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
#include "oneflow/core/embedding/rocks_key_value_store.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/embedding/hash_functions.cuh"
#include <rocksdb/db.h>
#include <rocksdb/slice.h>
#include <rocksdb/options.h>
#include <rocksdb/slice_transform.h>
#include <rocksdb/table.h>
#include <omp.h>

using namespace ROCKSDB_NAMESPACE;

namespace oneflow {

namespace embedding {

namespace {

void RocksCheck(const Status& status) { CHECK(status.ok()) << status.ToString(); }

class KeyValueStoreImpl : public KeyValueStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeyValueStoreImpl);
  explicit KeyValueStoreImpl(const RocksKeyValueStoreOptions& options)
      : device_index_(-1),
        value_length_(options.value_length),
        max_query_length_(options.max_query_length) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    key_size_ = GetSizeOfDataType(options.key_type);
    value_size_ = GetSizeOfDataType(options.value_type) * value_length_;
    Options db_options;
    db_options.IncreaseParallelism();
    db_options.OptimizeLevelStyleCompaction();
    // create the DB if it's not already present
    db_options.create_if_missing = true;
    PlainTableOptions table_options;
    table_options.user_key_len = key_size_;
    table_options.encoding_type = rocksdb::kPrefix;
    db_options.table_factory.reset(NewPlainTableFactory(table_options));
    db_options.prefix_extractor.reset(NewFixedPrefixTransform(4));
    db_options.allow_mmap_reads = true;
    db_options.compression = rocksdb::kNoCompression;
    db_options.unordered_write = true;
    db_options.target_file_size_base = 128 << 20;

    DB* db = nullptr;
    RocksCheck(DB::Open(db_options, options.path, &db));
    db_.reset(db);

    OF_CUDA_CHECK(
        NumaAwareCudaMallocHost(device_index_, &host_query_keys_, key_size_ * max_query_length_));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_, &host_query_values_,
                                          value_size_ * max_query_length_));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_, reinterpret_cast<void**>(&host_n_missing_),
                                          sizeof(uint32_t)));
    OF_CUDA_CHECK(
        NumaAwareCudaMallocHost(device_index_, &host_missing_keys_, key_size_ * max_query_length_));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_,
                                          reinterpret_cast<void**>(&host_missing_indices_),
                                          sizeof(uint32_t) * max_query_length_));
    key_slices_.reserve(max_query_length_);
    value_slices_.reserve(max_query_length_);
    for (size_t i = 0; i < max_query_length_; ++i) {
      key_slices_.emplace_back(
          Slice(static_cast<const char*>(host_query_keys_) + i * key_size_, key_size_));
      value_slices_.emplace_back(
          Slice(static_cast<const char*>(host_query_values_) + i * value_size_, value_size_));
    }
    result_slices_.resize(max_query_length_);
    statuses_.resize(max_query_length_);
  }
  ~KeyValueStoreImpl() {
    CudaCurrentDeviceGuard guard(device_index_);
    OF_CUDA_CHECK(cudaFreeHost(host_query_keys_));
    OF_CUDA_CHECK(cudaFreeHost(host_query_values_));
  }

  void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
           uint32_t* n_missing, void* missing_keys, uint32_t* missing_indices,
           uint64_t* context) override;
  void Put(ep::Stream* stream, uint32_t num_keys, const void* keys, const void* values,
           uint64_t* context) override;

 private:
  int device_index_;
  uint32_t value_length_;
  uint32_t max_query_length_;
  uint32_t key_size_;
  uint32_t value_size_;
  std::unique_ptr<DB> db_;
  void* host_query_keys_{};
  void* host_query_values_{};
  uint32_t* host_n_missing_{};
  void* host_missing_keys_{};
  uint32_t* host_missing_indices_{};
  std::vector<Slice> key_slices_;
  std::vector<Slice> value_slices_;
  std::vector<PinnableSlice> result_slices_;
  std::vector<Status> statuses_;
};

void KeyValueStoreImpl::Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
                            uint32_t* n_missing, void* missing_keys, uint32_t* missing_indices,
                            uint64_t* context) {
  auto cuda_stream = stream->As<ep::CudaStream>();
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_values_, values, num_keys * value_size_,
                                cudaMemcpyDefault, cuda_stream->cuda_stream()));
  CHECK_LE(num_keys, max_query_length_);
  if (num_keys == 0) {
    OF_CUDA_CHECK(cudaMemsetAsync(n_missing, 0, sizeof(uint32_t),
                                  stream->As<ep::CudaStream>()->cuda_stream()));
    return;
  }
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_keys_, keys, key_size_ * num_keys, cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  ReadOptions read_options;
  read_options.verify_checksums = false;
#pragma omp parallel num_threads(4)
  {
    const uint32_t thread_id = omp_get_thread_num();
    const uint32_t num_threads = omp_get_num_threads();
    const uint32_t keys_per_thread = (num_keys + num_threads - 1) / num_threads;
    const uint32_t start = thread_id * keys_per_thread;
    const uint32_t end = std::min(start + keys_per_thread, num_keys);
    db_->MultiGet(read_options, db_->DefaultColumnFamily(), end - start, key_slices_.data() + start,
                  result_slices_.data() + start, statuses_.data() + start);
  }
  *host_n_missing_ = 0;
  for (uint32_t i = 0; i < num_keys; i++) {
    if (statuses_[i].ok()) {
      std::memcpy(static_cast<char*>(host_query_values_) + i * value_size_,
                  result_slices_[i].data(), value_size_);
    } else if (statuses_[i].IsNotFound()) {
      std::memcpy(static_cast<char*>(host_missing_keys_) + (*host_n_missing_) * key_size_,
                  static_cast<char*>(host_query_keys_) + i * key_size_, key_size_);
      host_missing_indices_[*host_n_missing_] = i;
      *host_n_missing_ += 1;
    } else {
      LOG(FATAL) << statuses_[i].ToString();
    }
  }
  OF_CUDA_CHECK(cudaMemcpyAsync(values, host_query_values_, num_keys * value_size_,
                                cudaMemcpyDefault, cuda_stream->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(n_missing, host_n_missing_, sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(missing_keys, host_missing_keys_, (*host_n_missing_) * key_size_,
                                cudaMemcpyDefault, cuda_stream->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(missing_indices, host_missing_indices_,
                                (*host_n_missing_) * sizeof(uint32_t), cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
}

void KeyValueStoreImpl::Put(ep::Stream* stream, uint32_t num_keys, const void* keys,
                            const void* values, uint64_t* context) {
  CHECK_LE(num_keys, max_query_length_);
  if (num_keys == 0) { return; }
  auto cuda_stream = stream->As<ep::CudaStream>();
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_keys_, keys, key_size_ * num_keys, cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_values_, values, value_size_ * num_keys,
                                cudaMemcpyDefault, cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  WriteOptions write_options;
  write_options.disableWAL = true;
#pragma omp parallel num_threads(4)
  {
    const uint32_t thread_id = omp_get_thread_num();
    const uint32_t num_threads = omp_get_num_threads();
    const uint32_t keys_per_thread = (num_keys + num_threads - 1) / num_threads;
    const uint32_t start = thread_id * keys_per_thread;
    const uint32_t end = std::min(start + keys_per_thread, num_keys);
    WriteBatch wb;
    for (size_t i = start; i < end; ++i) { RocksCheck(wb.Put(key_slices_[i], value_slices_[i])); }
    RocksCheck(db_->Write(write_options, &wb));
  }
}

}  // namespace

std::unique_ptr<KeyValueStore> NewRocksKeyValueStore(const RocksKeyValueStoreOptions& options) {
  return std::unique_ptr<KeyValueStore>(new KeyValueStoreImpl(options));
}

}  // namespace embedding

}  // namespace oneflow
