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
#include "oneflow/core/embedding/block_based_key_value_store.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/embedding/hash_functions.cuh"
#include "oneflow/core/embedding/file_handle.h"
#include <omp.h>
#include <robin_hood.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <liburing.h>

namespace oneflow {

namespace embedding {

namespace {

constexpr uint64_t NUM_BLOCKS_PER_CHUNK = 4 * 1024 * 1024;
constexpr uint32_t NUM_IO_THREADS = 4;
constexpr uint32_t IO_QD = 128;
constexpr uint32_t kChunkNameSuffixLength = 8;

struct Buffer {
  uint32_t index = 0;
  uint32_t offset_in_block = 0;
  uint32_t buffer_id = 0;
  struct iovec io_vec {};
};

template<typename Key>
struct alignas(2 * std::max(sizeof(Key), sizeof(uint64_t))) IndexEntry {
  Key key;
  uint64_t index;
};

std::string GetChunkName(uint64_t chunk_id) {
  const std::string chunk_name_wo_leading_zero = std::to_string(chunk_id);
  CHECK_LE(chunk_name_wo_leading_zero.size(), kChunkNameSuffixLength);
  return std::string(kChunkNameSuffixLength - chunk_name_wo_leading_zero.size(), '0')
         + chunk_name_wo_leading_zero;
}

void ListChunkFiles(const std::string& base, const std::string& prefix,
                    std::unordered_map<uint64_t, std::string>* chunks) {
  DIR* dir = opendir(base.c_str());
  PCHECK(dir != nullptr);
  struct dirent* ent = nullptr;
  while ((ent = readdir(dir)) != nullptr) {
    if (strlen(ent->d_name) != prefix.size() + kChunkNameSuffixLength) { continue; }
    if (strncmp(ent->d_name, prefix.c_str(), prefix.size()) != 0) { continue; }
    size_t pos = 0;
    const uint64_t chunk_id = std::stoull(ent->d_name + prefix.size(), &pos, 10);
    CHECK_EQ(pos, kChunkNameSuffixLength);
    CHECK(chunks->emplace(chunk_id, base + "/" + std::string(ent->d_name)).second);
  }
}

template<typename Key>
class KeyValueStoreImpl : public KeyValueStore {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeyValueStoreImpl);
  explicit KeyValueStoreImpl(const BlockBasedKeyValueStoreOptions& options)
      : device_index_(-1),
        value_length_(options.value_length),
        max_query_length_(options.max_query_length),
        counter_(1),
        block_size_(options.block_size),
        path_(options.path) {
    OF_CUDA_CHECK(cudaGetDevice(&device_index_));
    key_size_ = GetSizeOfDataType(options.key_type);
    value_size_ = GetSizeOfDataType(options.value_type) * value_length_;
    CHECK_GE(block_size_, value_size_);
    values_per_block_ = block_size_ / value_size_;
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(
        device_index_, reinterpret_cast<void**>(&host_query_keys_), key_size_ * max_query_length_));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_,
                                          reinterpret_cast<void**>(&host_query_values_),
                                          value_size_ * max_query_length_));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_, reinterpret_cast<void**>(&host_n_missing_),
                                          sizeof(uint32_t)));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_,
                                          reinterpret_cast<void**>(&host_missing_keys_),
                                          key_size_ * max_query_length_));
    OF_CUDA_CHECK(NumaAwareCudaMallocHost(device_index_,
                                          reinterpret_cast<void**>(&host_missing_indices_),
                                          sizeof(uint32_t) * max_query_length_));
    host_query_blocks_ =
        static_cast<uint8_t*>(aligned_alloc(block_size_, (max_query_length_ + values_per_block_ - 1)
                                                             / values_per_block_ * block_size_));
    io_rings_.resize(NUM_IO_THREADS);
    for (auto& ring : io_rings_) { PCHECK(io_uring_queue_init(IO_QD, &ring, 0) == 0); }
    thread_buffers_.resize(NUM_IO_THREADS);
    for (auto& buffers : thread_buffers_) {
      buffers.resize(IO_QD);
      for (uint32_t i = 0; i < IO_QD; ++i) {
        buffers.at(i).buffer_id = i;
        buffers.at(i).io_vec.iov_len = block_size_;
        buffers.at(i).io_vec.iov_base = aligned_alloc(block_size_, block_size_);
      }
    }
    std::unordered_map<uint64_t, std::string> chunks;
    ListChunkFiles(path_, "value-", &chunks);
    for (auto& chunk : chunks) {
      if (value_files_.size() <= chunk.first) { value_files_.resize(chunk.first + 1); }
      CHECK_EQ(value_files_.at(chunk.first).fd(), -1);
      value_files_.at(chunk.first) = FileHandle(chunk.second.c_str(), O_RDWR | O_DIRECT, 0644);
    }
    if (!value_files_.empty()) {
      counter_ = ((value_files_.size() - 1) * NUM_BLOCKS_PER_CHUNK
                  + value_files_.back().Size() / block_size_)
                 * values_per_block_;
      LoadIndexFile();
    }
  }
  ~KeyValueStoreImpl() {
    CudaCurrentDeviceGuard guard(device_index_);
    for (auto& file : value_files_) { file.Close(); }
    SaveIndexFile();
    OF_CUDA_CHECK(cudaFreeHost(host_query_keys_));
    OF_CUDA_CHECK(cudaFreeHost(host_query_values_));
    free(host_query_blocks_);
    for (auto& ring : io_rings_) { io_uring_queue_exit(&ring); }
    for (auto& buffers : thread_buffers_) {
      for (auto& buffer : buffers) { free(buffer.io_vec.iov_base); }
    }
  }

  void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
           uint32_t* n_missing, void* missing_keys, uint32_t* missing_indices,
           uint64_t* context) override;
  void Put(ep::Stream* stream, uint32_t num_keys, const void* keys, const void* values,
           uint64_t* context) override;

 private:
  std::string ValueFileName(uint64_t chunk_id) const;
  std::string IndexFileName() const;
  void SaveIndexFile();
  void LoadIndexFile();

  int device_index_;
  uint32_t value_length_;
  uint32_t max_query_length_;
  uint32_t key_size_;
  uint32_t value_size_;
  Key* host_query_keys_{};
  uint8_t* host_query_values_{};
  uint32_t* host_n_missing_{};
  Key* host_missing_keys_{};
  uint32_t* host_missing_indices_{};
  std::atomic<uint64_t> counter_;
  uint64_t block_size_;
  uint64_t values_per_block_;
  robin_hood::unordered_flat_map<Key, uint64_t> key2id_;
  std::vector<FileHandle> value_files_;
  std::string path_;
  uint8_t* host_query_blocks_;
  std::vector<io_uring> io_rings_;
  std::vector<std::vector<Buffer>> thread_buffers_;
};

template<typename Key>
void KeyValueStoreImpl<Key>::Get(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                 void* values, uint32_t* n_missing, void* missing_keys,
                                 uint32_t* missing_indices, uint64_t* context) {
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
  std::atomic<uint64_t> missing_counter(0);
#pragma omp parallel num_threads(NUM_IO_THREADS)
  {
    std::unique_ptr<uint8_t> block_data(
        static_cast<uint8_t*>(aligned_alloc(block_size_, block_size_)));

    const uint32_t thread_id = omp_get_thread_num();
    const uint32_t num_threads = omp_get_num_threads();
    auto& ring = io_rings_.at(thread_id);
    const uint32_t keys_per_thread = (num_keys + num_threads - 1) / num_threads;
    const uint64_t start_i = thread_id * keys_per_thread;
    const uint64_t end_i = std::min(start_i + keys_per_thread, static_cast<uint64_t>(num_keys));
    const uint64_t num_to_read = end_i - start_i;
    uint64_t num_read_done = 0;
    uint64_t num_ring_reading = 0;
    for (uint64_t i = start_i; i < end_i; ++i) {
      const Key key = host_query_keys_[i];
      auto it = key2id_.find(key);
      if (it == key2id_.end()) {
        const uint64_t missing_i = missing_counter.fetch_add(1, std::memory_order_relaxed);
        host_missing_indices_[missing_i] = i;
        host_missing_keys_[missing_i] = key;
      } else {
        Buffer* buffer = nullptr;
        if (num_ring_reading == IO_QD) {
          struct io_uring_cqe* cqe = nullptr;
          PCHECK(io_uring_wait_cqe(&ring, &cqe) == 0);
          buffer = static_cast<Buffer*>(io_uring_cqe_get_data(cqe));
          std::memcpy(static_cast<uint8_t*>(host_query_values_) + buffer->index * value_size_,
                      static_cast<uint8_t*>(buffer->io_vec.iov_base) + buffer->offset_in_block,
                      value_size_);
          io_uring_cqe_seen(&ring, cqe);
        }
        if (buffer == nullptr) {
          CHECK_LT(num_ring_reading, IO_QD);
          buffer = &thread_buffers_.at(thread_id).at(num_ring_reading);
          num_ring_reading += 1;
        }
        const uint64_t id = it->second;
        const uint64_t block_id = id / values_per_block_;
        const uint64_t id_in_block = id - block_id * values_per_block_;
        const uint64_t chunk_id = block_id / NUM_BLOCKS_PER_CHUNK;
        const uint64_t block_in_chunk = block_id - chunk_id * NUM_BLOCKS_PER_CHUNK;
        const uint64_t block_offset = block_in_chunk * block_size_;
        const uint64_t offset_in_chunk = block_offset + id_in_block * value_size_;
        FileHandle& file = value_files_.at(chunk_id);
        buffer->index = i;
        buffer->offset_in_block = id_in_block * value_size_;
        io_uring_sqe* sqe = CHECK_NOTNULL(io_uring_get_sqe(&ring));
        io_uring_prep_readv(sqe, file.fd(), &buffer->io_vec, 1, block_offset);
        io_uring_sqe_set_data(sqe, buffer);
        io_uring_submit(&ring);
      }
    }
    while (num_ring_reading != 0) {
      struct io_uring_cqe* cqe = nullptr;
      PCHECK(io_uring_wait_cqe(&ring, &cqe) == 0);
      Buffer* buffer = static_cast<Buffer*>(io_uring_cqe_get_data(cqe));
      std::memcpy(static_cast<uint8_t*>(host_query_values_) + buffer->index * value_size_,
                  static_cast<uint8_t*>(buffer->io_vec.iov_base) + buffer->offset_in_block,
                  value_size_);
      io_uring_cqe_seen(&ring, cqe);
      num_ring_reading -= 1;
    }
  }

  *host_n_missing_ = missing_counter.load();
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

template<typename Key>
void KeyValueStoreImpl<Key>::Put(ep::Stream* stream, uint32_t num_keys, const void* keys,
                                 const void* values, uint64_t* context) {
  CHECK_LE(num_keys, max_query_length_);
  if (num_keys == 0) { return; }
  auto cuda_stream = stream->As<ep::CudaStream>();
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_keys_, keys, key_size_ * num_keys, cudaMemcpyDefault,
                                cuda_stream->cuda_stream()));
  OF_CUDA_CHECK(cudaMemcpyAsync(host_query_values_, values, value_size_ * num_keys,
                                cudaMemcpyDefault, cuda_stream->cuda_stream()));
  CHECK_JUST(cuda_stream->Sync());
  const uint32_t num_blocks = (num_keys + values_per_block_ - 1) / values_per_block_;
  const uint32_t num_padded_keys = num_blocks * values_per_block_;
  const uint64_t start = counter_.fetch_add(num_padded_keys);
  CHECK_EQ(start % values_per_block_, 0);
  const uint64_t start_block_id = start / values_per_block_;
  for (uint64_t i = 0; i < num_keys; ++i) {
    const uint64_t block_id = i / values_per_block_;
    const uint64_t idx_in_block = i - block_id * values_per_block_;
    std::memcpy(host_query_blocks_ + block_id * block_size_ + idx_in_block * value_size_,
                host_query_values_ + i * value_size_, value_size_);
  }
  for (uint64_t i = 0; i < num_keys; ++i) { key2id_[host_query_keys_[i]] = start + i; }
  const uint64_t end_block_id = start_block_id + num_blocks - 1;
  const uint64_t start_chunk_id = start_block_id / NUM_BLOCKS_PER_CHUNK;
  const uint64_t end_chunk_id = end_block_id / NUM_BLOCKS_PER_CHUNK;
  value_files_.reserve(end_chunk_id + 1);
  for (uint64_t new_chunk_id = value_files_.size(); new_chunk_id <= end_chunk_id; ++new_chunk_id) {
    value_files_.emplace_back(ValueFileName(new_chunk_id).c_str(), O_CREAT | O_RDWR | O_DIRECT,
                              0644);
  }
  uint64_t write_blocks = 0;
  while (write_blocks < num_blocks) {
    const uint64_t batch_start_block_id = start_block_id + write_blocks;
    const uint64_t batch_chunk_id = batch_start_block_id / NUM_BLOCKS_PER_CHUNK;
    const uint64_t block_in_chunk = batch_start_block_id - batch_chunk_id * NUM_BLOCKS_PER_CHUNK;
    const uint64_t blocks_to_write =
        std::min(num_blocks - write_blocks,
                 (batch_chunk_id + 1) * NUM_BLOCKS_PER_CHUNK - batch_start_block_id);
    const uint64_t bytes_to_write = blocks_to_write * block_size_;
    const uint64_t offset_in_chunk = block_in_chunk * block_size_;
    FileHandle& file = value_files_.at(batch_chunk_id);
    CHECK_LE(file.Size(), offset_in_chunk);
    file.Truncate(offset_in_chunk + bytes_to_write);
    PCHECK(pwrite(file.fd(), host_query_blocks_ + write_blocks * block_size_, bytes_to_write,
                  offset_in_chunk)
           == bytes_to_write);
    write_blocks += blocks_to_write;
  }
}

template<typename Key>
std::string KeyValueStoreImpl<Key>::ValueFileName(uint64_t chunk_id) const {
  return path_ + "/value-" + GetChunkName(chunk_id);
}

template<typename Key>
std::string KeyValueStoreImpl<Key>::IndexFileName() const {
  return path_ + "/index";
}

template<typename Key>
void KeyValueStoreImpl<Key>::SaveIndexFile() {
  FileHandle index_file(IndexFileName().c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
  IndexEntry<Key> entry{};
  for (const auto& pair : key2id_) {
    entry.key = pair.first;
    entry.index = pair.second;
    PCHECK(write(index_file.fd(), &entry, sizeof(IndexEntry<Key>)) == sizeof(IndexEntry<Key>));
  }
}

template<typename Key>
void KeyValueStoreImpl<Key>::LoadIndexFile() {
  FileHandle index_file(IndexFileName().c_str(), O_RDONLY, 0644);
  IndexEntry<Key> entry{};
  const size_t size = index_file.Size();
  CHECK_EQ(size % sizeof(IndexEntry<Key>), 0);
  key2id_.reserve(size / sizeof(IndexEntry<Key>));
  for (size_t i = 0; i < size / sizeof(IndexEntry<Key>); ++i) {
    PCHECK(read(index_file.fd(), &entry, sizeof(IndexEntry<Key>)) == sizeof(IndexEntry<Key>));
    CHECK(key2id_.emplace(entry.key, entry.index).second);
  }
}

}  // namespace

std::unique_ptr<KeyValueStore> NewBlockBasedKeyValueStore(
    const BlockBasedKeyValueStoreOptions& options) {
  return std::unique_ptr<KeyValueStore>(new KeyValueStoreImpl<uint64_t>(options));
}

}  // namespace embedding

}  // namespace oneflow
