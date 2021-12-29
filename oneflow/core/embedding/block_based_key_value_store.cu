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
#include <omp.h>
#include <robin_hood.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

namespace oneflow {

namespace embedding {

namespace {

constexpr uint64_t NUM_BLOCKS_PER_CHUNK = 4 * 1024 * 1024;

class FileGuard final {
 public:
  OF_DISALLOW_COPY(FileGuard);
  FileGuard() : fd_(-1) {}
  FileGuard(const std::string& filename, int flag, size_t trunc) : fd_(-1) {
    fd_ = open(filename.c_str(), flag, 0644);
    PCHECK(fd_ != -1);
    if (trunc != 0) { PCHECK(ftruncate(fd_, trunc) == 0); }
  }
  FileGuard& operator=(FileGuard&& other) noexcept {
    fd_ = other.fd_;
    other.fd_ = -1;
    return *this;
  }
  ~FileGuard() { Close(); }

  int fd() { return fd_; }

  void Close() {
    if (fd_ != -1) {
      PCHECK(close(fd_) == 0);
      fd_ = -1;
    }
  }

  size_t Size() {
    if (fd_ == -1) { return 0; }
    struct stat sb {};
    PCHECK(fstat(fd_, &sb) == 0);
    return sb.st_size;
  }

 private:
  int fd_;
};

enum class ChunkType {
  kInvalid,
  kMapped,
  kFile,
};

struct Chunk {
  Chunk() {}
  Chunk(Chunk&& other) noexcept { *this = std::move(other); }
  Chunk& operator=(Chunk&& other) noexcept {
    chunk_type = other.chunk_type;
    other.chunk_type = ChunkType::kInvalid;
    ptr = other.ptr;
    other.ptr = nullptr;
    file = std::move(other.file);
    size = other.size;
    other.size = 0;
    return *this;
  }
  size_t size = 0;
  void* ptr = nullptr;
  ChunkType chunk_type = ChunkType::kInvalid;
  FileGuard file;
};

Chunk OpenMappedChunk(const std::string& filename, size_t size) {
  Chunk chunk;
  chunk.chunk_type = ChunkType::kMapped;
  FileGuard file(filename, O_RDWR | O_CREAT, size);
  chunk.file = std::move(file);
  chunk.ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, chunk.file.fd(), 0);
  chunk.size = size;
  PCHECK(chunk.ptr != MAP_FAILED);
  return chunk;
}

void CloseMappedChunk(Chunk* chunk) {
  CHECK(chunk->chunk_type == ChunkType::kMapped);
  PCHECK(munmap(chunk->ptr, chunk->size) == 0);
  chunk->file.Close();
}

Chunk OpenFileChunk(const std::string& filename) {
  Chunk chunk;
  chunk.chunk_type = ChunkType::kFile;
  FileGuard file(filename, O_RDONLY | O_DIRECT, 0);
  chunk.file = std::move(file);
  chunk.size = file.Size();
  return chunk;
}

void CloseFileChunk(Chunk* chunk) {
  CHECK(chunk->chunk_type == ChunkType::kFile);
  chunk->file.Close();
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
  }
  ~KeyValueStoreImpl() {
    CudaCurrentDeviceGuard guard(device_index_);
    for (auto& chunk : chunks_) {
      if (chunk.chunk_type == ChunkType::kMapped) { CloseMappedChunk(&chunk); }
    }
    OF_CUDA_CHECK(cudaFreeHost(host_query_keys_));
    OF_CUDA_CHECK(cudaFreeHost(host_query_values_));
  }

  void Get(ep::Stream* stream, uint32_t num_keys, const void* keys, void* values,
           uint32_t* n_missing, void* missing_keys, uint32_t* missing_indices,
           uint64_t* context) override;
  void Put(ep::Stream* stream, uint32_t num_keys, const void* keys, const void* values,
           uint64_t* context) override;

 private:
  std::string ChunkName(uint64_t chunk_id);

  int device_index_;
  uint32_t value_length_;
  uint32_t max_query_length_;
  uint32_t key_size_;
  uint32_t value_size_;
  void* host_query_keys_{};
  void* host_query_values_{};
  uint32_t* host_n_missing_{};
  void* host_missing_keys_{};
  uint32_t* host_missing_indices_{};
  std::atomic<uint64_t> counter_;
  uint64_t block_size_;
  uint64_t values_per_block_;
  robin_hood::unordered_flat_map<Key, uint64_t> key2id_;
  std::vector<Chunk> chunks_;
  std::string path_;
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
#pragma omp parallel num_threads(4)
  {
    std::unique_ptr<uint8_t> block_data(
        static_cast<uint8_t*>(aligned_alloc(block_size_, block_size_)));
    const uint32_t thread_id = omp_get_thread_num();
    const uint32_t num_threads = omp_get_num_threads();
    const uint32_t keys_per_thread = (num_keys + num_threads - 1) / num_threads;
    const uint64_t start_i = thread_id * keys_per_thread;
    const uint64_t end_i = std::min(start_i + keys_per_thread, static_cast<uint64_t>(num_keys));
    for (uint64_t i = start_i; i < end_i; ++i) {
      const Key key = static_cast<const Key*>(host_query_keys_)[i];
      auto it = key2id_.find(key);
      if (it == key2id_.end()) {
        const uint64_t missing_i = missing_counter.fetch_add(1, std::memory_order_relaxed);
        host_missing_indices_[missing_i] = i;
        static_cast<Key*>(host_missing_keys_)[missing_i] = key;
      } else {
        const uint64_t id = it->second;
        const uint64_t block_id = id / values_per_block_;
        const uint64_t id_in_block = id - block_id * values_per_block_;
        const uint64_t chunk_id = block_id / NUM_BLOCKS_PER_CHUNK;
        const uint64_t block_in_chunk = block_id - chunk_id * NUM_BLOCKS_PER_CHUNK;
        const uint64_t block_offset = block_in_chunk * block_size_;
        const uint64_t offset_in_chunk = block_offset + id_in_block * value_size_;
        Chunk& chunk = chunks_.at(chunk_id);
        if (chunk.chunk_type == ChunkType::kMapped) {
          std::memcpy(static_cast<uint8_t*>(host_query_values_) + i * value_size_,
                      static_cast<uint8_t*>(chunk.ptr) + offset_in_chunk, value_size_);
        } else if (chunk.chunk_type == ChunkType::kFile) {
          PCHECK(pread(chunk.file.fd(), block_data.get(), block_size_, block_offset)
                 == block_size_);
          std::memcpy(static_cast<uint8_t*>(host_query_values_) + i * value_size_,
                      block_data.get() + id_in_block * value_size_, value_size_);
        } else {
          UNIMPLEMENTED();
        }
      }
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
  const Key* keys_ptr = static_cast<const Key*>(host_query_keys_);
  const uint64_t start = counter_.fetch_add(num_keys);
  const uint64_t end = start + num_keys;
  for (uint64_t i = 0; i < num_keys; ++i) { key2id_[keys_ptr[i]] = start + i; }
  const uint64_t min_block_id = start / values_per_block_;
  const uint64_t max_block_id = (end - 1) / values_per_block_;
  const uint64_t min_chunk_id = min_block_id / NUM_BLOCKS_PER_CHUNK;
  const uint64_t max_chunk_id = max_block_id / NUM_BLOCKS_PER_CHUNK;
  chunks_.reserve(max_chunk_id + 1);
  for (uint64_t new_chunk_id = chunks_.size(); new_chunk_id <= max_chunk_id; ++new_chunk_id) {
    chunks_.emplace_back(
        OpenMappedChunk(ChunkName(new_chunk_id), NUM_BLOCKS_PER_CHUNK * block_size_));
  }
#pragma omp parallel num_threads(4)
  {
    const uint32_t thread_id = omp_get_thread_num();
    const uint32_t num_threads = omp_get_num_threads();
    const uint32_t keys_per_thread = (num_keys + num_threads - 1) / num_threads;
    const uint64_t start_i = thread_id * keys_per_thread;
    const uint64_t end_i = std::min(start_i + keys_per_thread, static_cast<uint64_t>(num_keys));
    for (uint64_t i = start_i; i < end_i; ++i) {
      const uint64_t block_id = (start + i) / values_per_block_;
      const uint64_t id_in_block = (start + i) - block_id * values_per_block_;
      const uint64_t chunk_id = block_id / NUM_BLOCKS_PER_CHUNK;
      const uint64_t block_in_chunk = block_id - chunk_id * NUM_BLOCKS_PER_CHUNK;
      const uint64_t block_offset = block_in_chunk * block_size_;
      const uint64_t offset_in_chunk = block_offset + id_in_block * value_size_;
      std::memcpy(static_cast<uint8_t*>(chunks_.at(chunk_id).ptr) + offset_in_chunk,
                  static_cast<const uint8_t*>(host_query_values_) + i * value_size_, value_size_);
    }
  }
  const uint64_t next_chunk_id = end / values_per_block_ / NUM_BLOCKS_PER_CHUNK;
  const uint64_t max_close_chunk_id =
      next_chunk_id != max_chunk_id ? max_chunk_id + 1 : max_chunk_id;
  for (uint64_t chunk_id = min_chunk_id; chunk_id < max_close_chunk_id; ++chunk_id) {
    CloseMappedChunk(&chunks_.at(chunk_id));
    chunks_.at(chunk_id) = std::move(OpenFileChunk(ChunkName(chunk_id)));
  }
}

template<typename Key>
std::string KeyValueStoreImpl<Key>::ChunkName(uint64_t chunk_id) {
  return path_ + "/" + std::to_string(chunk_id);
}

}  // namespace

std::unique_ptr<KeyValueStore> NewBlockBasedKeyValueStore(
    const BlockBasedKeyValueStoreOptions& options) {
  return std::unique_ptr<KeyValueStore>(new KeyValueStoreImpl<uint64_t>(options));
}

}  // namespace embedding

}  // namespace oneflow
