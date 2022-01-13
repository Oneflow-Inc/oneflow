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
#include "oneflow/core/embedding/fixed_table.h"
#include "oneflow/core/embedding/posix_file.h"
#include <omp.h>
#include <robin_hood.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <sys/syscall.h>
#include <linux/aio_abi.h>
#include <unistd.h>
#ifdef WITH_LIBURING
#include <liburing.h>
#endif  // WITH_LIBURING

namespace oneflow {

namespace embedding {

namespace {

constexpr uint32_t kNumReadThreads = 4;
constexpr uint32_t kRingQueueDepth = 128;
constexpr uint32_t kRingSubmitBatch = 32;
constexpr uint32_t kAioQueueDepth = 128;
constexpr uint32_t kChunkNameSuffixLength = 8;
constexpr char const* kSnapshotFilenamePrefix = "";
constexpr char const* kValueFilenamePrefix = "value-";
constexpr char const* kLockFilename = "FIXED_TABLE";
constexpr char const* kKeySizeFilename = "KEY_SIZE";
constexpr char const* kValueSizeFilename = "VALUE_SIZE";
constexpr char const* kPhysicalBlockSizeFilename = "PHYSICAL_BLOCK_SIZE";
constexpr char const* kNumBlocksPerChunkFilename = "NUM_BLOCKS_PER_CHUNK";

void InitOrCheckMetaValue(const std::string& pathname, int64_t expected, bool init) {
  bool exists = PosixFile::FileExists(pathname.c_str());
  if (init) {
    CHECK(!exists);
    std::ofstream ofs(pathname);
    ofs << expected << std::endl;
  } else {
    CHECK(exists);
    std::ifstream ifs(pathname);
    int64_t value = 0;
    ifs >> value;
    if (value != expected) { LOG(FATAL) << "Check failed: " << pathname; }
  }
}

template<typename Key>
struct IndexEntry {
  static constexpr size_t align_size = std::max(sizeof(Key), sizeof(uint64_t));
  union {
    typename std::aligned_storage<align_size, align_size>::type pad;
    Key data;
  } key;
  union {
    typename std::aligned_storage<align_size, align_size>::type pad;
    uint64_t data;
  } index;
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
  PCHECK(closedir(dir) == 0);
}

template<typename Key>
class KeyIteratorImpl : public FixedTable::KeyIterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KeyIteratorImpl);
  explicit KeyIteratorImpl(const robin_hood::unordered_flat_map<Key, uint64_t>& map)
      : pos_(map.begin()), end_(map.end()) {}
  ~KeyIteratorImpl() override = default;

  void Next(uint32_t num_keys, uint32_t* return_keys, void* keys) override {
    uint32_t count = 0;
    while (count < num_keys && pos_ != end_) {
      static_cast<Key*>(keys)[count] = pos_->first;
      count++;
      pos_++;
    }
    *return_keys = count;
  }

 private:
  typename robin_hood::unordered_flat_map<Key, uint64_t>::const_iterator pos_;
  typename robin_hood::unordered_flat_map<Key, uint64_t>::const_iterator end_;
};

#ifdef WITH_LIBURING

class RingEngine final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RingEngine);
  RingEngine() : ring_{}, pending_submit_(0), num_readings_(0) {
    PCHECK(io_uring_queue_init(kRingQueueDepth, &ring_, 0) == 0);
  }
  ~RingEngine() {
    WaitUntilDone();
    io_uring_queue_exit(&ring_);
  }

  void AsyncPread(int fd, void* buf, size_t count, off_t offset) {
    if (num_readings_ == kRingQueueDepth) {
      struct io_uring_cqe* cqe = nullptr;
      PCHECK(io_uring_wait_cqe(&ring_, &cqe) == 0);
      CHECK_GE(cqe->res, 0);
      io_uring_cqe_seen(&ring_, cqe);
    } else {
      num_readings_ += 1;
    }
    io_uring_sqe* sqe = CHECK_NOTNULL(io_uring_get_sqe(&ring_));
    io_uring_prep_read(sqe, fd, buf, count, offset);
    pending_submit_ += 1;
    if (pending_submit_ == kRingSubmitBatch) {
      PCHECK(io_uring_submit(&ring_) == pending_submit_);
      pending_submit_ = 0;
    }
  }

  void WaitUntilDone() {
    if (pending_submit_ > 0) {
      PCHECK(io_uring_submit(&ring_) == pending_submit_);
      pending_submit_ = 0;
    }
    while (num_readings_ != 0) {
      struct io_uring_cqe* cqe = nullptr;
      PCHECK(io_uring_wait_cqe(&ring_, &cqe) == 0);
      CHECK_GE(cqe->res, 0);
      io_uring_cqe_seen(&ring_, cqe);
      num_readings_ -= 1;
    }
  }

 private:
  io_uring ring_;
  uint32_t pending_submit_;
  uint32_t num_readings_;
};

#endif  // WITH_LIBURING

class AioEngine final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AioEngine);
  AioEngine() : ctx_{}, num_readings_(0) {
    PCHECK(syscall(__NR_io_setup, kAioQueueDepth, &ctx_) >= 0);
    cbs_.resize(kAioQueueDepth);
    cbs_ptr_.resize(kAioQueueDepth);
    for (uint32_t i = 0; i < kAioQueueDepth; ++i) { cbs_ptr_[i] = &cbs_[i]; }
    events_.resize(kAioQueueDepth);
  }
  ~AioEngine() {
    WaitUntilDone();
    PCHECK(syscall(__NR_io_destroy, ctx_) >= 0);
  }

  void AsyncPread(int fd, void* buf, size_t count, off_t offset) {
    if (num_readings_ == kAioQueueDepth) { WaitUntilDone(); }
    struct iocb* cb = &cbs_.at(num_readings_);
    cb->aio_fildes = fd;
    cb->aio_lio_opcode = IOCB_CMD_PREAD;
    cb->aio_reqprio = 0;
    cb->aio_buf = reinterpret_cast<uintptr_t>(buf);
    cb->aio_nbytes = count;
    cb->aio_offset = offset;
    const long nr = 1;
    PCHECK(syscall(__NR_io_submit, ctx_, nr, &cbs_ptr_.at(num_readings_)) >= 0);
    num_readings_ += 1;
  }

  void WaitUntilDone() {
    if (num_readings_ != 0) {
      PCHECK(syscall(__NR_io_getevents, ctx_, num_readings_, num_readings_, events_.data(), nullptr)
             >= 0);
      for (long i = 0; i < num_readings_; ++i) { CHECK_GT(events_.at(i).res, 0); }
      num_readings_ = 0;
    }
  }

 private:
  aio_context_t ctx_;
  long num_readings_;
  std::vector<struct iocb> cbs_;
  std::vector<struct iocb*> cbs_ptr_;
  std::vector<struct io_event> events_;
};

template<typename Key, typename Engine>
class FixedTableImpl : public FixedTable {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FixedTableImpl);
  explicit FixedTableImpl(const FixedTableOptions& options)
      : path_(options.path),
        key_size_(options.key_size),
        value_size_(options.value_size),
        num_blocks_per_chunk_(options.num_blocks_per_chunk) {
    PosixFile::RecursiveCreateDirectory(options.path, 0755);
    const std::string lock_filename = options.path + "/" + kLockFilename;
    const bool init = !PosixFile::FileExists(lock_filename.c_str());
    lock_ = PosixFileLockGuard(PosixFile(lock_filename, O_CREAT | O_RDWR, 0644));
    InitOrCheckMetaValue(options.path + "/" + kKeySizeFilename, key_size_, init);
    InitOrCheckMetaValue(options.path + "/" + kValueSizeFilename, value_size_, init);
    InitOrCheckMetaValue(options.path + "/" + kPhysicalBlockSizeFilename,
                         options.physical_block_size, init);
    InitOrCheckMetaValue(options.path + "/" + kNumBlocksPerChunkFilename, num_blocks_per_chunk_,
                         init);
    logical_block_size_ = options.physical_block_size >= value_size_
                              ? options.physical_block_size
                              : RoundUp(value_size_, options.physical_block_size);
    values_per_block_ = logical_block_size_ / value_size_;
    engines_.resize(kNumReadThreads);
    for (uint32_t tid = 0; tid < kNumReadThreads; ++tid) { engines_.at(tid).reset(new Engine); }
    std::unordered_map<uint64_t, std::string> chunks;
    ListChunkFiles(path_, kValueFilenamePrefix, &chunks);
    for (auto& chunk : chunks) {
      if (value_files_.size() <= chunk.first) { value_files_.resize(chunk.first + 1); }
      CHECK_EQ(value_files_.at(chunk.first).fd(), -1);
      PosixFile value_file(chunk.second.c_str(), O_RDWR | O_DIRECT, 0644);
      value_files_.at(chunk.first) = std::move(value_file);
    }
    if (!value_files_.empty()) {
      physical_table_size_ = ((value_files_.size() - 1) * num_blocks_per_chunk_
                              + value_files_.back().Size() / logical_block_size_)
                             * values_per_block_;
    } else {
      physical_table_size_ = 0;
    }
  }
  ~FixedTableImpl() override {
    for (auto& file : value_files_) { file.Close(); }
  }

  uint32_t KeySize() const override { return key_size_; }

  uint32_t ValueSize() const override { return value_size_; }

  uint16_t LogicalBlockSize() const override;
  void Test(uint32_t num_keys, const void* keys, uint32_t* n_missing,
            uint32_t* missing_indices) override;
  void GetBlocks(uint32_t num_keys, const void* keys, void* blocks, uint16_t* offsets) override;
  void Get(uint32_t num_keys, const void* keys, void* values, uint32_t* n_missing,
           uint32_t* missing_indices) override;
  void PutBlocks(uint32_t num_keys, const void* keys, const void* blocks) override;
  void Put(uint32_t num_keys, const void* keys, const void* values) override;
  void WithKeyIterator(const std::function<void(KeyIterator* iter)>& fn) override;
  bool SnapshotExists(const std::string& name) override;
  void LoadSnapshot(const std::string& name) override;
  void SaveSnapshot(const std::string& name) override;

 private:
  std::string ValueFileName(uint64_t chunk_id) const;
  std::string SnapshotFilename(const std::string& name) const;
  void LoadSnapshotImpl(const std::string& name);
  void SaveSnapshotImpl(const std::string& name);

  std::string path_;
  uint32_t key_size_;
  uint32_t value_size_;
  uint64_t num_blocks_per_chunk_;
  uint16_t logical_block_size_;

  uint32_t values_per_block_;

  std::vector<std::unique_ptr<Engine>> engines_;

  std::vector<uint16_t> offsets_buffer_;
  std::vector<char> blocks_buffer_;

  std::recursive_mutex mutex_;
  uint64_t physical_table_size_;
  robin_hood::unordered_flat_map<Key, uint64_t> row_id_mapping_;
  std::vector<PosixFile> value_files_;
  PosixFileLockGuard lock_;
};

template<typename Key, typename Engine>
uint16_t FixedTableImpl<Key, Engine>::LogicalBlockSize() const {
  return logical_block_size_;
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::Test(uint32_t num_keys, const void* keys, uint32_t* n_missing,
                                       uint32_t* missing_indices) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::atomic<uint32_t> atomic_n_missing(0);
#pragma omp parallel num_threads(kNumReadThreads)
  {
    const uint32_t thread_id = omp_get_thread_num();
    const uint32_t num_threads = omp_get_num_threads();
    const uint32_t keys_per_thread = (num_keys + num_threads - 1) / num_threads;
    const uint32_t start_i = thread_id * keys_per_thread;
    const uint32_t end_i = std::min(start_i + keys_per_thread, num_keys);
    for (uint64_t i = start_i; i < end_i; ++i) {
      const Key key = static_cast<const Key*>(keys)[i];
      auto it = row_id_mapping_.find(key);
      if (it == row_id_mapping_.end()) {
        const uint32_t index = atomic_n_missing.fetch_add(1, std::memory_order_relaxed);
        missing_indices[index] = i;
      }
    }
  }
  *n_missing = atomic_n_missing.load(std::memory_order_relaxed);
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::GetBlocks(uint32_t num_keys, const void* keys, void* blocks,
                                            uint16_t* offsets) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
#pragma omp parallel num_threads(kNumReadThreads)
  {
    const uint32_t thread_id = omp_get_thread_num();
    const uint32_t num_threads = omp_get_num_threads();
    auto& engine = *engines_.at(thread_id);
    const uint32_t keys_per_thread = (num_keys + num_threads - 1) / num_threads;
    const uint32_t start_i = thread_id * keys_per_thread;
    const uint32_t end_i = std::min(start_i + keys_per_thread, num_keys);
    for (uint64_t i = start_i; i < end_i; ++i) {
      const Key key = static_cast<const Key*>(keys)[i];
      auto it = row_id_mapping_.find(key);
      if (it == row_id_mapping_.end()) {
        offsets[i] = logical_block_size_;
      } else {
        const uint64_t id = it->second;
        const uint64_t block_id = id / values_per_block_;
        const uint16_t id_in_block = id - block_id * values_per_block_;
        const uint64_t offset_in_block = id_in_block * value_size_;
        const uint64_t chunk_id = block_id / num_blocks_per_chunk_;
        const uint64_t block_in_chunk = block_id - chunk_id * num_blocks_per_chunk_;
        const uint64_t block_offset = block_in_chunk * logical_block_size_;
        PosixFile& file = value_files_.at(chunk_id);
        offsets[i] = offset_in_block;
        engine.AsyncPread(file.fd(), static_cast<char*>(blocks) + i * logical_block_size_,
                          logical_block_size_, block_offset);
      }
    }
    engine.WaitUntilDone();
  }
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::Get(uint32_t num_keys, const void* keys, void* values,
                                      uint32_t* n_missing, uint32_t* missing_indices) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  offsets_buffer_.resize(num_keys);
  void* blocks_ptr = nullptr;
  if (value_size_ == logical_block_size_) {
    blocks_ptr = values;
  } else {
    blocks_buffer_.resize((num_keys + 1) * logical_block_size_);
    blocks_ptr = blocks_buffer_.data()
                 + (logical_block_size_
                    - reinterpret_cast<uintptr_t>(blocks_buffer_.data()) % logical_block_size_);
  }
  GetBlocks(num_keys, keys, blocks_ptr, offsets_buffer_.data());
  uint32_t missing_count = 0;
  for (uint32_t i = 0; i < num_keys; ++i) {
    if (offsets_buffer_.at(i) == logical_block_size_) {
      missing_indices[missing_count] = i;
      missing_count += 1;
    } else {
      if (value_size_ != logical_block_size_) {
        std::memcpy(
            static_cast<char*>(values) + i * value_size_,
            static_cast<char*>(blocks_ptr) + (i * logical_block_size_) + offsets_buffer_.at(i),
            value_size_);
      }
    }
  }
  *n_missing = missing_count;
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::PutBlocks(uint32_t num_keys, const void* keys,
                                            const void* blocks) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const uint32_t num_blocks = (num_keys + values_per_block_ - 1) / values_per_block_;
  const uint32_t num_padded_keys = num_blocks * values_per_block_;
  const uint64_t start = physical_table_size_;
  physical_table_size_ += num_padded_keys;
  CHECK_EQ(start % values_per_block_, 0);
  const uint64_t start_block_id = start / values_per_block_;
  for (uint64_t i = 0; i < num_keys; ++i) {
    row_id_mapping_[static_cast<const Key*>(keys)[i]] = start + i;
  }
  uint64_t write_blocks = 0;
  while (write_blocks < num_blocks) {
    const uint64_t batch_start_block_id = start_block_id + write_blocks;
    const uint64_t batch_chunk_id = batch_start_block_id / num_blocks_per_chunk_;
    if (batch_chunk_id == value_files_.size()) {
      value_files_.emplace_back(ValueFileName(batch_chunk_id).c_str(), O_CREAT | O_RDWR | O_DIRECT,
                                0644);
    } else {
      CHECK_LE(batch_chunk_id, value_files_.size());
    }
    PosixFile& file = value_files_.at(batch_chunk_id);
    const uint64_t block_in_chunk = batch_start_block_id - batch_chunk_id * num_blocks_per_chunk_;
    const uint64_t blocks_to_write =
        std::min(num_blocks - write_blocks,
                 (batch_chunk_id + 1) * num_blocks_per_chunk_ - batch_start_block_id);
    const uint64_t bytes_to_write = blocks_to_write * logical_block_size_;
    const uint64_t offset_in_chunk = block_in_chunk * logical_block_size_;
    CHECK_LE(file.Size(), offset_in_chunk);
    file.Truncate(offset_in_chunk + bytes_to_write);
    PCHECK(pwrite(file.fd(), static_cast<const char*>(blocks) + write_blocks * logical_block_size_,
                  bytes_to_write, offset_in_chunk)
           == bytes_to_write);
    write_blocks += blocks_to_write;
  }
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::Put(uint32_t num_keys, const void* keys, const void* values) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const void* blocks_ptr = nullptr;
  if (value_size_ == logical_block_size_) {
    blocks_ptr = values;
  } else {
    const uint32_t num_blocks = (num_keys + values_per_block_ - 1) / values_per_block_;
    blocks_buffer_.resize((num_blocks + 1) * logical_block_size_);
    void* blocks_buffer_ptr =
        blocks_buffer_.data()
        + (logical_block_size_
           - reinterpret_cast<uintptr_t>(blocks_buffer_.data()) % logical_block_size_);
    for (uint32_t i = 0; i < num_keys; i += values_per_block_) {
      const uint32_t block_id = i / values_per_block_;
      const uint32_t copy_size =
          (num_keys - i) < values_per_block_ ? (num_keys - i) * value_size_ : logical_block_size_;
      std::memcpy(static_cast<char*>(blocks_buffer_ptr) + block_id * logical_block_size_,
                  static_cast<const char*>(values) + i * value_size_, copy_size);
    }
    blocks_ptr = blocks_buffer_ptr;
  }
  PutBlocks(num_keys, keys, blocks_ptr);
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::WithKeyIterator(
    const std::function<void(KeyIterator* iter)>& fn) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  KeyIteratorImpl<Key> iter(row_id_mapping_);
  fn(&iter);
}

template<typename Key, typename Engine>
std::string FixedTableImpl<Key, Engine>::ValueFileName(uint64_t chunk_id) const {
  return path_ + "/" + kValueFilenamePrefix + GetChunkName(chunk_id);
}

template<typename Key, typename Engine>
std::string FixedTableImpl<Key, Engine>::SnapshotFilename(const std::string& name) const {
  return path_ + "/" + kSnapshotFilenamePrefix + name;
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::LoadSnapshotImpl(const std::string& name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  row_id_mapping_.clear();
  PosixFile snapshot_file(SnapshotFilename(name).c_str(), O_RDONLY, 0644);
  using Entry = IndexEntry<Key>;
  const size_t size = snapshot_file.Size();
  if (size == 0) { return; }
  PosixMappedFile mapped_file(std::move(snapshot_file), size, PROT_READ);
  const Entry* entries = static_cast<Entry*>(mapped_file.ptr());
  CHECK_EQ(size % sizeof(Entry), 0);
  size_t n_entries = size / sizeof(Entry);
  row_id_mapping_.reserve(n_entries);
  for (size_t i = 0; i < n_entries; ++i) {
    CHECK(row_id_mapping_.emplace(entries[i].key.data, entries[i].index.data).second);
  }
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::SaveSnapshotImpl(const std::string& name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  PosixFile snapshot_file(SnapshotFilename(name).c_str(), O_CREAT | O_RDWR, 0644);
  if (row_id_mapping_.empty()) { return; }
  using Entry = IndexEntry<Key>;
  const size_t total_size = sizeof(Entry) * row_id_mapping_.size();
  snapshot_file.Truncate(total_size);
  PosixMappedFile mapped_file(std::move(snapshot_file), total_size, PROT_READ | PROT_WRITE);
  Entry* entries = static_cast<Entry*>(mapped_file.ptr());
  size_t count = 0;
  for (const auto& pair : row_id_mapping_) {
    entries[count].key.data = pair.first;
    entries[count].index.data = pair.second;
    count += 1;
  }
}

template<typename Key, typename Engine>
bool FixedTableImpl<Key, Engine>::SnapshotExists(const std::string& name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return PosixFile::FileExists(SnapshotFilename(name).c_str());
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::LoadSnapshot(const std::string& name) {
  LoadSnapshotImpl(name);
}

template<typename Key, typename Engine>
void FixedTableImpl<Key, Engine>::SaveSnapshot(const std::string& name) {
  SaveSnapshotImpl(name);
}

}  // namespace

std::unique_ptr<FixedTable> NewFixedTable(const FixedTableOptions& options) {
  CHECK(!options.path.empty());
  CHECK_GT(options.value_size, 0);
  CHECK_GT(options.num_blocks_per_chunk, 0);
  CHECK_GT(options.physical_block_size, 0);
  CHECK_GT(options.key_size, 0);
#ifdef WITH_LIBURING
  using Engine = RingEngine;
#else
  using Engine = AioEngine;
#endif  // WITH_LIBURING

  if (options.key_size == 4) {
    return std::unique_ptr<FixedTable>(new FixedTableImpl<uint32_t, Engine>(options));
  } else if (options.key_size == 8) {
    return std::unique_ptr<FixedTable>(new FixedTableImpl<uint64_t, Engine>(options));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

}  // namespace embedding

}  // namespace oneflow
