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
#include "oneflow/core/embedding/persistent_table.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/embedding/hash_functions.cuh"

#ifdef __linux__

#include "oneflow/core/common/channel.h"
#include "oneflow/core/embedding/posix_file.h"
#include "oneflow/core/common/blocking_counter.h"
#include <robin_hood.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dirent.h>
#include <sys/syscall.h>
#include <linux/aio_abi.h>
#include <unistd.h>

#endif  // __linux__

namespace oneflow {

namespace embedding {

#ifdef __linux__

namespace {

constexpr uint32_t kDefaultNumWorkerThreads = 4;
constexpr uint32_t kRingQueueDepth = 128;
constexpr uint32_t kRingSubmitBatch = 32;
constexpr uint32_t kAioQueueDepth = 128;
constexpr uint32_t kChunkNameSuffixLength = 12;
constexpr char const* kKeyFileNamePrefix = "key-";
constexpr char const* kIndexFileNamePrefix = "index-";
constexpr char const* kValueFileNamePrefix = "value-";
constexpr char const* kLockFileName = "LOCK";
constexpr char const* kKeySizeFileName = "KEY_SIZE";
constexpr char const* kValueSizeFileName = "VALUE_SIZE";
constexpr char const* kPhysicalBlockSizeFileName = "PHYSICAL_BLOCK_SIZE";
constexpr char const* kNumLogicalBlocksPerChunkFileName = "NUM_LOGICAL_BLOCKS_PER_CHUNK";
constexpr char const* kKeysDirName = "keys";
constexpr char const* kValuesDirName = "values";
constexpr char const* kSnapshotsDirName = "snapshots";
constexpr char const* kSnapshotListFileName = "LIST";
constexpr size_t kParallelForStride = 256;

template<typename T>
T* BytesOffset(T* ptr, size_t bytes) {
  return reinterpret_cast<T*>(
      const_cast<unsigned char*>((reinterpret_cast<const unsigned char*>(ptr) + bytes)));
}

void MemcpyOffset(void* dst, size_t dst_off, const void* src, size_t src_off, size_t n) {
  std::memcpy(BytesOffset(dst, dst_off), BytesOffset(src, src_off), n);
}

void InitOrCheckMetaValue(const std::string& pathname, int64_t expected, bool init) {
  bool exists = PosixFile::FileExists(pathname);
  if (init) {
    CHECK(!exists) << pathname;
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

std::string GetChunkName(uint64_t chunk_id) {
  const std::string chunk_name_wo_leading_zero = std::to_string(chunk_id);
  CHECK_LE(chunk_name_wo_leading_zero.size(), kChunkNameSuffixLength);
  return std::string(kChunkNameSuffixLength - chunk_name_wo_leading_zero.size(), '0')
         + chunk_name_wo_leading_zero;
}

uint64_t GetChunkId(const std::string& chunk_name) {
  size_t pos = 0;
  const uint64_t chunk_id = std::stoull(chunk_name, &pos, 10);
  CHECK_EQ(pos, kChunkNameSuffixLength);
  return chunk_id;
}

uint64_t GetChunkId(const std::string& filename, const std::string& prefix) {
  CHECK_EQ(filename.compare(0, prefix.size(), prefix), 0);
  return GetChunkId(filename.substr(prefix.size()));
}

void ListChunkFiles(const std::string& base, const std::string& prefix,
                    std::unordered_map<uint64_t, std::string>* chunks) {
  DIR* dir = opendir(base.c_str());
  PCHECK(dir != nullptr);
  struct dirent* ent = nullptr;
  while ((ent = readdir(dir)) != nullptr) {
    if (strlen(ent->d_name) != prefix.size() + kChunkNameSuffixLength) { continue; }
    if (strncmp(ent->d_name, prefix.c_str(), prefix.size()) != 0) { continue; }
    const uint64_t chunk_id = GetChunkId(ent->d_name + prefix.size());
    CHECK(chunks->emplace(chunk_id, PosixFile::JoinPath(base, ent->d_name)).second);
  }
  PCHECK(closedir(dir) == 0);
}

uint32_t GetLogicalBlockSize(uint32_t physical_block_size, uint32_t value_size) {
  return physical_block_size >= value_size ? physical_block_size
                                           : RoundUp(value_size, physical_block_size);
}

class AlignedBuffer final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AlignedBuffer);
  explicit AlignedBuffer(size_t alignment) : alignment_(alignment), size_(0) {}
  ~AlignedBuffer() = default;

  void Resize(size_t new_size) {
    if (new_size > size_) {
      ptr_.reset(static_cast<char*>(aligned_alloc(alignment_, new_size)));
      size_ = new_size;
    }
  }

  void* ptr() { return ptr_.get(); }

 private:
  size_t alignment_;
  size_t size_;
  std::unique_ptr<char> ptr_;
};

template<typename Key>
class ChunkIteratorImpl : public PersistentTable::Iterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ChunkIteratorImpl);
  ChunkIteratorImpl(uint32_t value_size, uint32_t logical_block_size, uint32_t num_values_per_block,
                    uint64_t num_values_per_chunk, uint64_t chunk_id, uint64_t n,
                    const Key* chunk_keys, const uint64_t* chunk_indices, const void* chunk_values)
      : pos_(0),
        value_size_(value_size),
        logical_block_size_(logical_block_size),
        num_values_per_block_(num_values_per_block),
        num_values_per_chunk_(num_values_per_chunk),
        n_(n),
        chunk_keys_(chunk_keys),
        chunk_indices_(chunk_indices),
        chunk_values_(chunk_values),
        chunk_index_offset_(chunk_id * num_values_per_chunk_) {}
  ~ChunkIteratorImpl() override = default;

  void Next(uint32_t num_keys, uint32_t* return_keys, void* keys, void* values) override {
    uint32_t count = 0;
    while (count < num_keys && pos_ != n_) {
      const uint64_t index_in_chunk = chunk_indices_[pos_] - chunk_index_offset_;
      static_cast<Key*>(keys)[count] = chunk_keys_[index_in_chunk];
      const uint64_t block_in_chunk = index_in_chunk / num_values_per_block_;
      const uint32_t index_in_block = index_in_chunk - block_in_chunk * num_values_per_block_;
      const uint32_t value_offset =
          block_in_chunk * logical_block_size_ + index_in_block * value_size_;
      std::memcpy(static_cast<char*>(values) + count * value_size_,
                  static_cast<const char*>(chunk_values_) + value_offset, value_size_);
      count++;
      pos_++;
    }
    *return_keys = count;
  }

  void Reset() override { pos_ = 0; }

 private:
  uint64_t pos_;
  uint32_t value_size_;
  uint32_t logical_block_size_;
  uint32_t num_values_per_block_;
  uint64_t num_values_per_chunk_;
  uint64_t n_;
  const Key* chunk_keys_;
  const uint64_t* chunk_indices_;
  const void* chunk_values_;
  uint64_t chunk_index_offset_;
};

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

constexpr size_t kCacheLineSize = 64;

template<typename Engine>
using IoTask = std::function<void(Engine* engine)>;

template<typename Engine>
using ForRange = std::function<void(Engine* engine, size_t start, size_t end)>;

template<typename Engine>
class Worker final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Worker);
  Worker() { thread_ = std::thread(&Worker<Engine>::PullTask, this); }
  ~Worker() {
    Shutdown();
    thread_.join();
  }

  void Schedule(IoTask<Engine> task) { tasks_.Send(std::move(task)); }

  void Shutdown() { tasks_.Close(); }

 private:
  void PullTask() {
    while (true) {
      IoTask<Engine> task;
      const ChannelStatus status = tasks_.Receive(&task);
      if (status == ChannelStatus::kChannelStatusErrorClosed) { break; }
      CHECK_EQ(status, ChannelStatus::kChannelStatusSuccess);
      task(&engine_);
    }
  }
  Channel<IoTask<Engine>> tasks_;
  Engine engine_;
  std::thread thread_;
};

template<typename Key, typename Engine>
class SnapshotIteratorImpl;

template<typename Key, typename Engine>
class PersistentTableImpl : public PersistentTable {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistentTableImpl);
  explicit PersistentTableImpl(const PersistentTableOptions& options);
  ~PersistentTableImpl() override;

  uint32_t KeySize() const override { return key_size_; }

  uint32_t ValueSize() const override { return value_size_; }

  uint32_t LogicalBlockSize() const override;
  void GetBlocks(uint32_t num_keys, const void* keys, void* blocks, uint32_t* offsets) override;
  void Get(uint32_t num_keys, const void* keys, void* values, uint32_t* n_missing,
           uint32_t* missing_indices) override;
  void PutBlocks(uint32_t num_keys, const void* keys, const void* blocks) override;
  void Put(uint32_t num_keys, const void* keys, const void* values) override;
  bool SnapshotExists(const std::string& name) override;
  void LoadSnapshot(const std::string& name) override;
  void LoadSnapshot(const std::string& name,
                    const std::function<void(Iterator* iter)>& Hook) override;
  void SaveSnapshot(const std::string& name) override;
  Iterator* ReadSnapshot(const std::string& name) override;

 private:
  friend class SnapshotIteratorImpl<Key, Engine>;
  std::string KeyFilePath(uint64_t chunk_id) const;
  std::string ValueFilePath(uint64_t chunk_id) const;
  std::string IndexFilePath(const std::string& name, uint64_t chunk_id) const;
  std::string SnapshotDirPath(const std::string& name) const;
  std::string SnapshotListFilePath(const std::string& name) const;
  void LoadSnapshotImpl(const std::string& name);
  void SaveSnapshotImpl(const std::string& name);
  void ParallelFor(size_t total, const ForRange<Engine>& for_range);

  std::string root_dir_;
  std::string keys_dir_;
  std::string values_dir_;
  std::string snapshots_dir_;
  uint32_t key_size_;
  uint32_t value_size_;
  uint64_t num_logical_blocks_per_chunk_;
  uint64_t num_values_per_chunk_;
  uint32_t num_values_per_block_;
  uint32_t physical_block_size_;
  uint32_t logical_block_size_;

  std::vector<std::unique_ptr<Worker<Engine>>> workers_;

  std::vector<uint32_t> offsets_buffer_;
  AlignedBuffer blocks_buffer_;

  std::recursive_mutex mutex_;
  uint64_t physical_table_size_;
  robin_hood::unordered_flat_map<Key, uint64_t> row_id_mapping_;
  std::vector<PosixFile> value_files_;
  PosixFile writable_key_file_;
  uint64_t writable_key_file_chunk_id_;
  PosixFileLockGuard lock_;
  bool read_only_;
};

template<typename Key, typename Engine>
PersistentTableImpl<Key, Engine>::PersistentTableImpl(const PersistentTableOptions& options)
    : root_dir_(options.path),
      key_size_(options.key_size),
      value_size_(options.value_size),
      physical_block_size_(options.physical_block_size),
      logical_block_size_(GetLogicalBlockSize(options.physical_block_size, value_size_)),
      blocks_buffer_(options.physical_block_size),
      writable_key_file_chunk_id_(-1),
      read_only_(options.read_only) {
  const uint64_t capacity_hint = ParseIntegerFromEnv(
      "ONEFLOW_ONE_EMBEDDING_PERSISTENT_TABLE_CAPACITY_HINT", options.capacity_hint);
  if (capacity_hint > 0) { row_id_mapping_.reserve(capacity_hint); }
  PosixFile::RecursiveCreateDirectory(options.path, 0755);
  const std::string lock_filename = PosixFile::JoinPath(options.path, kLockFileName);
  const bool init = !PosixFile::FileExists(lock_filename);
  if (read_only_) {
    CHECK(!init) << "The table must be initialized in read only mode";
  } else {
    lock_ = PosixFileLockGuard(PosixFile(lock_filename, O_CREAT | O_RDWR, 0644));
  }
  const uint64_t target_chunk_size = options.target_chunk_size_mb * 1024 * 1024;
  CHECK_GE(target_chunk_size, logical_block_size_);
  num_logical_blocks_per_chunk_ = target_chunk_size / logical_block_size_,
  num_values_per_block_ = logical_block_size_ / value_size_;
  num_values_per_chunk_ = num_values_per_block_ * num_logical_blocks_per_chunk_;
  InitOrCheckMetaValue(PosixFile::JoinPath(options.path, kKeySizeFileName), key_size_, init);
  InitOrCheckMetaValue(PosixFile::JoinPath(options.path, kValueSizeFileName), value_size_, init);
  InitOrCheckMetaValue(PosixFile::JoinPath(options.path, kPhysicalBlockSizeFileName),
                       options.physical_block_size, init);
  InitOrCheckMetaValue(PosixFile::JoinPath(options.path, kNumLogicalBlocksPerChunkFileName),
                       num_logical_blocks_per_chunk_, init);
  keys_dir_ = PosixFile::JoinPath(options.path, kKeysDirName);
  values_dir_ = PosixFile::JoinPath(options.path, kValuesDirName);
  snapshots_dir_ = PosixFile::JoinPath(options.path, kSnapshotsDirName);
  if (init) {
    PosixFile::RecursiveCreateDirectory(keys_dir_, 0755);
    PosixFile::RecursiveCreateDirectory(values_dir_, 0755);
  }
  const uint32_t num_workers = ParseIntegerFromEnv(
      "ONEFLOW_ONE_EMBEDDING_PERSISTENT_TABLE_NUM_WORKERS", kDefaultNumWorkerThreads);
  workers_.resize(num_workers);
  for (uint32_t tid = 0; tid < workers_.size(); ++tid) {
    workers_.at(tid).reset(new Worker<Engine>);
  }
  std::unordered_map<uint64_t, std::string> chunks;
  ListChunkFiles(values_dir_, kValueFileNamePrefix, &chunks);
  for (auto& chunk : chunks) {
    if (value_files_.size() <= chunk.first) { value_files_.resize(chunk.first + 1); }
    CHECK_EQ(value_files_.at(chunk.first).fd(), -1);
    const int flags = read_only_ ? (O_RDONLY | O_DIRECT) : (O_RDWR | O_DIRECT);
    PosixFile value_file(chunk.second, flags, 0644);
    value_files_.at(chunk.first) = std::move(value_file);
  }
  if (!value_files_.empty()) {
    physical_table_size_ = ((value_files_.size() - 1) * num_logical_blocks_per_chunk_
                            + value_files_.back().Size() / logical_block_size_)
                           * num_values_per_block_;
  } else {
    physical_table_size_ = 0;
  }
}

template<typename Key, typename Engine>
PersistentTableImpl<Key, Engine>::~PersistentTableImpl() {
  for (uint32_t tid = 0; tid < workers_.size(); ++tid) { workers_.at(tid)->Shutdown(); }
}

template<typename Key, typename Engine>
uint32_t PersistentTableImpl<Key, Engine>::LogicalBlockSize() const {
  return logical_block_size_;
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::GetBlocks(uint32_t num_keys, const void* keys, void* blocks,
                                                 uint32_t* offsets) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  ParallelFor(num_keys, [&](Engine* engine, size_t start, size_t end) {
    for (uint64_t i = start; i < end; ++i) {
      const Key key = static_cast<const Key*>(keys)[i];
      auto it = row_id_mapping_.find(key);
      if (it == row_id_mapping_.end()) {
        offsets[i] = logical_block_size_;
      } else {
        const uint64_t id = it->second;
        const uint64_t block_id = id / num_values_per_block_;
        const uint32_t id_in_block = id - block_id * num_values_per_block_;
        const uint32_t offset_in_block = id_in_block * value_size_;
        const uint64_t chunk_id = block_id / num_logical_blocks_per_chunk_;
        const uint64_t block_in_chunk = block_id - chunk_id * num_logical_blocks_per_chunk_;
        const uint64_t block_offset = block_in_chunk * logical_block_size_;
        PosixFile& file = value_files_.at(chunk_id);
        offsets[i] = offset_in_block;
        engine->AsyncPread(file.fd(), BytesOffset(blocks, i * logical_block_size_),
                           logical_block_size_, block_offset);
      }
    }
  });
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::Get(uint32_t num_keys, const void* keys, void* values,
                                           uint32_t* n_missing, uint32_t* missing_indices) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  offsets_buffer_.resize(num_keys);
  void* blocks_ptr = nullptr;
  if (value_size_ == logical_block_size_
      && reinterpret_cast<uintptr_t>(values) % physical_block_size_ == 0) {
    blocks_ptr = values;
  } else {
    blocks_buffer_.Resize(num_keys * logical_block_size_);
    blocks_ptr = blocks_buffer_.ptr();
  }
  GetBlocks(num_keys, keys, blocks_ptr, offsets_buffer_.data());
  uint32_t missing_count = 0;
  for (uint32_t i = 0; i < num_keys; ++i) {
    if (offsets_buffer_.at(i) == logical_block_size_) {
      missing_indices[missing_count] = i;
      missing_count += 1;
    } else {
      if (value_size_ != logical_block_size_) {
        MemcpyOffset(values, i * value_size_, blocks_ptr,
                     (i * logical_block_size_) + offsets_buffer_[i], value_size_);
      }
    }
  }
  *n_missing = missing_count;
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::PutBlocks(uint32_t num_keys, const void* keys,
                                                 const void* blocks) {
  CHECK(!read_only_);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const uint32_t num_blocks = RoundUp(num_keys, num_values_per_block_) / num_values_per_block_;
  const uint32_t num_padded_keys = num_blocks * num_values_per_block_;
  const uint64_t start_index = physical_table_size_;
  physical_table_size_ += num_padded_keys;
  CHECK_EQ(start_index % num_values_per_block_, 0);
  const uint64_t start_block_id = start_index / num_values_per_block_;
  uint64_t written_blocks = 0;
  const uint64_t block_keys_size = num_values_per_block_ * sizeof(Key);
  BlockingCounter bc(1);
  workers_.at(0)->Schedule([&](Engine*) {
    while (written_blocks < num_blocks) {
      const uint64_t batch_start_block_id = start_block_id + written_blocks;
      const uint64_t batch_chunk_id = batch_start_block_id / num_logical_blocks_per_chunk_;
      if (batch_chunk_id == value_files_.size()) {
        value_files_.emplace_back(ValueFilePath(batch_chunk_id), O_CREAT | O_RDWR | O_DIRECT, 0644);
      } else {
        CHECK_LE(batch_chunk_id, value_files_.size());
      }
      if ((!writable_key_file_.IsOpen()) || writable_key_file_chunk_id_ != batch_chunk_id) {
        writable_key_file_ = PosixFile(KeyFilePath(batch_chunk_id), O_CREAT | O_RDWR, 0644);
      }
      PosixFile& value_file = value_files_.at(batch_chunk_id);
      const uint64_t block_id_in_chunk =
          batch_start_block_id - batch_chunk_id * num_logical_blocks_per_chunk_;
      const uint64_t blocks_to_write =
          std::min(num_blocks - written_blocks,
                   (batch_chunk_id + 1) * num_logical_blocks_per_chunk_ - batch_start_block_id);
      const uint64_t values_bytes = blocks_to_write * logical_block_size_;
      const uint64_t values_offset_in_file = block_id_in_chunk * logical_block_size_;
      CHECK_LE(value_file.Size(), values_offset_in_file);
      value_file.Truncate(values_offset_in_file + values_bytes);
      PCHECK(pwrite(value_file.fd(), BytesOffset(blocks, written_blocks * logical_block_size_),
                    values_bytes, values_offset_in_file)
             == values_bytes);
      const uint64_t keys_offset_in_file = block_id_in_chunk * block_keys_size;
      writable_key_file_.Truncate(keys_offset_in_file + blocks_to_write * block_keys_size);
      const uint64_t keys_bytes = std::min(num_keys - written_blocks * num_values_per_block_,
                                           blocks_to_write * num_values_per_block_)
                                  * sizeof(Key);
      PCHECK(pwrite(writable_key_file_.fd(), BytesOffset(keys, written_blocks * block_keys_size),
                    keys_bytes, keys_offset_in_file)
             == keys_bytes);
      written_blocks += blocks_to_write;
    }
    bc.Decrease();
  });
  for (uint64_t i = 0; i < num_keys; ++i) {
    row_id_mapping_[static_cast<const Key*>(keys)[i]] = start_index + i;
  }
  bc.WaitForeverUntilCntEqualZero();
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::Put(uint32_t num_keys, const void* keys,
                                           const void* values) {
  CHECK(!read_only_);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const void* blocks_ptr = nullptr;
  if (value_size_ == logical_block_size_
      && reinterpret_cast<uintptr_t>(values) % physical_block_size_ == 0) {
    blocks_ptr = values;
  } else {
    const uint32_t num_blocks = RoundUp(num_keys, num_values_per_block_);
    blocks_buffer_.Resize(num_blocks * logical_block_size_);
    for (uint32_t i = 0; i < num_keys; i += num_values_per_block_) {
      const uint32_t block_id = i / num_values_per_block_;
      const uint32_t copy_size = (num_keys - i) < num_values_per_block_
                                     ? (num_keys - i) * value_size_
                                     : logical_block_size_;
      MemcpyOffset(blocks_buffer_.ptr(), block_id * logical_block_size_, values, i * value_size_,
                   copy_size);
    }
    blocks_ptr = blocks_buffer_.ptr();
  }
  PutBlocks(num_keys, keys, blocks_ptr);
}

template<typename Key, typename Engine>
std::string PersistentTableImpl<Key, Engine>::KeyFilePath(uint64_t chunk_id) const {
  return PosixFile::JoinPath(keys_dir_, kKeyFileNamePrefix + GetChunkName(chunk_id));
}

template<typename Key, typename Engine>
std::string PersistentTableImpl<Key, Engine>::ValueFilePath(uint64_t chunk_id) const {
  return PosixFile::JoinPath(values_dir_, kValueFileNamePrefix + GetChunkName(chunk_id));
}

template<typename Key, typename Engine>
std::string PersistentTableImpl<Key, Engine>::IndexFilePath(const std::string& name,
                                                            uint64_t chunk_id) const {
  return PosixFile::JoinPath(SnapshotDirPath(name), kIndexFileNamePrefix + GetChunkName(chunk_id));
}

template<typename Key, typename Engine>
std::string PersistentTableImpl<Key, Engine>::SnapshotDirPath(const std::string& name) const {
  return PosixFile::JoinPath(snapshots_dir_, name);
}

template<typename Key, typename Engine>
std::string PersistentTableImpl<Key, Engine>::SnapshotListFilePath(const std::string& name) const {
  return PosixFile::JoinPath(SnapshotDirPath(name), kSnapshotListFileName);
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::LoadSnapshotImpl(const std::string& name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const std::string snapshot_base = SnapshotDirPath(name);
  const std::string snapshot_list = SnapshotListFilePath(name);
  row_id_mapping_.clear();
  std::ifstream list_if(snapshot_list);
  std::string index_filename;
  while (std::getline(list_if, index_filename)) {
    const uint64_t chunk_id = GetChunkId(index_filename, kIndexFileNamePrefix);
    PosixFile index_file(PosixFile::JoinPath(snapshot_base, index_filename), O_RDONLY, 0644);
    const size_t index_file_size = index_file.Size();
    CHECK_EQ(index_file_size % sizeof(uint64_t), 0);
    if (index_file_size == 0) { return; }
    const size_t n_entries = index_file_size / sizeof(uint64_t);
    PosixMappedFile mapped_index(std::move(index_file), index_file_size, PROT_READ);
    PosixFile key_file(KeyFilePath(chunk_id), O_RDONLY, 0644);
    PosixMappedFile mapped_key(std::move(key_file), key_file.Size(), PROT_READ);
    const uint64_t* indices = static_cast<const uint64_t*>(mapped_index.ptr());
    const Key* keys = static_cast<const Key*>(mapped_key.ptr());
    const uint64_t chunk_start_index = chunk_id * num_values_per_chunk_;
    row_id_mapping_.reserve(row_id_mapping_.size() + n_entries);
    for (size_t i = 0; i < n_entries; ++i) {
      CHECK(row_id_mapping_.emplace(keys[indices[i] - chunk_start_index], indices[i]).second);
    }
  }
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::SaveSnapshotImpl(const std::string& name) {
  CHECK(!read_only_);
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  PosixFile::RecursiveCreateDirectory(SnapshotDirPath(name), 0755);
  std::ofstream list_ofs(SnapshotListFilePath(name));
  if (row_id_mapping_.empty()) { return; }
  std::vector<PosixMappedFile> index_files(value_files_.size());
  std::vector<uint64_t> counters(value_files_.size());
  const uint64_t max_index_file_size = num_values_per_chunk_ * sizeof(uint64_t);
  for (const auto& pair : row_id_mapping_) {
    const uint64_t chunk_id = pair.second / num_values_per_chunk_;
    CHECK(chunk_id < value_files_.size());
    if (index_files[chunk_id].ptr() == nullptr) {
      PosixFile snapshot_file(IndexFilePath(name, chunk_id), O_CREAT | O_RDWR, 0644);
      snapshot_file.Truncate(max_index_file_size);
      index_files[chunk_id] =
          PosixMappedFile(std::move(snapshot_file), max_index_file_size, PROT_READ | PROT_WRITE);
    }
    uint64_t* indices = static_cast<uint64_t*>(index_files[chunk_id].ptr());
    uint64_t& count = counters[chunk_id];
    CHECK_LT(count, num_values_per_chunk_);
    indices[count] = pair.second;
    count += 1;
  }
  for (size_t i = 0; i < value_files_.size(); ++i) {
    const uint64_t count = counters[i];
    if (count > 0) {
      index_files[i].file().Truncate(count * sizeof(uint64_t));
      list_ofs << kIndexFileNamePrefix + GetChunkName(i) << std::endl;
    } else {
      CHECK(index_files[i].ptr() == nullptr);
    }
  }
}

template<typename Key, typename Engine>
bool PersistentTableImpl<Key, Engine>::SnapshotExists(const std::string& name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return PosixFile::FileExists(SnapshotListFilePath(name));
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::LoadSnapshot(const std::string& name) {
  LoadSnapshotImpl(name);
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::LoadSnapshot(
    const std::string& name, const std::function<void(Iterator* iter)>& Hook) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  int mmap_flags = MAP_SHARED;
  if (ParseBooleanFromEnv("ONEFLOW_ONE_EMBEDDING_PERSISTENT_TABLE_SNAPSHOT_LOAD_MAP_POPULATE",
                          true)) {
    mmap_flags |= MAP_POPULATE;
  }
  const std::string snapshot_base = SnapshotDirPath(name);
  const std::string snapshot_list = SnapshotListFilePath(name);
  row_id_mapping_.clear();
  std::ifstream list_if(snapshot_list);
  std::string index_filename;
  while (std::getline(list_if, index_filename)) {
    const uint64_t chunk_id = GetChunkId(index_filename, kIndexFileNamePrefix);
    PosixFile index_file(PosixFile::JoinPath(snapshot_base, index_filename), O_RDONLY, 0644);
    const size_t index_file_size = index_file.Size();
    CHECK_EQ(index_file_size % sizeof(uint64_t), 0);
    if (index_file_size == 0) { return; }
    const size_t n_entries = index_file_size / sizeof(uint64_t);
    PosixMappedFile mapped_index(std::move(index_file), index_file_size, PROT_READ, mmap_flags);
    PosixFile key_file(KeyFilePath(chunk_id), O_RDONLY, 0644);
    PosixMappedFile mapped_key(std::move(key_file), key_file.Size(), PROT_READ, mmap_flags);
    const uint64_t* indices = static_cast<const uint64_t*>(mapped_index.ptr());
    const Key* keys = static_cast<const Key*>(mapped_key.ptr());
    const uint64_t chunk_start_index = chunk_id * num_values_per_chunk_;
    row_id_mapping_.reserve(row_id_mapping_.size() + n_entries);
    for (size_t i = 0; i < n_entries; ++i) {
      CHECK(row_id_mapping_.emplace(keys[indices[i] - chunk_start_index], indices[i]).second);
    }
    if (Hook) {
      PosixFile value_file(ValueFilePath(chunk_id), O_RDONLY, 0644);
      PosixMappedFile mapped_value(std::move(value_file), value_file.Size(), PROT_READ, mmap_flags);
      ChunkIteratorImpl<Key> chunk_iterator(value_size_, logical_block_size_, num_values_per_block_,
                                            num_values_per_chunk_, chunk_id, n_entries, keys,
                                            indices, mapped_value.ptr());
      Hook(&chunk_iterator);
    }
  }
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::SaveSnapshot(const std::string& name) {
  SaveSnapshotImpl(name);
}

template<typename Key, typename Engine>
PersistentTable::Iterator* PersistentTableImpl<Key, Engine>::ReadSnapshot(const std::string& name) {
  return new SnapshotIteratorImpl<Key, Engine>(this, name, value_size_, logical_block_size_,
                                               num_values_per_block_, num_values_per_chunk_);
}

template<typename Key, typename Engine>
void PersistentTableImpl<Key, Engine>::ParallelFor(size_t total,
                                                   const ForRange<Engine>& for_range) {
  BlockingCounter bc(workers_.size());
  std::atomic<size_t> counter(0);
  for (size_t i = 0; i < workers_.size(); ++i) {
    workers_.at(i)->Schedule([&](Engine* engine) {
      while (true) {
        const size_t start = counter.fetch_add(kParallelForStride, std::memory_order_relaxed);
        if (start >= total) { break; }
        const size_t next_start = start + kParallelForStride;
        const size_t end = std::min(next_start, total);
        for_range(engine, start, end);
      }
      engine->WaitUntilDone();
      bc.Decrease();
    });
  }
  bc.WaitForeverUntilCntEqualZero();
}

template<typename Key, typename Engine>
class SnapshotIteratorImpl : public PersistentTable::Iterator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SnapshotIteratorImpl);
  SnapshotIteratorImpl(PersistentTableImpl<Key, Engine>* table, const std::string& snapshot_name,
                       uint32_t value_size, uint32_t logical_block_size,
                       uint32_t num_values_per_block, uint64_t num_values_per_chunk)
      : table_(table),
        snapshot_name_(snapshot_name),
        value_size_(value_size),
        logical_block_size_(logical_block_size),
        num_values_per_block_(num_values_per_block),
        num_values_per_chunk_(num_values_per_chunk),
        current_chunk_(0) {
    const std::string snapshot_list = table_->SnapshotListFilePath(snapshot_name);
    std::ifstream list_if(snapshot_list);
    std::string index_filename;
    while (std::getline(list_if, index_filename)) { indices_names_.push_back(index_filename); }
  }
  ~SnapshotIteratorImpl() override = default;

  void Next(uint32_t num_keys, uint32_t* return_keys, void* keys, void* values) override {
    *return_keys = 0;
    while (current_chunk_ < indices_names_.size()) {
      if (!chunk_iterator_) {
        const std::string snapshot_base = table_->SnapshotDirPath(snapshot_name_);
        const uint64_t chunk_id = GetChunkId(indices_names_[current_chunk_], kIndexFileNamePrefix);
        PosixFile index_file(PosixFile::JoinPath(snapshot_base, indices_names_[current_chunk_]),
                             O_RDONLY, 0644);
        const size_t index_file_size = index_file.Size();
        CHECK_EQ(index_file_size % sizeof(uint64_t), 0);
        if (index_file_size == 0) {
          current_chunk_ += 1;
          continue;
        }
        const size_t n_entries = index_file_size / sizeof(uint64_t);
        indices_file_.reset(new PosixMappedFile(std::move(index_file), index_file_size, PROT_READ));
        PosixFile key_file(table_->KeyFilePath(chunk_id), O_RDONLY, 0644);
        keys_file_.reset(new PosixMappedFile(std::move(key_file), key_file.Size(), PROT_READ));
        PosixFile value_file(table_->ValueFilePath(chunk_id), O_RDONLY, 0644);
        values_file_.reset(
            new PosixMappedFile(std::move(value_file), value_file.Size(), PROT_READ));
        chunk_iterator_.reset(new ChunkIteratorImpl<Key>(
            value_size_, logical_block_size_, num_values_per_block_, num_values_per_chunk_,
            chunk_id, n_entries, static_cast<const Key*>(keys_file_->ptr()),
            static_cast<const uint64_t*>(indices_file_->ptr()), values_file_->ptr()));
      }
      chunk_iterator_->Next(num_keys, return_keys, keys, values);
      if (*return_keys == 0) {
        chunk_iterator_.reset();
        keys_file_.reset();
        values_file_.reset();
        indices_file_.reset();
        current_chunk_ += 1;
        continue;
      } else {
        return;
      }
    }
  }

  void Reset() override { UNIMPLEMENTED(); }

 private:
  PersistentTableImpl<Key, Engine>* table_;
  std::string snapshot_name_;
  uint32_t value_size_;
  uint32_t logical_block_size_;
  uint32_t num_values_per_block_;
  uint64_t num_values_per_chunk_;
  size_t current_chunk_;
  std::vector<std::string> indices_names_;
  std::unique_ptr<PosixMappedFile> keys_file_;
  std::unique_ptr<PosixMappedFile> values_file_;
  std::unique_ptr<PosixMappedFile> indices_file_;
  std::unique_ptr<ChunkIteratorImpl<Key>> chunk_iterator_;
};

template<typename Engine>
std::unique_ptr<PersistentTable> DispatchKeyType(const PersistentTableOptions& options) {
  if (options.key_size == 4) {
    return std::unique_ptr<PersistentTable>(new PersistentTableImpl<uint32_t, Engine>(options));
  } else if (options.key_size == 8) {
    return std::unique_ptr<PersistentTable>(new PersistentTableImpl<uint64_t, Engine>(options));
  } else {
    UNIMPLEMENTED();
    return nullptr;
  }
}

std::unique_ptr<PersistentTable> DispatchEngine(const PersistentTableOptions& options) {
  return DispatchKeyType<AioEngine>(options);
}

}  // namespace

#endif  // __linux__

std::unique_ptr<PersistentTable> NewPersistentTable(const PersistentTableOptions& options) {
#ifdef __linux__
  CHECK(!options.path.empty());
  CHECK_GT(options.value_size, 0);
  CHECK_GT(options.target_chunk_size_mb, 0);
  CHECK_GT(options.physical_block_size, 0);
  CHECK_GT(options.key_size, 0);
  return DispatchEngine(options);
#else
  UNIMPLEMENTED();
  return nullptr;
#endif  // __linux__
}

}  // namespace embedding

}  // namespace oneflow
