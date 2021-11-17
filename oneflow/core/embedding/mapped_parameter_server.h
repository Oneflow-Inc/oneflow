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
#ifndef ONEFLOW_CORE_EMBEDDING_MAPPED_PARAMETER_SERVER_H_
#define ONEFLOW_CORE_EMBEDDING_MAPPED_PARAMETER_SERVER_H_

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <sys/mman.h>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include "oneflow/core/embedding/hash_function.cuh"
#include <stdlib.h>

namespace oneflow {

namespace embedding {

namespace {

class DirectMappedFile final {
 public:
  DirectMappedFile(const std::string& path, size_t size)
      : fd_guard_(open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, 0644)), size_(size) {
    const size_t page_size = sysconf(_SC_PAGE_SIZE);
    void* ptr = aligned_alloc(page_size, size_);
    if (ptr == nullptr) { throw std::runtime_error("malloc error"); }
    ptr_.reset(reinterpret_cast<unsigned char*>(ptr));
    if (fd_guard_.fd() < 0) { throw std::runtime_error(strerror(errno)); }
    struct stat sb {};
    if (fstat(fd_guard_.fd(), &sb) != 0) { throw std::runtime_error(strerror(errno)); }
    if (sb.st_size == 0) {
      memset(ptr, 0, size_);
    } else if (sb.st_size != size) {
      throw std::runtime_error("invalid size");
    } else {
      size_t n_read = 0;
      while (n_read != size_) {
        ssize_t n = read(fd_guard_.fd(), ptr_.get() + n_read, size - n_read);
        if (n == -1) { throw std::runtime_error("read error"); }
        n_read += n;
      }
    }
  }
  ~DirectMappedFile() { Sync(); }

  void Sync() {
    lseek(fd_guard_.fd(), 0, SEEK_SET);
    size_t n_write = 0;
    while (n_write != size_) {
      ssize_t n = write(fd_guard_.fd(), ptr_.get() + n_write, size_ - n_write);
      if (n == -1) { throw std::runtime_error("sync error"); }
      n_write += n;
    }
  }

  void* Ptr() { return ptr_.get(); }

 private:
  class FileDescriptorGuard final {
   public:
    explicit FileDescriptorGuard(int fd) : fd_(fd) {}
    ~FileDescriptorGuard() {
      if (fd_ != -1) { close(fd_); }
    }

    int fd() const { return fd_; }

   private:
    int fd_;
  };

  FileDescriptorGuard fd_guard_;
  size_t size_;
  std::unique_ptr<unsigned char> ptr_;
};

class MappedFile final {
 public:
  MappedFile(const std::string& path, size_t size, bool lock)
      : fd_guard_(open(path.c_str(), O_CREAT | O_RDWR, 0644)), size_(size), ptr_(MAP_FAILED) {
    if (fd_guard_.fd() < 0) { throw std::runtime_error(strerror(errno)); }
    struct stat sb {};
    if (fstat(fd_guard_.fd(), &sb) != 0) { throw std::runtime_error(strerror(errno)); }
    if (sb.st_size == 0) {
      if (ftruncate(fd_guard_.fd(), static_cast<off_t>(size)) != 0) {
        throw std::runtime_error(strerror(errno));
      }
    } else if (sb.st_size != size) {
      throw std::runtime_error("invalid size");
    }
    int flags = MAP_SHARED;
    if (lock) { flags |= MAP_LOCKED; }
    ptr_ = mmap(nullptr, size, PROT_READ | PROT_WRITE, flags, fd_guard_.fd(), 0);
    if (ptr_ == MAP_FAILED) { throw std::runtime_error(strerror(errno)); }
  }
  ~MappedFile() {
    if (ptr_ != MAP_FAILED) { munmap(ptr_, size_); }
  }

  void Sync() {
    if (msync(ptr_, size_, MS_SYNC) != 0) { throw std::runtime_error(strerror(errno)); }
  }

  void* Ptr() { return ptr_; }

 private:
  class FileDescriptorGuard final {
   public:
    explicit FileDescriptorGuard(int fd) : fd_(fd) {}
    ~FileDescriptorGuard() {
      if (fd_ != -1) { close(fd_); }
    }

    int fd() const { return fd_; }

   private:
    int fd_;
  };

  FileDescriptorGuard fd_guard_;
  size_t size_;
  void* ptr_;
};

template<typename Key, typename Elem, typename Index>
class MappedTable final {
 public:
  MappedTable(uint32_t log2_n_slot, uint64_t n_elem_per_value, const std::string& base_dir)
      : log2_n_slot_(log2_n_slot), n_elem_per_value_(n_elem_per_value) {
    n_slot_ = 1 << log2_n_slot_;
    slot_mask_ = ~((~0ULL) << log2_n_slot_);
    value_offset_in_key_value_pair_ = AlignedSize(sizeof(Key), sizeof(Elem));
    key_value_pair_size_ = AlignedSize(
        value_offset_in_key_value_pair_ + n_elem_per_value_ * sizeof(Elem), sizeof(Key));
    key_value_mapped_.reset(
        new MappedFile(base_dir + "/key_value.mm", n_slot_ * key_value_pair_size_, false));
    key_value_base_ = key_value_mapped_->Ptr();
    key_to_index_mapped_.reset(
        new DirectMappedFile(base_dir + "/key_to_index.mm", n_slot_ * sizeof(Index)));
    key_to_index_ = reinterpret_cast<Index*>(key_to_index_mapped_->Ptr());
    size_mapped_.reset(new MappedFile(base_dir + "/size.mm", sizeof(Index), true));
    size_ = reinterpret_cast<Index*>(size_mapped_->Ptr());
  }

  inline Index LookupIndex(Key key, uint64_t hash, bool insert) {
    const uint64_t init_slot = (hash & slot_mask_);
    uint64_t next_slot = init_slot;
    while (true) {
      Index slot_index = key_to_index_[next_slot];
      if (slot_index == 0) {
        if (insert) {
          Index new_index = (*size_) + 1;
          *size_ = new_index;
          key_to_index_[next_slot] = new_index;
          void* pair_base = GetKeyValuePtr(new_index);
          *GetKeyPtr(pair_base) = key;
          return new_index;
        } else {
          return 0;
        }
      } else {
        void* pair_base = GetKeyValuePtr(slot_index);
        if (*GetKeyPtr(pair_base) == key) {
          return slot_index;
        } else {
          next_slot = ((next_slot + 1) & slot_mask_);
          if (next_slot == init_slot) { break; }
        }
      }
    }
    return 0;
  }

  inline void ReadVector(Index index, Elem* vector) {
    void* pair_base = GetKeyValuePtr(index);
    Elem* value_ptr = GetValuePtr(pair_base);
    std::copy(value_ptr, value_ptr + n_elem_per_value_, vector);
  }

  inline void WriteVector(Index index, const Elem* vector) {
    void* pair_base = GetKeyValuePtr(index);
    Elem* value_ptr = GetValuePtr(pair_base);
    std::copy(vector, vector + n_elem_per_value_, value_ptr);
  }

  void Sync() {
    key_value_mapped_->Sync();
    key_to_index_mapped_->Sync();
    size_mapped_->Sync();
  }

 private:
  static inline uint64_t AlignedSize(uint64_t size, uint64_t align) {
    return (size + align - 1) / align * align;
  }

  inline Key* GetKeyPtr(void* pair_base) const { return reinterpret_cast<Key*>(pair_base); }

  inline Elem* GetValuePtr(void* pair_base) const {
    return reinterpret_cast<Elem*>(PtrOffset(pair_base, value_offset_in_key_value_pair_));
  }

  inline void* GetKeyValuePtr(Index index) const {
    return reinterpret_cast<unsigned char*>(key_value_base_) + key_value_pair_size_ * index;
  }

  inline void* PtrOffset(void* ptr, size_t offset) const {
    return reinterpret_cast<unsigned char*>(ptr) + offset;
  }

  uint32_t log2_n_slot_;
  uint64_t n_slot_;
  uint64_t n_elem_per_value_;
  void* key_value_base_;
  Index* key_to_index_;
  Index* size_;
  uint64_t slot_mask_;
  uint64_t value_offset_in_key_value_pair_;
  uint64_t key_value_pair_size_;
  std::unique_ptr<MappedFile> key_value_mapped_;
  std::unique_ptr<DirectMappedFile> key_to_index_mapped_;
  std::unique_ptr<MappedFile> size_mapped_;
};

}  // namespace

template<typename Key, typename Elem, typename Idx>
class MappedParameterServer final {
 public:
  MappedParameterServer(uint32_t log2_capacity, uint64_t n_elem_per_value,
                        const std::string& base_dir)
      : log2_capacity_(log2_capacity), n_elem_per_value_(n_elem_per_value) {
    table_.reset(new MappedTable<Key, Elem, uint64_t>(log2_capacity_, n_elem_per_value_, base_dir));
  }
  ~MappedParameterServer() = default;

  void Pull(Idx* n, const Key* keys, Idx* n_found, Idx* found_indices, Elem* found_vectors,
            Idx* n_miss, Idx* miss_indices) {
    *n_found = 0;
    *n_miss = 0;
    for (Idx i = 0; i < *n; ++i) {
      const Key key = keys[i];
      uint64_t hash = XXH64()(key);
      const uint64_t index = table_->LookupIndex(key, hash, false);
      if (index == 0) {
        miss_indices[*n_miss] = i;
        *n_miss += 1;
      } else {
        found_indices[*n_found] = i;
        table_->ReadVector(index, found_vectors + *n_found * n_elem_per_value_);
        *n_found += 1;
      }
    }
  }

  void Push(Idx* n, const Key* keys, const Elem* vectors) {
    for (Idx i = 0; i < *n; ++i) {
      const Key key = keys[i];
      uint64_t hash = XXH64()(key);
      const uint64_t index = table_->LookupIndex(key, hash, true);
      table_->WriteVector(index, vectors + i * n_elem_per_value_);
    }
  }

  void Sync() { table_->Sync(); }

 private:
  uint32_t log2_capacity_;
  uint64_t n_elem_per_value_;
  std::unique_ptr<MappedTable<Key, Elem, uint64_t>> table_;
};

}  // namespace embedding

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EMBEDDING_MAPPED_PARAMETER_SERVER_H_
