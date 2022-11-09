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
#include "oneflow/core/ipc/shared_memory.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/pcheck.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/env_var/env_var.h"
#ifdef __linux__
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <error.h>
#include <dirent.h>
#endif

namespace oneflow {
namespace ipc {

namespace {

#ifdef __linux__

// return errno
int ShmOpen(const std::string& shm_name, int* fd, bool create) {
  SharedMemoryManager::get().AddShmName(shm_name);
  *fd = shm_open(("/" + shm_name).c_str(), (create ? O_CREAT : 0) | O_RDWR | O_EXCL,
                 S_IRUSR | S_IWUSR);
  return *fd == -1 ? errno : 0;
}

// return errno
int ShmOpen(std::string* shm_name, int* fd, bool create) {
  int err = EEXIST;
  while (true) {
    static constexpr int kNameLength = 8;
    *shm_name = std::string("ofshm_") + GenAlphaNumericString(kNameLength);
    err = ShmOpen(*shm_name, fd, create);
    if (err != EEXIST) { return err; }
  }
  return err;
}

int ShmMap(int fd, const size_t shm_size, void** ptr) {
  *ptr = mmap(NULL, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  return (*ptr == MAP_FAILED) ? errno : 0;
}

#endif

Maybe<void*> ShmSetUp(std::string* shm_name, size_t shm_size, bool create) {
#ifdef __linux__
  int fd = 0;
  PCHECK_OR_RETURN(ShmOpen(shm_name, &fd, create));
  PCHECK_OR_RETURN(posix_fallocate(fd, 0, shm_size)) << ReturnEmptyStr([&] { close(fd); });
  void* ptr = nullptr;
  PCHECK_OR_RETURN(ShmMap(fd, shm_size, &ptr)) << ReturnEmptyStr([&] { close(fd); });
  close(fd);
  std::memset(ptr, 0, shm_size);
  return ptr;
#else
  TODO_THEN_RETURN();
#endif
}

Maybe<void*> ShmSetUp(const std::string& shm_name, size_t* shm_size, bool create) {
#ifdef __linux__
  int fd = 0;
  PCHECK_OR_RETURN(ShmOpen(shm_name, &fd, create));
  struct stat st;  // NOLINT
  PCHECK_OR_RETURN(fstat(fd, &st)) << ReturnEmptyStr([&] { close(fd); });
  *shm_size = st.st_size;
  void* ptr = nullptr;
  PCHECK_OR_RETURN(ShmMap(fd, *shm_size, &ptr)) << ReturnEmptyStr([&] { close(fd); });
  close(fd);
  return ptr;
#else
  TODO_THEN_RETURN();
#endif
}

Maybe<std::set<std::string>> GetContentsOfShmDirectory() {
#ifdef __linux__
  std::set<std::string> contents;
  DIR* dir = opendir("/dev/shm/");
  CHECK_NOTNULL_OR_RETURN(dir)
      << "/dev/shm directory does not exist, there may be a problem with your machine!";
  while (dirent* f = readdir(dir)) {
    if (f->d_name[0] == '.') continue;
    contents.insert(f->d_name);
  }
  closedir(dir);
  return contents;
#else
  TODO_THEN_RETURN();
#endif
}
}  // namespace

SharedMemoryManager& SharedMemoryManager::get() {
  // Must be a static singleton variable instead of Singleton<SharedMemoryManager>.
  // Subprocesses don't have chance to call `Singleton<SharedMemoryManager>::Delete()`
  static SharedMemoryManager shared_memory_manager;
  return shared_memory_manager;
}

void SharedMemoryManager::FindAndDeleteOutdatedShmNames() {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  static size_t counter = 0;
  const int delete_invalid_names_interval =
      EnvInteger<ONEFLOW_DELETE_OUTDATED_SHM_NAMES_INTERVAL>();
  if (counter % delete_invalid_names_interval == 0) {
    const auto& existing_shm_names = CHECK_JUST(GetContentsOfShmDirectory());
    // std::remove_if doesn't support std::map
    for (auto it = shm_names_.begin(); it != shm_names_.end(); /* do nothing */) {
      if (existing_shm_names->find(*it) == existing_shm_names->end()) {
        it = shm_names_.erase(it);
      } else {
        it++;
      }
    }
  }
  counter++;
}

void SharedMemoryManager::AddShmName(const std::string& shm_name) {
  FindAndDeleteOutdatedShmNames();
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  shm_names_.insert(shm_name);
}

Maybe<void> SharedMemoryManager::DeleteShmName(const std::string& shm_name) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = std::find(shm_names_.begin(), shm_names_.end(), shm_name);
  if (it != shm_names_.end()) {
    shm_names_.erase(it);
  } else {
    return Error::RuntimeError() << "shared memory was not created but attempted to be freed.";
  }
  return Maybe<void>::Ok();
}

void SharedMemoryManager::UnlinkAllShms() {
#ifdef __linux__
  // Here we deliberately do not handle unlink errors.
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  for (const auto& shm : shm_names_) { shm_unlink(shm.c_str()); }
  shm_names_.clear();
#else
  UNIMPLEMENTED();
#endif
}

SharedMemoryManager::~SharedMemoryManager() { UnlinkAllShms(); }

SharedMemory::~SharedMemory() { CHECK_JUST(Close()); }

Maybe<SharedMemory> SharedMemory::Open(size_t shm_size, bool create) {
  std::string shm_name;
  char* ptr = static_cast<char*>(JUST(ShmSetUp(&shm_name, shm_size, create)));
  return std::shared_ptr<SharedMemory>(new SharedMemory(ptr, shm_name, shm_size));
}

Maybe<SharedMemory> SharedMemory::Open(const std::string& shm_name, bool create) {
  size_t shm_size = 0;
  char* ptr = static_cast<char*>(JUST(ShmSetUp(shm_name, &shm_size, create)));
  return std::shared_ptr<SharedMemory>(new SharedMemory(ptr, shm_name, shm_size));
}

Maybe<void> SharedMemory::Close() {
#ifdef __linux__
  if (buf_ != nullptr) {
    PCHECK_OR_RETURN(munmap(buf_, size_));
    buf_ = nullptr;
  }
  return Maybe<void>::Ok();
#else
  TODO_THEN_RETURN();
#endif
}

Maybe<void> SharedMemory::Unlink() {
#ifdef __linux__
  PCHECK_OR_RETURN(shm_unlink(name_.c_str()));
  JUST(SharedMemoryManager::get().DeleteShmName(name_));
  return Maybe<void>::Ok();
#else
  TODO_THEN_RETURN();
#endif
}

}  // namespace ipc
}  // namespace oneflow
