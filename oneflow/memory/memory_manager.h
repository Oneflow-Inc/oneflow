#ifndef ONEFLOW_JOB_ID_MANAGER_H_
#define ONEFLOW_JOB_ID_MANAGER_H_

#include "cuda.h"
#include "common/util.h"

namespace oneflow {

enum class MemoryType {
  kHostPageableMemory = 0,
  kHostPinnedMemory,
  kGPUDeviceMemory
};
struct MemoryCase {
  MemoryType type;
  int32_t id;
};

template<typename Dptr>
class MemoryMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MemoryMgr);
  MemoryMgr();
  ~MemoryMgr() = default;
  
  static MemoryMgr& Singleton() {
    static MemoryMgr obj;
    return obj;
  }

  std::pair<Dptr*, std::function<void(Dptr*)>> AllocateMem(
      MemoryCase mem_cas,std::size_t size) {
    switch(mem_cas.type) {
      case MemoryType::kHostPageableMemory : {
                                               dptr = (Dptr*) malloc(size);
                                               break;
                                             }
      case MemoryType::kHostPinnedMemory : {
                                            CHECK_EQ(cudaMallocHost(&dptr, size), 0);
                                             break;
                                           }
      case MemoryType::kGPUDeviceMemory : {
                                            CHECK_EQ(cudaSetDevice(mem_cas.id), 0);
                                            CHECK_EQ(cudaMalloc(&dptr, size), 0);
                                            break;
                                          }
    }
    return std::make_pair(dptr, std::bind(&MemoryMgr::DeallocateMem, this, _1, mem_cas));
  }

 private:
  void DeallocateMem(Dptr* dptr, MemoryCase mem_cas) {
    switch(mem_cas.type) {
      case MemoryType::kHostPageableMemory : {
                                               dptr = free(dptr);
                                               break;
                                             }
      case MemoryType::kHostPinnedMemory : {
                                             CHECK_EQ(cudaFreeHost(&dptr), 0);
                                             break;
                                           }
      case MemoryType::kGPUDeviceMemory : {
                                            CHECK_EQ(cudaSetDevice(mem_cas.id), 0);
                                            CHECK_EQ(cudaFree(&dptr), 0);
                                            break;
                                          }
    } 
  }
};

}
#endif
