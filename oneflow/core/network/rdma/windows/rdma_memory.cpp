#include "oneflow/core/network/rdma/windows/rdma_memory.h"
#include "oneflow/core/network/rdma/windows/interface.h"

namespace oneflow {

RdmaMemory::RdmaMemory(IND2MemoryRegion* memory_region) {
  memory_region_ = memory_region;
}

// Register as ND memory region
void RdmaMemory::Register() {
  OVERLAPPED ov;
  ov.hEvent = CreateEvent(NULL, false, false, NULL);

  HRESULT hr = memory_region_->Register(
      memory_,
      size_,
      ND_MR_FLAG_ALLOW_LOCAL_WRITE |
      ND_MR_FLAG_ALLOW_REMOTE_READ |
      ND_MR_FLAG_ALLOW_REMOTE_WRITE,
      // TODO(shiyuan): TEST(FLAG)
      &ov);


  if (hr == ND_PENDING) {
    hr = memory_region_->GetOverlappedResult(&ov, TRUE);
  }

  // Set ND2_SGE
  sge_.Buffer = memory_;
  sge_.BufferLength = size_;  // TODO(shiyuan)
  sge_.MemoryRegionToken = memory_region_->GetLocalToken();

  descriptor_.address = (UINT64)memory_;
  descriptor_.remote_token = memory_region_->GetRemoteToken();

  registered_ = true;
  CloseHandle(ov.hEvent);
}

void RdmaMemory::Unregister() {
  OVERLAPPED ov;
  ov.hEvent = CreateEvent(NULL, FALSE, FALSE, NULL);
  HRESULT hr = memory_region_->Deregister(&ov);
  if (hr == ND_PENDING) {
    hr = memory_region_->GetOverlappedResult(&ov, TRUE);
  }
  registered_ = false;
  CloseHandle(ov.hEvent);
}

/*const void* Memory::sge() const {
  return &sge_;
}*/

}  // namespace oneflow
