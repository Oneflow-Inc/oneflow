#include "network/rdma/windows/memory.h"
#include "network/rdma/windows/interface.h"

namespace oneflow {

Memory::Memory(IND2MemoryRegion* memory_region) {
  memory_region_ = memory_region;
}

// Register as ND memory region
void Memory::Register() {
  OVERLAPPED ov;
  ov.hEvent = CreateEvent(NULL, false, false, NULL);

  HRESULT hr = memory_region_->Register(
      memory_,
      size_,
      ND_MR_FLAG_ALLOW_REMOTE_WRITE,
      // TODO(feiga): add argument to decide the flag
      // TODO(shiyuan): add local_read, local_write, and remote read access authority
      &ov);
  // TODO(jiyuan): for message, it should allow read.

  if (hr == ND_PENDING) {
    hr = memory_region_->GetOverlappedResult(&ov, TRUE);
  }

  // Set ND2_SGE
  sge_.Buffer = memory_;
  sge_.BufferLength = size_;
  sge_.MemoryRegionToken = memory_region_->GetLocalToken();

  descriptor_.address = (UINT64)memory_;
  descriptor_.remote_token = memory_region_->GetRemoteToken();

  registered_ = true;
  CloseHandle(ov.hEvent);
}

void Memory::Unregister() {
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
