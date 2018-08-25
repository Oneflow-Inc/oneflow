#include "oneflow/core/register/blob.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

// used by gdb only
namespace gdb {

// used by passing std::string param
static std::string param0;

static const Blob* CpuBlobCopiedFromGpuBlobPtr(uint64_t gpu_blob_ptr) {
  Blob* gpu_blob = reinterpret_cast<Blob*>(gpu_blob_ptr);
  char* cpu_body_ptr = reinterpret_cast<char*>(malloc(gpu_blob->ByteSizeOfDataContentField()));
  cudaMemcpy(cpu_body_ptr, gpu_blob->dptr(), gpu_blob->ByteSizeOfDataContentField(),
             cudaMemcpyDeviceToHost);
  return new Blob(const_cast<Regst*>(gpu_blob->regst()), gpu_blob->blob_desc_ptr(),
                  reinterpret_cast<char*>(gpu_blob->mut_header_ptr()), cpu_body_ptr);
}

static Blob* Blob4BnInOp(const std::function<Blob*(const std::string&)>* BnInOp2Blob,
                         const char* bn_in_op) {
  return (*BnInOp2Blob)(std::string(bn_in_op));
}

}  // namespace gdb

}  // namespace oneflow
