#ifndef _BLOB_UTIL_H_
#define _BLOG_UTIL_H_
#include <glog/logging.h>
#include "memory/blob.h"
#include "device/device_alternate.h"
namespace caffe {
template <typename Dtype>
void AsyncCopyH2D(const Blob<Dtype>& src,
  Blob<Dtype>* dst, cudaStream_t stream) {
  CHECK_EQ(src.type_(), kHostPinnedMemory);
  CHECK_EQ(dst->type_(), kDeviceMemory);
  CHECK_EQ(src.byte_size(), dst->byte_size());
  CUDA_CHECK(cudaMemcpyAsync(dst->mutable_data(),
    src.data(), src.byte_size(),
    cudaMemcpyHostToDevice, stream));
}

template <typename Dtype>
void AyncCopyD2H(const Blob<Dtype>& src,
  Blob<Dtype>* dst, cudaStream_t stream) {
  CHECK_EQ(src.type_(), kDeviceMemory);
  CHECK_EQ(dst->type_(), kHostPinnedMemory);
  CHECK_EQ(src.byte_size(), dst->byte_size());
  CUDA_CHECK(cudaMemcpyAsync(dst->mutable_data(),
    src.data(), src.byte_size(),
    cudaMemcpyDeviceToHost, stream));
}

template <typename Dtype>
void SyncCopyH2H(const Blob<Dtype>& src,
  Blob<Dtype>* dst) {
  CHECK(src.type() == kHostPageableMemory
    || src.type() == kHostPinnedMemory);
  CHECK(dst->type() == kHostPageableMemory
    || dst->type() == kHostPinnedMemory);
  CHECK_EQ(src.byte_size(), dst->byte_size());
  memcpy(dst->mutable_data(), src.data(), src.byte_size());
}
}  // namespace caffe
#endif  // _BLOB_UTIL_H_
