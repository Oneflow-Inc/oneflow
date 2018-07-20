#include "oneflow/core/register/lazy_blob.h"

namespace oneflow {

namespace test {

namespace {

template<typename T>
Blob* NewTestBlob(const BlobDesc* blob_desc, T* data) {
  return NewBlob(nullptr, blob_desc, reinterpret_cast<char*>(data), DeviceType::kCPU);
}

}  // namespace

TEST(LazyEvaluation, simple) {
  auto* blob_desc = new BlobDesc(Shape({1}), DataType::kInt32, false, false, 1);
  std::vector<int32_t> data{1};
  Blob* x_blob = NewTestBlob(blob_desc, data.data());
  CHECK_EQ(x_blob->dptr<int32_t>(), data.data());
  std::vector<int32_t> ret_data{0};
  Blob* ret_blob = NewTestBlob(blob_desc, ret_data.data());
  WithLazyEvaluation<int32_t>(
      [&](LazyBlobVarBuilder<int32_t>& var) { var(ret_blob) = var(x_blob) + var(x_blob); });
  ASSERT_EQ(ret_blob->dptr<int32_t>()[0], 1);
}

}  // namespace test

}  // namespace oneflow
