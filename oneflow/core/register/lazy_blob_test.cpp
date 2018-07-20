#include "oneflow/core/register/lazy_blob.h"

namespace oneflow {

namespace test {

namespace {

template<typename T>
Blob* NewTestBlob(const BlobDesc* blob_desc, T* data) {
  return NewBlob(nullptr, blob_desc, reinterpret_cast<char*>(data), DeviceType::kCPU);
}

}  // namespace

TEST(LazyEvaluation, simple_add) {
  auto* blob_desc = new BlobDesc(Shape({1}), DataType::kInt32, false, false, 1);
  std::vector<int32_t> data{1};
  Blob* x_blob = NewTestBlob(blob_desc, data.data());
  CHECK_EQ(x_blob->dptr<int32_t>(), data.data());
  std::vector<int32_t> ret_data{0};
  Blob* ret_blob = NewTestBlob(blob_desc, ret_data.data());
  LAZY_EVALUATE(int32_t, var) { var(ret_blob) = var(x_blob) + var(x_blob); }
  ASSERT_EQ(ret_blob->dptr<int32_t>()[0], 2);
}

TEST(LazyEvaluation, simple_mul) {
  auto* blob_desc = new BlobDesc(Shape({1}), DataType::kInt32, false, false, 1);
  std::vector<int32_t> data{1};
  Blob* x_blob = NewTestBlob(blob_desc, data.data());
  CHECK_EQ(x_blob->dptr<int32_t>(), data.data());
  std::vector<int32_t> ret_data{0};
  Blob* ret_blob = NewTestBlob(blob_desc, ret_data.data());
  LAZY_EVALUATE(int32_t, var) {
    auto& val0 = var(x_blob) + var(x_blob);
    auto& val1 = var(x_blob) + var(x_blob) + var(x_blob);
    var(ret_blob) = val0 * val1;
  }
  ASSERT_EQ(ret_blob->dptr<int32_t>()[0], 6);
}

TEST(LazyEvaluation, add) {
  std::vector<int32_t> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto* blob_desc =
      new BlobDesc(Shape({static_cast<int64_t>(data.size())}), DataType::kInt32, false, false, 1);
  Blob* x_blob = NewTestBlob(blob_desc, data.data());
  CHECK_EQ(x_blob->dptr<int32_t>(), data.data());
  std::vector<int32_t> ret_data(data);
  Blob* ret_blob = NewTestBlob(blob_desc, ret_data.data());
  LAZY_EVALUATE(int32_t, var) { var(ret_blob) = var(x_blob) + var(x_blob); }
  FOR_RANGE(int32_t, i, 0, data.size()) { ASSERT_EQ(ret_blob->dptr<int32_t>()[i], i * 2); }
}

TEST(LazyEvaluation, mul) {
  std::vector<int32_t> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto* blob_desc =
      new BlobDesc(Shape({static_cast<int64_t>(data.size())}), DataType::kInt32, false, false, 1);
  Blob* x_blob = NewTestBlob(blob_desc, data.data());
  CHECK_EQ(x_blob->dptr<int32_t>(), data.data());
  std::vector<int32_t> ret_data(data);
  Blob* ret_blob = NewTestBlob(blob_desc, ret_data.data());
  LAZY_EVALUATE(int32_t, var) {
    auto& val0 = var(x_blob) + var(x_blob);
    auto& val1 = var(x_blob) + var(x_blob) + var(x_blob);
    var(ret_blob) = val0 * val1;
  }
  FOR_RANGE(int32_t, i, 0, data.size()) { ASSERT_EQ(ret_blob->dptr<int32_t>()[i], i * i * 6); }
}

}  // namespace test

}  // namespace oneflow
