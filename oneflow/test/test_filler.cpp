#include <gtest/gtest.h>

#include "caffe.pb.h"
#include "common/filler.h"
#include "common/shape.h"
#include "memory/blob.h"
#include "test/test_job.h"

namespace caffe {

template <typename Dtype>
class ConstantFillerTest : public ::testing::Test {
 protected:
  ConstantFillerTest(): blob_(), filler_param_() {
    filler_param_.set_value((Dtype)10.);
    filler_.reset(new ConstantFiller<Dtype>(filler_param_));
    blob_ = std::make_shared<Blob<Dtype>>(Shape(10, 10, 10, 10),
      MemoryType::kDeviceMemory);
    filler_->fill(blob_.get());
  }
  virtual ~ConstantFillerTest() {}
  FillerParameter filler_param_;
  std::shared_ptr<Blob<Dtype>> blob_;
  std::shared_ptr<ConstantFiller<Dtype> > filler_;
};

TYPED_TEST_CASE(ConstantFillerTest, TestDtypes);

TYPED_TEST(ConstantFillerTest, TestFill) {
  typedef typename TypeParam Dtype;
  EXPECT_TRUE(blob_);
  const int count = blob_->shape().count();
  Dtype* host_data = reinterpret_cast<Dtype*>(calloc(count, sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(host_data, blob_->data(),
    count*sizeof(Dtype), cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(host_data[i], filler_param_.value());
  }
  free(host_data);
}
template<typename Dtype>
class UniformFillerTest : public ::testing::Test{
 protected:
  UniformFillerTest(): blob_(), filler_param_() {
    filler_param_.set_min(Dtype(100.));
    filler_param_.set_max(Dtype(10000.));
    filler_.reset(new UniformFiller<Dtype>(filler_param_));
    blob_ = std::make_shared<Blob<Dtype>>(Shape(10, 10, 10, 10),
      MemoryType::kDeviceMemory);
    filler_->fill(blob_.get());
  }
  virtual ~UniformFillerTest() {}
  std::shared_ptr<Blob<Dtype>> blob_;
  FillerParameter filler_param_;
  std::shared_ptr<UniformFiller<Dtype>> filler_;
};
TYPED_TEST_CASE(UniformFillerTest, TestDtypes);

TYPED_TEST(UniformFillerTest, TestFill) {
  typedef typename TypeParam Dtype;
  EXPECT_TRUE(blob_);
  const int count = blob_->shape().count();
  Dtype* host_data = reinterpret_cast<Dtype*>(calloc(count, sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(host_data, blob_->data(),
    count*sizeof(Dtype), cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(host_data[i], filler_param_.min());
    EXPECT_LE(host_data[i], filler_param_.max());
  }
  free(host_data);
}
template<typename Dtype>
class DiscreteUniformFillerTest : public ::testing::Test{
 protected:
  DiscreteUniformFillerTest() :blob_(), filler_param_() {
    filler_param_.set_min(Dtype(100.));
    filler_param_.set_max(Dtype(10000.));
    filler_.reset(new DiscreteUniformFiller<Dtype>(filler_param_));
    blob_ = std::make_shared<Blob<Dtype>>(Shape(10, 10, 10, 10),
      MemoryType::kDeviceMemory);
    filler_->fill(blob_.get());
  }
  virtual ~DiscreteUniformFillerTest() {}
  std::shared_ptr<Blob<Dtype>> blob_;
  FillerParameter filler_param_;
  std::shared_ptr<DiscreteUniformFiller<Dtype>> filler_;
};
TYPED_TEST_CASE(DiscreteUniformFillerTest, TestDtypes);

TYPED_TEST(DiscreteUniformFillerTest, TestFill) {
  typedef typename TypeParam Dtype;
  EXPECT_TRUE(blob_);
  const int count = blob_->shape().count();
  Dtype* host_data = reinterpret_cast<Dtype*>(calloc(count, sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(host_data, blob_->data(),
    count*sizeof(Dtype), cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(host_data[i], filler_param_.min());
    EXPECT_LE(host_data[i], filler_param_.max());
    EXPECT_EQ(host_data[i] - floor(host_data[i]), 0);
  }
  free(host_data);
}
template<typename Dtype>
class GaussianFillerTest : public ::testing::Test {
 protected:
  GaussianFillerTest() :blob_(), filler_param_() {
    filler_param_.set_mean(Dtype(10.));
    filler_param_.set_std(Dtype(1.));
    filler_.reset(new GaussianFiller<Dtype>(filler_param_));
    blob_ = std::make_shared<Blob<Dtype>>(Shape(10, 10, 10, 10),
      MemoryType::kDeviceMemory);
    filler_->fill(blob_.get());
  }
  virtual ~GaussianFillerTest() {}
  std::shared_ptr<Blob<Dtype>> blob_;
  FillerParameter filler_param_;
  std::shared_ptr<GaussianFiller<Dtype>> filler_;
};
TYPED_TEST_CASE(GaussianFillerTest, TestDtypes);

TYPED_TEST(GaussianFillerTest, TestFill) {
  typedef typename TypeParam Dtype;
  EXPECT_TRUE(blob_);
  const int count = blob_->shape().count();
  Dtype* host_data = reinterpret_cast<Dtype*>(calloc(count, sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(host_data, blob_->data(),
    count*sizeof(Dtype), cudaMemcpyDeviceToHost));
  Dtype mean = 0., std = 0.;
  for (int i = 0; i < count; ++i) {
    mean += host_data[i];
    std += (host_data[i] - filler_param_.mean())*(host_data[i] -
      filler_param_.mean());
  }
  mean = mean / count;
  std = std / count;
  EXPECT_GT(mean, filler_param_.mean() - filler_param_.std() * 5);
  EXPECT_LT(mean, filler_param_.mean() + filler_param_.std() * 5);
  EXPECT_GT(std, pow(filler_param_.std(), 2) / 5.);
  EXPECT_LT(std, pow(filler_param_.std(), 2) * 5.);
  free(host_data);
}
template<typename Dtype>
class PositiveUnitballFillerTest : public ::testing::Test{
 protected:
  PositiveUnitballFillerTest() :blob_(), filler_param_() {
    filler_.reset(new PositiveUnitballFiller<Dtype>(filler_param_));
    blob_ = std::make_shared<Blob<Dtype>>(Shape(10, 10, 10, 10),
      MemoryType::kDeviceMemory);
    filler_->fill(blob_.get());
  }
  virtual ~PositiveUnitballFillerTest() {}
  std::shared_ptr<Blob<Dtype>> blob_;
  FillerParameter filler_param_;
  std::shared_ptr<PositiveUnitballFiller<Dtype>> filler_;
};
TYPED_TEST_CASE(PositiveUnitballFillerTest, TestDtypes);

TYPED_TEST(PositiveUnitballFillerTest, TestFill) {
  typedef typename TypeParam Dtype;
  EXPECT_TRUE(blob_);
  const int count = blob_->shape().count();
  const int num = blob_->shape().num();
  const int dim = count / num;
  Dtype* host_data = reinterpret_cast<Dtype*>(calloc(count, sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(host_data, blob_->data(),
    count*sizeof(Dtype), cudaMemcpyDeviceToHost));
  for (int i = 0; i < num; ++i) {
    Dtype sum = 0;
    for (int j = 0; j < dim; ++j) {
      EXPECT_GE(host_data[i*dim+j], 0);
      EXPECT_LE(host_data[i*dim+j], 1);
      sum += host_data[i*dim + j];
    }
    EXPECT_NEAR(sum, (Dtype)1.0, 1e-5);
  }
  free(host_data);
}
template<typename Dtype>
class XavierFillerTest : public ::testing::Test{
 protected:
  XavierFillerTest() :blob_(), filler_param_() {
    filler_.reset(new XavierFiller<Dtype>(filler_param_));
    blob_ = std::make_shared<Blob<Dtype>>(Shape(10, 10, 10, 10),
      MemoryType::kDeviceMemory);
    filler_->fill(blob_.get());
  }
  virtual ~XavierFillerTest() {}
  std::shared_ptr<Blob<Dtype>> blob_;
  FillerParameter filler_param_;
  std::shared_ptr<XavierFiller<Dtype>> filler_;
};
TYPED_TEST_CASE(XavierFillerTest, TestDtypes);

TYPED_TEST(XavierFillerTest, TestFill) {
  typedef typename TypeParam Dtype;
  EXPECT_TRUE(blob_);
  const int count = blob_->shape().count();
  const int num = blob_->shape().num();
  Dtype* host_data = reinterpret_cast<Dtype*>(calloc(count, sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(host_data, blob_->data(),
    count*sizeof(Dtype), cudaMemcpyDeviceToHost));
  Dtype scale = std::sqrt(Dtype(3) / (count / num));
  for (int i = 0; i < count; ++i) {
    EXPECT_GE(host_data[i], -scale);
    EXPECT_LE(host_data[i], scale);
  }
  free(host_data);
}
}  // namespace caffe
