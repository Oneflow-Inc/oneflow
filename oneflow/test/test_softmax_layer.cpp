#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

#include "gtest/gtest.h"

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "dag/blob_meta.h"
#include "layers/base_layer.h"
#include "layers/softmax_layer.h"
#include "memory/blob.h"

#include "test/test_proto_engine.h"
#include "test/test_gradient_check_util.h"
#include "test/test_job.h"

namespace caffe {
template<typename Dtype>
void softmax_forward(Blob<Dtype>* in, Blob<Dtype>* out) {
  const Dtype* data = in->data();
  size_t in_num = in->shape().num();
  size_t in_channels = in->shape().channels();
  size_t in_spatial = in->shape().height()*in->shape().width();
  for (int i = 0; i < out->shape().count(); ++i) {
    out->mutable_data()[i] = in->data()[i];
  }
  for (int i = 0; i < in_num*in_spatial; ++i) {
    int n = i / in_spatial;
    int s = i % in_spatial;
    Dtype maxval = -FLT_MAX;
    Dtype sum = 0;
    for (int c = 0; c < in_channels; ++c) {
      maxval = std::max(data[(n*in_channels + c)*in_spatial + s], maxval);
    }
    for (int c = 0; c < in_channels; ++c) {
      sum += std::exp(data[(n*in_channels + c)*in_spatial + s] - maxval);
    }
    for (int c = 0; c < in_channels; ++c) {
      out->mutable_data()[(n*in_channels + c)*in_spatial + s] = std::exp(
        out->mutable_data()[(n*in_channels + c)*in_spatial + s] - maxval) / sum;
    }
  }
}

template<typename Dtype>
class SoftmaxLayerTest :public ::testing::Test {
 protected:
  SoftmaxLayerTest() {
    RNG::set_seed(kSeed);
    cudaSetDevice(kDeviceId);
    cublasCreate(&ctx.cublas_handle);
    cudaStreamCreate(&ctx.cuda_stream);
    cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);

    layer_name = "softmax";
    softmax_proto = std::make_shared<SoftmaxProto>(
      *dynamic_cast<const SoftmaxProto*>(
      ProtoEngine<Dtype>::proto_param(layer_name)));
  }
  ~SoftmaxLayerTest() {
    cublasDestroy(ctx.cublas_handle);
    cudaStreamDestroy(ctx.cuda_stream);
  }

  void InitParameters() {
    for (int i = 0; i < softmax_data->blob_names().size(); ++i) {
      (*softmax_data->name_to_blob_pptr().find(
        softmax_data->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
    softmax_model = std::shared_ptr<SoftmaxModel<Dtype>>(
      dynamic_cast<SoftmaxModel<Dtype>*>(
      softmax_layer->CreateModelParam()));
    softmax_model->AllocateEmptyBlobs();
    softmax_model->AlignBlobShapes(
      *dynamic_cast<const SoftmaxModel<Dtype>*>(
      softmax_layer->GetModelParam()));

    for (int i = 0; i < softmax_model->blob_names().size(); ++i) {
      (*softmax_model->name_to_blob_pptr().find(
        softmax_model->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }

    FillerParameter gaussian_filler_param;
    gaussian_filler_param.set_mean(1.0);
    gaussian_filler_param.set_std(0.5);
    std::shared_ptr<GaussianFiller<Dtype>> gaussian_filler =
      std::make_shared<GaussianFiller<Dtype>>(gaussian_filler_param);
    gaussian_filler->fill(softmax_data->in);
  }

  ContextParam ctx;
  std::string layer_name;
  std::shared_ptr<SoftmaxProto> softmax_proto;
  std::shared_ptr<SoftmaxLayer<Dtype>> softmax_layer;
  std::shared_ptr<SoftmaxData<Dtype>> softmax_data;
  std::shared_ptr<SoftmaxModel<Dtype>> softmax_model;
};

TYPED_TEST_CASE(SoftmaxLayerTest, TestDtypes);

TYPED_TEST(SoftmaxLayerTest, TestSetup) {
  typedef typename TypeParam Dtype;

  softmax_layer = std::make_shared<SoftmaxLayer<Dtype>>(
    layer_name, softmax_proto->DebugString());
  softmax_layer->InitParam();
  softmax_data = std::shared_ptr<SoftmaxData<Dtype>>(
    dynamic_cast<SoftmaxData<Dtype>*>(softmax_layer->CreateDataParam()));
  softmax_data->AllocateEmptyBlobs();
  softmax_data->in->set_shape(Shape(2, 10, 2, 3));

  softmax_layer->InitFromInputShape(softmax_data.get());

  Shape in_shape = softmax_data->in->shape();
  Shape out_shape = softmax_data->out->shape();
  EXPECT_EQ(in_shape.count(), out_shape.count());
  EXPECT_EQ(in_shape.num(), out_shape.num());
  EXPECT_EQ(in_shape.channels(), out_shape.channels());
  EXPECT_EQ(in_shape.height(), out_shape.height());
  EXPECT_EQ(in_shape.width(), out_shape.width());
}

TYPED_TEST(SoftmaxLayerTest, TestForward) {
  typedef typename TypeParam Dtype;

  softmax_layer = std::make_shared<SoftmaxLayer<Dtype>>(
    layer_name, softmax_proto->DebugString());
  softmax_layer->InitParam();
  softmax_data = std::shared_ptr<SoftmaxData<Dtype>>(
    dynamic_cast<SoftmaxData<Dtype>*>(softmax_layer->CreateDataParam()));
  softmax_data->AllocateEmptyBlobs();
  softmax_data->in->set_shape(Shape(2, 10, 2, 3));

  softmax_layer->InitFromInputShape(softmax_data.get());

  InitParameters();

  Dtype* in_data_ = reinterpret_cast<Dtype*>(
    malloc(softmax_data->in->byte_size()));
  CUDA_CHECK(cudaMemcpy(in_data_, softmax_data->in->data(),
    softmax_data->in->byte_size(), cudaMemcpyDeviceToHost));

  softmax_layer->Forward(ctx, softmax_data.get(), softmax_model.get());

  Dtype* out_data_ = reinterpret_cast<Dtype*>(malloc(
    softmax_data->out->byte_size()));
  Dtype* refer_ = reinterpret_cast<Dtype*>(calloc(
    softmax_data->out->shape().count(), sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(out_data_, softmax_data->out->data(),
    softmax_data->out->byte_size(), cudaMemcpyDeviceToHost));
  Blob<Dtype>* in_ = new Blob<Dtype>(in_data_, softmax_data->in->shape(),
    MemoryType::kHostPageableMemory);
  Blob<Dtype>* ref_ = new Blob<Dtype>(refer_, softmax_data->out->shape(),
    MemoryType::kHostPageableMemory);
  softmax_forward(in_, ref_);
  for (int i = 0; i < softmax_data->out->shape().count(); ++i) {
    EXPECT_NEAR(out_data_[i], refer_[i], 1e-5);
  }
  free(in_data_);
  free(out_data_);
  free(refer_);
  free(in_);
  free(ref_);
}
TYPED_TEST(SoftmaxLayerTest, TestGradient) {
  typedef typename TypeParam Dtype;

  softmax_layer = std::make_shared<SoftmaxLayer<Dtype>>(
    layer_name, softmax_proto->DebugString());
  softmax_layer->InitParam();
  softmax_data = std::shared_ptr<SoftmaxData<Dtype>>(
    dynamic_cast<SoftmaxData<Dtype>*>(softmax_layer->CreateDataParam()));
  softmax_data->AllocateEmptyBlobs();
  softmax_data->in->set_shape(Shape(2, 10, 2, 3));

  softmax_layer->InitFromInputShape(softmax_data.get());

  InitParameters();

  GradientChecker<Dtype> gradient_checker(1e-2, 1e-2);
  gradient_checker.CheckGradientExhaustive(ctx, softmax_layer.get(),
    softmax_data.get(), softmax_model.get());
}
}  // namespace caffe
