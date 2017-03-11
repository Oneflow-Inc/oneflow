#include <string>
#include <vector>
#include <unordered_map>

#include "gtest/gtest.h"

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "dag/blob_meta.h"
#include "layers/base_layer.h"
#include "layers/innerproduct_layer.h"
#include "memory/blob.h"

#include "test/test_gradient_check_util.h"
#include "test/test_job.h"
#include "test/test_proto_engine.h"

namespace caffe {
template <typename Dtype>
void caffe_innerproduct(
  InnerProductProto* innerproduct_proto,
  Blob<Dtype>* in, Blob<Dtype>* weight, Blob<Dtype>* bias, Blob<Dtype>* out) {
  size_t out_num = out->shape().num();
  size_t out_dim = out->shape().dim();
  for (int i = 0; i < out_num; ++i) {
    for (int j = 0; j < out_dim; ++j) {
      Dtype result = 0;
      size_t in_dim = in->shape().count() / in->shape().num();
      size_t weight_dim = weight->shape().count() / weight->shape().num();
      for (int k = 0; k < in_dim; ++k) {
        result += in->data()[i*in_dim + k] *
          weight->data()[j*weight_dim + k];
      }
      if (innerproduct_proto->bias_term()) {
        result += (Dtype)bias->data()[j];
      }
      out->mutable_data()[i*out_dim + j] = result;
    }
  }
}

template <typename Dtype>
class InnerProductLayerTest : public ::testing::Test {
 protected:
  InnerProductLayerTest() {
    RNG::set_seed(kSeed);
    cudaSetDevice(kDeviceId);
    cublasCreate(&ctx.cublas_handle);
    cudaStreamCreate(&ctx.cuda_stream);
    cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);

    layer_name = "ip1";
    innerproduct_proto = std::make_shared<InnerProductProto>(
      *dynamic_cast<const InnerProductProto*>(
      ProtoEngine<Dtype>::proto_param(layer_name)));
  }
  ~InnerProductLayerTest() {
    cublasDestroy(ctx.cublas_handle);
    cudaStreamDestroy(ctx.cuda_stream);
  }

  void InitParameters() {
    for (int i = 0; i < innerproduct_data->blob_names().size(); ++i) {
      (*innerproduct_data->name_to_blob_pptr().find(
        innerproduct_data->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
    innerproduct_model = std::shared_ptr<InnerProductModel<Dtype>>(
      dynamic_cast<InnerProductModel<Dtype>*>(
      innerproduct_layer->CreateModelParam()));
    innerproduct_model->AllocateEmptyBlobs();
    innerproduct_model->AlignBlobShapes(
      *dynamic_cast<const InnerProductModel<Dtype>*>(
      innerproduct_layer->GetModelParam()));

    for (int i = 0; i < innerproduct_model->blob_names().size(); ++i) {
      (*innerproduct_model->name_to_blob_pptr().find(
        innerproduct_model->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
    FillerParameter gaussian_filler_param;
    gaussian_filler_param.set_mean(0.5);
    gaussian_filler_param.set_std(0.1);
    std::shared_ptr<GaussianFiller<Dtype>> gaussian_filler =
      std::make_shared<GaussianFiller<Dtype>>(gaussian_filler_param);
    gaussian_filler->fill(innerproduct_data->in);
    gaussian_filler->fill(innerproduct_model->weight);
    gaussian_filler->fill(innerproduct_model->bias);

    FillerParameter constant_filler_param;
    constant_filler_param.set_value(1);
    std::shared_ptr<ConstantFiller<Dtype>> constant_filler =
      std::make_shared<ConstantFiller<Dtype>>(constant_filler_param);
    constant_filler->fill(innerproduct_model->bias_multiplier);
  }

  ContextParam ctx;
  std::string layer_name;
  std::shared_ptr<InnerProductProto> innerproduct_proto;
  std::shared_ptr<InnerProductLayer<Dtype>> innerproduct_layer;
  std::shared_ptr<InnerProductData<Dtype>> innerproduct_data;
  std::shared_ptr<InnerProductModel<Dtype>> innerproduct_model;
};

TYPED_TEST_CASE(InnerProductLayerTest, TestDtypes);

TYPED_TEST(InnerProductLayerTest, TestSetUp) {
  typedef typename TypeParam Dtype;

  innerproduct_proto->set_num_output(10);

  innerproduct_layer = std::make_shared<InnerProductLayer<Dtype>>(
    layer_name, innerproduct_proto->DebugString());
  innerproduct_layer->InitParam();
  innerproduct_data = std::shared_ptr<InnerProductData<Dtype>>(
    dynamic_cast<InnerProductData<Dtype>*>(
    innerproduct_layer->CreateDataParam()));
  innerproduct_data->AllocateEmptyBlobs();
  innerproduct_data->in->set_shape(Shape(2, 3, 4, 5));

  innerproduct_layer->InitFromInputShape(innerproduct_data.get());

  Shape weight_shape = dynamic_cast<const InnerProductModel<Dtype>*>(
    innerproduct_layer->GetModelParam())->weight->shape();
  Shape bias_shape = dynamic_cast<const InnerProductModel<Dtype>*>(
    innerproduct_layer->GetModelParam())->bias->shape();

  EXPECT_EQ(innerproduct_data->out->shape().num(), 2);
  EXPECT_EQ(innerproduct_data->out->shape().dim(), 10);
  EXPECT_EQ(innerproduct_data->out_diff->shape().num(), 2);
  EXPECT_EQ(innerproduct_data->out_diff->shape().dim(), 10);
  EXPECT_EQ(weight_shape.num(), 10);
  EXPECT_EQ(weight_shape.dim(), 3 * 4 * 5);
  EXPECT_EQ(bias_shape.num(), 10);
}

TYPED_TEST(InnerProductLayerTest, TestSimpleForward) {
  typedef typename TypeParam Dtype;

  innerproduct_proto->set_num_output(10);

  innerproduct_layer = std::make_shared<InnerProductLayer<Dtype>>(
    layer_name, innerproduct_proto->DebugString());
  innerproduct_layer->InitParam();
  innerproduct_data = std::shared_ptr<InnerProductData<Dtype>>(
    dynamic_cast<InnerProductData<Dtype>*>(
    innerproduct_layer->CreateDataParam()));
  innerproduct_data->AllocateEmptyBlobs();
  innerproduct_data->in->set_shape(Shape(2, 3, 4, 5));

  innerproduct_layer->InitFromInputShape(innerproduct_data.get());

  InitParameters();

  innerproduct_layer->Forward(ctx, innerproduct_data.get(),
    innerproduct_model.get());

  std::shared_ptr<Dtype> in_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_data->in->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> out_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> refer_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> weight_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_model->weight->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> bias_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_model->bias->shape().count(),
    sizeof(Dtype))));
  // copy from device to host
  CUDA_CHECK(cudaMemcpy(in_data_.get(), innerproduct_data->in->data(),
    innerproduct_data->in->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_data_.get(), innerproduct_data->out->data(),
    innerproduct_data->out->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(weight_data_.get(), innerproduct_model->weight->data(),
    innerproduct_model->weight->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(bias_data_.get(), innerproduct_model->bias->data(),
    innerproduct_model->bias->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  // encapsulation
  std::shared_ptr<Blob<Dtype>> in_ = std::make_shared<Blob<Dtype>>(
    in_data_.get(), innerproduct_data->in->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> refer_ = std::make_shared<Blob<Dtype>>(
    refer_data_.get(), innerproduct_data->out->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> wei_ = std::make_shared<Blob<Dtype>>(
    weight_data_.get(), innerproduct_model->weight->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> bi_ = std::make_shared<Blob<Dtype>>(
    bias_data_.get(), innerproduct_model->bias->shape(),
    MemoryType::kHostPageableMemory);
  // innerproduct
  caffe_innerproduct<Dtype>(innerproduct_proto.get(), in_.get(), wei_.get(),
    bi_.get(), refer_.get());
  // check
  for (int k = 0; k < innerproduct_data->out->shape().count(); ++k) {
    EXPECT_NEAR(out_data_.get()[k], refer_data_.get()[k], 1e-5);
  }
}

TYPED_TEST(InnerProductLayerTest, TestSimpleForwardNoBatch) {
  typedef typename TypeParam Dtype;

  innerproduct_proto->set_num_output(10);

  innerproduct_layer = std::make_shared<InnerProductLayer<Dtype>>(
    layer_name, innerproduct_proto->DebugString());
  innerproduct_layer->InitParam();
  innerproduct_data = std::shared_ptr<InnerProductData<Dtype>>(
    dynamic_cast<InnerProductData<Dtype>*>(
    innerproduct_layer->CreateDataParam()));
  innerproduct_data->AllocateEmptyBlobs();
  innerproduct_data->in->set_shape(Shape(1, 2, 3, 4));

  innerproduct_layer->InitFromInputShape(innerproduct_data.get());

  InitParameters();

  innerproduct_layer->Forward(ctx, innerproduct_data.get(),
    innerproduct_model.get());

  std::shared_ptr<Dtype> in_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_data->in->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> out_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> refer_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> weight_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_model->weight->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> bias_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(innerproduct_model->bias->shape().count(),
    sizeof(Dtype))));
  // copy from device to host
  CUDA_CHECK(cudaMemcpy(in_data_.get(), innerproduct_data->in->data(),
    innerproduct_data->in->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_data_.get(), innerproduct_data->out->data(),
    innerproduct_data->out->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(weight_data_.get(), innerproduct_model->weight->data(),
    innerproduct_model->weight->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(bias_data_.get(), innerproduct_model->bias->data(),
    innerproduct_model->bias->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  // encapsulation
  std::shared_ptr<Blob<Dtype>> in_ = std::make_shared<Blob<Dtype>>(
    in_data_.get(), innerproduct_data->in->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> refer_ = std::make_shared<Blob<Dtype>>(
    refer_data_.get(), innerproduct_data->out->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> wei_ = std::make_shared<Blob<Dtype>>(
    weight_data_.get(), innerproduct_model->weight->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> bi_ = std::make_shared<Blob<Dtype>>(
    bias_data_.get(), innerproduct_model->bias->shape(),
    MemoryType::kHostPageableMemory);
  // innerproduct
  caffe_innerproduct<Dtype>(innerproduct_proto.get(), in_.get(), wei_.get(),
    bi_.get(), refer_.get());
  // check
  for (int k = 0; k < innerproduct_data->out->shape().count(); ++k) {
    EXPECT_NEAR(out_data_.get()[k], refer_data_.get()[k], 1e-5);
  }
}

TYPED_TEST(InnerProductLayerTest, TestGradient) {
  typedef typename TypeParam Dtype;

  innerproduct_proto->set_num_output(10);

  innerproduct_layer = std::make_shared<InnerProductLayer<Dtype>>(
    layer_name, innerproduct_proto->DebugString());
  innerproduct_layer->InitParam();
  innerproduct_data = std::shared_ptr<InnerProductData<Dtype>>(
    dynamic_cast<InnerProductData<Dtype>*>(
    innerproduct_layer->CreateDataParam()));
  innerproduct_data->AllocateEmptyBlobs();
  innerproduct_data->in->set_shape(Shape(2, 3, 4, 5));

  innerproduct_layer->InitFromInputShape(innerproduct_data.get());

  InitParameters();

  GradientChecker<Dtype> gradient_checker(1e-2, 1e-2);
  gradient_checker.CheckGradientExhaustive(ctx, innerproduct_layer.get(),
    innerproduct_data.get(), innerproduct_model.get());
}

}  // namespace caffe
