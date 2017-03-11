#include <cmath>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "dag/blob_meta.h"
#include "layers/base_layer.h"
#include "layers/multinomial_logistic_loss_layer.h"
#include "memory/blob.h"

#include "test/test_gradient_check_util.h"
#include "test/test_job.h"
#include "test/test_proto_engine.h"

namespace caffe {
template <typename Dtype>
class MultinomialLogisticLossLayerTest : public ::testing::Test{
 protected:
  MultinomialLogisticLossLayerTest() {
    RNG::set_seed(kSeed);
    cudaSetDevice(kDeviceId);
    cublasCreate(&ctx.cublas_handle);
    cublasSetPointerMode(ctx.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
    cudaStreamCreate(&ctx.cuda_stream);
    cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);

    layer_name = "loss";
    mll_proto = std::make_shared<MultinomialLogisticLossProto>(
      *dynamic_cast<const MultinomialLogisticLossProto*>(
      ProtoEngine<Dtype>::proto_param(layer_name)));
    mll_layer = std::make_shared<MultinomialLogisticLossLayer<Dtype>>(
      layer_name, mll_proto->DebugString());
    mll_layer->InitParam();
    mll_data = std::shared_ptr<MultinomialLogisticLossData<Dtype>>(
      dynamic_cast<MultinomialLogisticLossData<Dtype>*>(
      mll_layer->CreateDataParam()));
    mll_data->AllocateEmptyBlobs();
    mll_data->data->set_shape(Shape(10, 5));
    mll_data->data_diff->set_shape(Shape(10, 5));
    mll_data->label->set_shape(Shape(10, 1));
  }
  ~MultinomialLogisticLossLayerTest() {
    cublasDestroy(ctx.cublas_handle);
    cudaStreamDestroy(ctx.cuda_stream);
  }

  void InitParameters() {
    for (int i = 0; i < mll_data->blob_names().size(); ++i) {
      (*mll_data->name_to_blob_pptr().find(
        mll_data->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
    mll_model = std::shared_ptr<MultinomialLogisticLossModel<Dtype>>(
      dynamic_cast<MultinomialLogisticLossModel<Dtype>*>(
      mll_layer->CreateModelParam()));
    mll_model->AllocateEmptyBlobs();
    mll_model->AlignBlobShapes(
      *dynamic_cast<const MultinomialLogisticLossModel<Dtype>*>(
      mll_layer->GetModelParam()));

    for (int i = 0; i < mll_model->blob_names().size(); ++i) {
      (*mll_model->name_to_blob_pptr().find(
        mll_model->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }

    FillerParameter positive_unitball_filler_param;
    std::shared_ptr<PositiveUnitballFiller<Dtype>> positive_unitball_filler =
      std::make_shared<PositiveUnitballFiller<Dtype>>(
      positive_unitball_filler_param);
    positive_unitball_filler->fill(mll_data->data);

    FillerParameter discrete_uniform_filler_param;
    discrete_uniform_filler_param.set_min(0);
    discrete_uniform_filler_param.set_max(mll_data->data->shape().dim());
    std::shared_ptr<DiscreteUniformFiller<Dtype>> discrete_uniform_filler =
      std::make_shared<DiscreteUniformFiller<Dtype>>(
      discrete_uniform_filler_param);
    discrete_uniform_filler->fill(mll_data->label);

    FillerParameter constant_filler_param;
    constant_filler_param.set_value(1);
    std::shared_ptr<ConstantFiller<Dtype>> constant_filler =
      std::make_shared<ConstantFiller<Dtype>>(constant_filler_param);
    constant_filler->fill(mll_model->loss_multiplier);
  }
  ContextParam ctx;
  std::string layer_name;
  std::shared_ptr<MultinomialLogisticLossProto> mll_proto;
  std::shared_ptr<MultinomialLogisticLossLayer<Dtype>> mll_layer;
  std::shared_ptr<MultinomialLogisticLossData<Dtype>> mll_data;
  std::shared_ptr<MultinomialLogisticLossModel<Dtype>> mll_model;
};

TYPED_TEST_CASE(MultinomialLogisticLossLayerTest, TestDtypes);

TYPED_TEST(MultinomialLogisticLossLayerTest, TestSetup) {
  typedef typename TypeParam Dtype;

  mll_layer->InitFromInputShape(mll_data.get());

  EXPECT_EQ(mll_data->loss->shape().count(), 1);
  EXPECT_EQ(mll_data->loss_buffer->shape().count(), 50);
}
TYPED_TEST(MultinomialLogisticLossLayerTest, TestSimpleForward) {
  typedef typename TypeParam Dtype;

  mll_layer->InitFromInputShape(mll_data.get());

  InitParameters();

  mll_layer->Forward(ctx, mll_data.get(), mll_model.get());

  Shape data_shape = mll_data->data->shape();
  Shape label_shape = mll_data->label->shape();
  Shape output_shape = mll_data->loss->shape();
  Dtype* data = reinterpret_cast<Dtype*>(calloc(data_shape.count(),
    sizeof(Dtype)));
  Dtype* label = reinterpret_cast<Dtype*>(calloc(label_shape.count(),
    sizeof(Dtype)));
  Dtype* output = reinterpret_cast<Dtype*>(calloc(output_shape.count(),
    sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(data, mll_data->data->data(),
    mll_data->data->byte_size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(label, mll_data->label->data(),
    mll_data->label->byte_size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(output, mll_data->loss->data(),
    mll_data->loss->byte_size(), cudaMemcpyDeviceToHost));
  Dtype loss = 0.0;
  size_t dim = data_shape.count() / data_shape.num();
  for (int i = 0; i < data_shape.num(); ++i) {
    loss -= log(std::max(data[i*dim + static_cast<int>(label[i])],
      (Dtype)1e-20));
  }
  loss /= data_shape.num();
  EXPECT_NEAR(loss, output[0], 1e-6);
  free(data);
  free(label);
  free(output);
}
TYPED_TEST(MultinomialLogisticLossLayerTest, TestGradient) {
  typedef typename TypeParam Dtype;

  mll_layer->InitFromInputShape(mll_data.get());

  InitParameters();

  GradientChecker<Dtype> gradient_checker(1e-4, 1e-2);
  gradient_checker.CheckGradientExhaustive(ctx, mll_layer.get(),
    mll_data.get(), mll_model.get());
}

}  // namespace caffe
