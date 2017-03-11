#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "dag/blob_meta.h"
#include "layers/base_layer.h"
#include "layers/cudnn_relu_layer.h"
#include "memory/blob.h"

#include "test/test_gradient_check_util.h"
#include "test/test_job.h"
#include "test/test_proto_engine.h"


namespace caffe {
template <typename Dtype>
class CuDNNReLULayerTest : public ::testing::Test{
protected:
  CuDNNReLULayerTest() {
    RNG::set_seed(kSeed);
    cudaSetDevice(kDeviceId);
    //cublasCreate(&ctx.cublas_handle);
    cudnnCreate(&ctx.cudnn_handle);
    cudaStreamCreate(&ctx.cuda_stream);
    //cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);
    cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream);

    layer_name = "relu1";
    relu_proto = std::make_shared<ReLUProto>(
      *dynamic_cast<const ReLUProto*>(
      ProtoEngine<Dtype>::proto_param(layer_name)));

    relu_layer = std::make_shared<CuDNNReLULayer<Dtype>>(
      layer_name, relu_proto->DebugString());
    relu_layer->InitParam();
    relu_data = std::shared_ptr<ReLUData<Dtype>>(
      dynamic_cast<ReLUData<Dtype>*>(relu_layer->CreateDataParam()));
    relu_data->AllocateEmptyBlobs();
    relu_data->in->set_shape(Shape(2, 3, 4, 5));
  }
  ~CuDNNReLULayerTest() {
    //cublasDestroy(ctx.cublas_handle);
    cudnnDestroy(ctx.cudnn_handle);
    cudaStreamDestroy(ctx.cuda_stream);
  }

  void InitParameters() {
    relu_model = std::shared_ptr<ReLUModel<Dtype>>(
      dynamic_cast<ReLUModel<Dtype>*>(relu_layer->CreateModelParam()));
    relu_data->in->Reallocate(MemoryType::kDeviceMemory);
    relu_data->out->Reallocate(MemoryType::kDeviceMemory);
    relu_data->in_diff->Reallocate(MemoryType::kDeviceMemory);
    relu_data->out_diff->Reallocate(MemoryType::kDeviceMemory);
    Dtype* in_data = reinterpret_cast<Dtype*>(malloc(
      relu_data->in->byte_size()));
    int sign = 0;
    for (int i = 0; i < relu_data->in->shape().count(); ++i) {
      in_data[i] = (sign++) % 2 ? 1 : -1;
    }
    CUDA_CHECK(cudaMemcpy(relu_data->in->mutable_data(),
      in_data, relu_data->in->byte_size(), cudaMemcpyHostToDevice));
    free(in_data);
  }
  ContextParam ctx;
  std::string layer_name;
  std::shared_ptr<ReLUProto> relu_proto;
  std::shared_ptr<CuDNNReLULayer<Dtype>> relu_layer;
  std::shared_ptr<ReLUData<Dtype>> relu_data;
  std::shared_ptr<ReLUModel<Dtype>> relu_model;
};

TYPED_TEST_CASE(CuDNNReLULayerTest, TestDtypes);

TYPED_TEST(CuDNNReLULayerTest, TestSetup) {
  typedef typename TypeParam Dtype;

  relu_layer->InitFromInputShape(relu_data.get());

  Shape in = relu_data->in->shape();
  Shape out = relu_data->out->shape();
  Shape out_diff = relu_data->out_diff->shape();
  Shape in_diff = relu_data->in_diff->shape();
  EXPECT_EQ(in.count(), out.count());
  EXPECT_EQ(in.count(), out_diff.count());
  EXPECT_EQ(in.count(), in_diff.count());
}

TYPED_TEST(CuDNNReLULayerTest, TestSimpleForward) {
  typedef typename TypeParam Dtype;

  relu_layer->InitFromInputShape(relu_data.get());

  InitParameters();

  relu_layer->Forward(ctx, relu_data.get(), relu_model.get());

  Dtype* out_data = reinterpret_cast<Dtype*>(malloc(
    relu_data->out->byte_size()));
  CUDA_CHECK(cudaMemcpy(out_data, relu_data->out->data(),
    relu_data->out->byte_size(), cudaMemcpyDeviceToHost));
  for (int i = 0; i < relu_data->out->shape().count(); ++i) {
    EXPECT_GE(out_data[i], 0);
  }

  Dtype* in_ = reinterpret_cast<Dtype*>(malloc(
    relu_data->in->shape().count() * sizeof(Dtype)));
  Dtype* out_ = reinterpret_cast<Dtype*>(malloc(
    relu_data->out->shape().count() * sizeof(Dtype)));
  CUDA_CHECK(cudaMemcpy(in_, relu_data->in->data(),
    relu_data->in->shape().count() * sizeof(Dtype), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_, relu_data->out->data(),
    relu_data->out->shape().count() * sizeof(Dtype), cudaMemcpyDeviceToHost));
  for (int i = 0; i < relu_data->in->shape().count(); ++i) {
    EXPECT_EQ((in_[i] < 0 ? 0 : in_[i]), out_[i]);
  }
}
// TODO(xcdu): Gradient checker need to update to let x + step and x - step
// have no sign change
TYPED_TEST(CuDNNReLULayerTest, TestGradient) {
  typedef typename TypeParam Dtype;

  relu_layer->InitFromInputShape(relu_data.get());

  InitParameters();

  GradientChecker<Dtype> gradient_checker(1e-2, 1e-2);
  gradient_checker.CheckGradientExhaustive(ctx, relu_layer.get(),
    relu_data.get(), relu_model.get());
}

}  // namespace caffe
