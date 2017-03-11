#include <gtest/gtest.h>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "dag/blob_meta.h"
#include "layers/base_layer.h"
#include "layers/cudnn_lrn_layer.h"
#include "memory/blob.h"

#include "test/test_job.h"
#include "test/test_gradient_check_util.h"
#include "test/test_proto_engine.h"

namespace caffe {
template <typename Dtype>
class CuDNNLRNLayerTest : public ::testing::Test{
protected:
  CuDNNLRNLayerTest() {
    RNG::set_seed(kSeed);
    cudaSetDevice(kDeviceId);
    //cublasCreate(&ctx.cublas_handle);
    cudnnCreate(&ctx.cudnn_handle);
    cudaStreamCreate(&ctx.cuda_stream);
    //cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);
    cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream);

    layer_name = "norm1";
    lrn_proto = std::make_shared<LRNProto>(
      *dynamic_cast<const LRNProto*>(
      ProtoEngine<Dtype>::proto_param(layer_name)));
  }
  ~CuDNNLRNLayerTest() {
    //cublasDestroy(ctx.cublas_handle);
    cudnnDestroy(ctx.cudnn_handle);
    cudaStreamDestroy(ctx.cuda_stream);
  }

  void InitParameters() {
    for (int i = 0; i < lrn_data->blob_names().size(); ++i) {

      (*lrn_data->name_to_blob_pptr().find(
        lrn_data->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
    lrn_model = std::shared_ptr<LRNModel<Dtype>>(
      dynamic_cast<LRNModel<Dtype>*>(
      lrn_layer->CreateModelParam()));
    lrn_model->AllocateEmptyBlobs();
    lrn_model->AlignBlobShapes(
      *dynamic_cast<const LRNModel<Dtype>*>(
      lrn_layer->GetModelParam()));

    for (int i = 0; i < lrn_model->blob_names().size(); ++i) {
      (*lrn_model->name_to_blob_pptr().find(
        lrn_model->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
  }

  ContextParam ctx;
  std::string layer_name;
  std::shared_ptr<LRNProto> lrn_proto;
  std::shared_ptr<CuDNNLRNLayer<Dtype>> lrn_layer;
  std::shared_ptr<LRNData<Dtype>> lrn_data;
  std::shared_ptr<LRNModel<Dtype>> lrn_model;

  void ReferenceLRNForward(const Shape& in_shape, const Dtype* in_data,
    Blob<Dtype>* blob_out) {

    Dtype* out_data = blob_out->mutable_data();

    Dtype alpha = lrn_proto->alpha();
    Dtype beta = lrn_proto->beta();
    int size = lrn_proto->local_size();
    switch (lrn_proto->norm_region()) {
    case LRNProto_NormRegion_ACROSS_CHANNELS:
      for (int n = 0; n < in_shape.num(); ++n) {
        for (int c = 0; c < in_shape.channels(); ++c) {
          for (int h = 0; h < in_shape.height(); ++h) {
            for (int w = 0; w < in_shape.width(); ++w) {
              int c_start = c - (size - 1) / 2;
              int c_end;// = //min(c_start + size, in_shape.channels());

              if (c_start + size < in_shape.channels()) c_end = c_start + size;
              else c_end = in_shape.channels();

              if (c_start < 0) c_start = 0;


              Dtype scale = 1.;
              for (int i = c_start; i < c_end; ++i) {

                Dtype value = in_data[in_shape.offset(n, i, h, w)];
                scale += value * value * alpha / size;
              }

              out_data[in_shape.offset(n, c, h, w)] =
                in_data[in_shape.offset(n, c, h, w)] / pow(scale, beta);
            }
          }
        }
      }
      break;
    case LRNProto_NormRegion_WITHIN_CHANNEL:
      // To do ...
      break;
    default:
      LOG(FATAL) << "Unknown normalization region.";
    }
  }

  void TestForwardForAcrossChannels() {

    std::shared_ptr<LRNProto> lrn_param =
      std::make_shared<LRNProto>();
    lrn_param->set_in(lrn_proto->in());
    lrn_param->set_out(lrn_proto->out());
    lrn_param->set_local_size(lrn_proto->local_size());
    lrn_param->set_alpha(lrn_proto->alpha());
    lrn_param->set_beta(lrn_proto->beta());
    lrn_param->set_k(lrn_proto->k());
    lrn_param->set_norm_region(LRNProto_NormRegion_ACROSS_CHANNELS);

    lrn_layer = std::make_shared<CuDNNLRNLayer<Dtype>>(
      layer_name, lrn_param->DebugString());

    lrn_layer->InitParam();
    lrn_data = std::shared_ptr<LRNData<Dtype>>(
      dynamic_cast<LRNData<Dtype>*>(lrn_layer->CreateDataParam()));
    lrn_data->AllocateEmptyBlobs();
    const int channels = 30;
    lrn_data->in->set_shape(Shape(2, channels, 3, 5));
    lrn_layer->InitFromInputShape(lrn_data.get());

    InitParameters();

    const Shape& in_shape = lrn_data->in->shape();
    const Shape& out_shape = lrn_data->out->shape();

    Dtype inputs_data_[2 * channels * 3 * 5];
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < in_shape.count(); i += 15) {
      inputs_data_[i + 0] = 1;
      inputs_data_[i + 1] = 2;
      inputs_data_[i + 2] = 5;
      inputs_data_[i + 3] = 2;
      inputs_data_[i + 4] = 3;
      inputs_data_[i + 5] = 9;
      inputs_data_[i + 6] = 4;
      inputs_data_[i + 7] = 1;
      inputs_data_[i + 8] = 4;
      inputs_data_[i + 9] = 8;
      inputs_data_[i + 10] = 1;
      inputs_data_[i + 11] = 2;
      inputs_data_[i + 12] = 5;
      inputs_data_[i + 13] = 2;
      inputs_data_[i + 14] = 3;
    }

    CUDA_CHECK(cudaMemcpy(lrn_data->in->mutable_data(), inputs_data_,
      in_shape.count()*sizeof(Dtype), cudaMemcpyHostToDevice));
    lrn_layer->Forward(ctx, lrn_data.get(), lrn_model.get());

    Dtype outputs_data_[2 * channels * 3 * 5];
    CUDA_CHECK(cudaMemcpy(outputs_data_, lrn_data->out->data(),
      out_shape.count()*sizeof(Dtype),
      cudaMemcpyDeviceToHost));

    Blob<Dtype> top_reference(lrn_data->in->shape(),
      MemoryType::kHostPageableMemory);
    this->ReferenceLRNForward(
      lrn_data->in->shape(), inputs_data_, &top_reference);

    for (int i = 0; i < out_shape.count(); ++i) {
      EXPECT_NEAR(outputs_data_[i], top_reference.data()[i], 1e-5);
    }
  }

  void TestGradientForAcrossChannels() {
    std::shared_ptr<LRNProto> lrn_param =
      std::make_shared<LRNProto>();
    lrn_param->set_in(lrn_proto->in());
    lrn_param->set_out(lrn_proto->out());
    lrn_param->set_local_size(lrn_proto->local_size());
    lrn_param->set_alpha(lrn_proto->alpha());
    lrn_param->set_beta(lrn_proto->beta());
    lrn_param->set_k(lrn_proto->k());
    lrn_param->set_norm_region(LRNProto_NormRegion_ACROSS_CHANNELS);

    lrn_layer = std::make_shared<CuDNNLRNLayer<Dtype>>(
      layer_name, lrn_param->DebugString());

    lrn_layer->InitParam();
    lrn_data = std::shared_ptr<LRNData<Dtype>>(
      dynamic_cast<LRNData<Dtype>*>(lrn_layer->CreateDataParam()));
    lrn_data->AllocateEmptyBlobs();
    const int channels = 30;
    lrn_data->in->set_shape(Shape(2, channels, 3, 5));
    lrn_layer->InitFromInputShape(lrn_data.get());

    InitParameters();

    const Shape& in_shape = lrn_data->in->shape();
    const Shape& out_shape = lrn_data->out->shape();

    Dtype inputs_data_[2 * channels * 3 * 5];
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < in_shape.count(); i += 15) {
      inputs_data_[i + 0] = 1;
      inputs_data_[i + 1] = 2;
      inputs_data_[i + 2] = 5;
      inputs_data_[i + 3] = 2;
      inputs_data_[i + 4] = 3;
      inputs_data_[i + 5] = 9;
      inputs_data_[i + 6] = 4;
      inputs_data_[i + 7] = 1;
      inputs_data_[i + 8] = 4;
      inputs_data_[i + 9] = 8;
      inputs_data_[i + 10] = 1;
      inputs_data_[i + 11] = 2;
      inputs_data_[i + 12] = 5;
      inputs_data_[i + 13] = 2;
      inputs_data_[i + 14] = 3;
    }

    CUDA_CHECK(cudaMemcpy(lrn_data->in->mutable_data(), inputs_data_,
      in_shape.count()*sizeof(Dtype), cudaMemcpyHostToDevice));

    GradientChecker<Dtype> gradient_checker(1e-2, 1e-2);
    gradient_checker.CheckGradientExhaustive(ctx, lrn_layer.get(),
      lrn_data.get(), lrn_model.get());
  }
};




TYPED_TEST_CASE(CuDNNLRNLayerTest, TestDtypes);

TYPED_TEST(CuDNNLRNLayerTest, TestSetupAcrossChannels) {
  typedef typename TypeParam Dtype;

  lrn_proto->set_local_size(5);
  lrn_proto->set_alpha(0.0001);
  lrn_proto->set_beta(0.75);
  lrn_proto->set_k(1.0);

  lrn_layer = std::make_shared<CuDNNLRNLayer<Dtype>>(
    layer_name, lrn_proto->DebugString());
  lrn_layer->InitParam();
  lrn_data = std::shared_ptr<LRNData<Dtype>>(
    dynamic_cast<LRNData<Dtype>*>(lrn_layer->CreateDataParam()));
  lrn_data->AllocateEmptyBlobs();
  lrn_data->in->set_shape(Shape(2, 3, 3, 5));
  lrn_layer->InitFromInputShape(lrn_data.get());


  EXPECT_EQ(lrn_data->out->shape().num(), lrn_data->in->shape().num());
  EXPECT_EQ(lrn_data->out->shape().channels(),
    lrn_data->in->shape().channels());
  EXPECT_EQ(lrn_data->out->shape().height(), 3);
  EXPECT_EQ(lrn_data->out->shape().width(), 5);;
  EXPECT_EQ(lrn_data->out_diff->shape().num(), lrn_data->in->shape().num());
  EXPECT_EQ(lrn_data->out_diff->shape().channels(),
    lrn_data->in->shape().channels());
  EXPECT_EQ(lrn_data->out_diff->shape().height(), 3);
  EXPECT_EQ(lrn_data->out_diff->shape().width(), 5);;
}

TYPED_TEST(CuDNNLRNLayerTest, TestForwardAcrossChannels) {
  lrn_proto->set_local_size(5);
  TestForwardForAcrossChannels();
}

TYPED_TEST(CuDNNLRNLayerTest, TestForwardAcrossChannelsLargeRegion) {
  lrn_proto->set_local_size(15);
  TestForwardForAcrossChannels();
}

TYPED_TEST(CuDNNLRNLayerTest, TestGradientAcrossChannels) {
  lrn_proto->set_local_size(5);
  TestGradientForAcrossChannels();
}

TYPED_TEST(CuDNNLRNLayerTest, TestGradientAcrossChannelsLargeRegion) {
  lrn_proto->set_local_size(15);
  TestGradientForAcrossChannels();
}


}
