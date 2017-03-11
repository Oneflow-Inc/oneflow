#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "dag/blob_meta.h"
#include "layers/base_layer.h"
#include "layers/cudnn_pooling_layer.h"
#include "memory/blob.h"

#include "test/test_job.h"
#include "test/test_gradient_check_util.h"
#include "test/test_proto_engine.h"

namespace caffe {
  template <typename Dtype>
  class CuDNNPoolingLayerTest : public ::testing::Test{
  protected:
    CuDNNPoolingLayerTest() {
      RNG::set_seed(kSeed);
      cudaSetDevice(kDeviceId);
      //cublasCreate(&ctx.cublas_handle);
      cudnnCreate(&ctx.cudnn_handle);
      cudaStreamCreate(&ctx.cuda_stream);
      //cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);
      cudnnSetStream(ctx.cudnn_handle, ctx.cuda_stream);

      layer_name = "pool1";
      pooling_proto = std::make_shared<PoolingProto>(
        *dynamic_cast<const PoolingProto*>(
        ProtoEngine<Dtype>::proto_param(layer_name)));
    }
    ~CuDNNPoolingLayerTest() {
      //cublasDestroy(ctx.cublas_handle);
      cudnnDestroy(ctx.cudnn_handle);
      cudaStreamDestroy(ctx.cuda_stream);
    }

    void InitParameters() {
      for (int i = 0; i < pooling_data->blob_names().size(); ++i) {
        (*pooling_data->name_to_blob_pptr().find(
          pooling_data->blob_names()[i])->second)->Reallocate(
          MemoryType::kDeviceMemory);
      }
      pooling_model = std::shared_ptr<PoolingModel<Dtype>>(
        dynamic_cast<PoolingModel<Dtype>*>(
        pooling_layer->CreateModelParam()));
      pooling_model->AllocateEmptyBlobs();
      pooling_model->AlignBlobShapes(
        *dynamic_cast<const PoolingModel<Dtype>*>(
        pooling_layer->GetModelParam()));

      for (int i = 0; i < pooling_model->blob_names().size(); ++i) {
        (*pooling_model->name_to_blob_pptr().find(
          pooling_model->blob_names()[i])->second)->Reallocate(
          MemoryType::kDeviceMemory);
      }
    }

    ContextParam ctx;
    std::string layer_name;
    std::shared_ptr<PoolingProto> pooling_proto;
    std::shared_ptr<CuDNNPoolingLayer<Dtype>> pooling_layer;
    std::shared_ptr<CuDNNPoolingData<Dtype>> pooling_data;
    std::shared_ptr<PoolingModel<Dtype>> pooling_model;

    void TestForwardSquare() {
      const int kNum = 2;
      const int kChannels = 2;
      std::shared_ptr<PoolingProto> proto_param =
        std::make_shared<PoolingProto>();
      proto_param->set_in(pooling_proto->in());
      proto_param->set_out(pooling_proto->out());
      proto_param->set_kernel_size(2);
      proto_param->set_pool(PoolingProto_PoolMethod_MAX);

      pooling_layer = std::make_shared<CuDNNPoolingLayer<Dtype>>(
        layer_name, proto_param->DebugString());
      pooling_layer->InitParam();
      pooling_data = std::shared_ptr<CuDNNPoolingData<Dtype>>(
        dynamic_cast<CuDNNPoolingData<Dtype>*>(
        pooling_layer->CreateDataParam()));
      pooling_data->AllocateEmptyBlobs();
      pooling_data->in->set_shape(Shape(kNum, kChannels, 3, 5));
      pooling_layer->InitFromInputShape(pooling_data.get());

      InitParameters();

      EXPECT_EQ(pooling_data->out->shape().num(), kNum);
      EXPECT_EQ(pooling_data->out->shape().channels(), kChannels);
      EXPECT_EQ(pooling_data->out->shape().height(), 2);
      EXPECT_EQ(pooling_data->out->shape().width(), 4);
      EXPECT_EQ(pooling_data->out_diff->shape().num(), kNum);
      EXPECT_EQ(pooling_data->out_diff->shape().channels(), kChannels);
      EXPECT_EQ(pooling_data->out_diff->shape().height(), 2);
      EXPECT_EQ(pooling_data->out_diff->shape().width(), 4);
      Dtype inputs_data_[kNum * kChannels * 3 * 5];
      // Input: 2x 2 channels of:
      //     [1 2 5 2 3]
      //     [9 4 1 4 8]
      //     [1 2 5 2 3]
      for (int i = 0; i < 15 * kNum * kChannels; i += 15) {
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

      CUDA_CHECK(cudaMemcpy(pooling_data->in->mutable_data(), inputs_data_,
        pooling_data->in->shape().count()*sizeof(Dtype), cudaMemcpyHostToDevice));
      pooling_layer->Forward(ctx, pooling_data.get(), pooling_model.get());
      // Expected output: 2x 2 channels of:
      //     [9 5 5 8]
      //     [9 5 5 8]
      Dtype outputs_data_[kNum * kChannels * 8];
      Dtype idx_[kNum * kChannels * 2 * 4];
      CUDA_CHECK(cudaMemcpy(outputs_data_, pooling_data->out->data(),
        pooling_data->out->shape().count()*sizeof(Dtype),
        cudaMemcpyDeviceToHost));
      // CUDA_CHECK(cudaMemcpy(idx_, pooling_data->idx->data(),
      //   pooling_data->idx->shape().count()*sizeof(Dtype),
      // cudaMemcpyDeviceToHost));
      for (int i = 0; i < 8 * kNum * kChannels; i += 8) {
        EXPECT_EQ(outputs_data_[i + 0], 9);
        EXPECT_EQ(outputs_data_[i + 1], 5);
        EXPECT_EQ(outputs_data_[i + 2], 5);
        EXPECT_EQ(outputs_data_[i + 3], 8);
        EXPECT_EQ(outputs_data_[i + 4], 9);
        EXPECT_EQ(outputs_data_[i + 5], 5);
        EXPECT_EQ(outputs_data_[i + 6], 5);
        EXPECT_EQ(outputs_data_[i + 7], 8);
      }
#if 0
      for (int i = 0; i < 8 * kNum * kChannels; i += 8) {
        EXPECT_EQ(idx_[i + 0], 5);
        EXPECT_EQ(idx_[i + 1], 2);
        EXPECT_EQ(idx_[i + 2], 2);
        EXPECT_EQ(idx_[i + 3], 9);
        EXPECT_EQ(idx_[i + 4], 5);
        EXPECT_EQ(idx_[i + 5], 12);
        EXPECT_EQ(idx_[i + 6], 12);
        EXPECT_EQ(idx_[i + 7], 9);
      }
#endif
    }

    // Test for 3x 2 rectangular pooling layer with kernel_h > kernel_w
    void TestForwardRectHigh() {
      const int kNum = 2;
      const int kChannels = 2;
      std::shared_ptr<PoolingProto> proto_param =
        std::make_shared<PoolingProto>();
      proto_param->set_in(pooling_proto->in());
      proto_param->set_out(pooling_proto->out());
      proto_param->set_kernel_h(3);
      proto_param->set_kernel_w(2);
      proto_param->set_pool(PoolingProto_PoolMethod_MAX);

      pooling_layer = std::make_shared<CuDNNPoolingLayer<Dtype>>(
        layer_name, proto_param->DebugString());
      pooling_layer->InitParam();
      pooling_data = std::shared_ptr<CuDNNPoolingData<Dtype>>(
        dynamic_cast<CuDNNPoolingData<Dtype>*>(
        pooling_layer->CreateDataParam()));
      pooling_data->AllocateEmptyBlobs();
      pooling_data->in->set_shape(Shape(kNum, kChannels, 6, 6));
      pooling_layer->InitFromInputShape(pooling_data.get());

      InitParameters();

      EXPECT_EQ(pooling_data->out->shape().num(), kNum);
      EXPECT_EQ(pooling_data->out->shape().channels(), kChannels);
      EXPECT_EQ(pooling_data->out->shape().height(), 4);
      EXPECT_EQ(pooling_data->out->shape().width(), 5);
      EXPECT_EQ(pooling_data->out_diff->shape().num(), kNum);
      EXPECT_EQ(pooling_data->out_diff->shape().channels(), kChannels);
      EXPECT_EQ(pooling_data->out_diff->shape().height(), 4);
      EXPECT_EQ(pooling_data->out_diff->shape().width(), 5);

      Dtype inputs_data_[kNum * kChannels * 6 * 6];
      // Input: 2x 2 channels of:
      // [35     1     6    26    19    24]
      // [ 3    32     7    21    23    25]
      // [31     9     2    22    27    20]
      // [ 8    28    33    17    10    15]
      // [30     5    34    12    14    16]
      // [ 4    36    29    13    18    11]
      for (int i = 0; i < 36 * kNum * kChannels; i += 36) {
        inputs_data_[i + 0] = 35;
        inputs_data_[i + 1] = 1;
        inputs_data_[i + 2] = 6;
        inputs_data_[i + 3] = 26;
        inputs_data_[i + 4] = 19;
        inputs_data_[i + 5] = 24;
        inputs_data_[i + 6] = 3;
        inputs_data_[i + 7] = 32;
        inputs_data_[i + 8] = 7;
        inputs_data_[i + 9] = 21;
        inputs_data_[i + 10] = 23;
        inputs_data_[i + 11] = 25;
        inputs_data_[i + 12] = 31;
        inputs_data_[i + 13] = 9;
        inputs_data_[i + 14] = 2;
        inputs_data_[i + 15] = 22;
        inputs_data_[i + 16] = 27;
        inputs_data_[i + 17] = 20;
        inputs_data_[i + 18] = 8;
        inputs_data_[i + 19] = 28;
        inputs_data_[i + 20] = 33;
        inputs_data_[i + 21] = 17;
        inputs_data_[i + 22] = 10;
        inputs_data_[i + 23] = 15;
        inputs_data_[i + 24] = 30;
        inputs_data_[i + 25] = 5;
        inputs_data_[i + 26] = 34;
        inputs_data_[i + 27] = 12;
        inputs_data_[i + 28] = 14;
        inputs_data_[i + 29] = 16;
        inputs_data_[i + 30] = 4;
        inputs_data_[i + 31] = 36;
        inputs_data_[i + 32] = 29;
        inputs_data_[i + 33] = 13;
        inputs_data_[i + 34] = 18;
        inputs_data_[i + 35] = 11;
      }

      CUDA_CHECK(cudaMemcpy(pooling_data->in->mutable_data(), inputs_data_,
        pooling_data->in->shape().count()*sizeof(Dtype), cudaMemcpyHostToDevice));

      pooling_layer->Forward(ctx, pooling_data.get(), pooling_model.get());

      Dtype outputs_data_[kNum * kChannels * 20];
      // Dtype idx_[kNum * kChannels * 20];
      CUDA_CHECK(cudaMemcpy(outputs_data_, pooling_data->out->data(),
        pooling_data->out->shape().count()*sizeof(Dtype),
        cudaMemcpyDeviceToHost));
      // CUDA_CHECK(cudaMemcpy(idx_, pooling_data->idx->data(),
      //   pooling_data->idx->shape().count()*sizeof(Dtype),
      //   cudaMemcpyDeviceToHost));
      // Expected output: 2x 2 channels of:
      // [35    32    26    27    27]
      // [32    33    33    27    27]
      // [31    34    34    27    27]
      // [36    36    34    18    18]
      for (int i = 0; i < 20 * kNum * kChannels; i += 20) {
        EXPECT_EQ(outputs_data_[i + 0], 35);
        EXPECT_EQ(outputs_data_[i + 1], 32);
        EXPECT_EQ(outputs_data_[i + 2], 26);
        EXPECT_EQ(outputs_data_[i + 3], 27);
        EXPECT_EQ(outputs_data_[i + 4], 27);
        EXPECT_EQ(outputs_data_[i + 5], 32);
        EXPECT_EQ(outputs_data_[i + 6], 33);
        EXPECT_EQ(outputs_data_[i + 7], 33);
        EXPECT_EQ(outputs_data_[i + 8], 27);
        EXPECT_EQ(outputs_data_[i + 9], 27);
        EXPECT_EQ(outputs_data_[i + 10], 31);
        EXPECT_EQ(outputs_data_[i + 11], 34);
        EXPECT_EQ(outputs_data_[i + 12], 34);
        EXPECT_EQ(outputs_data_[i + 13], 27);
        EXPECT_EQ(outputs_data_[i + 14], 27);
        EXPECT_EQ(outputs_data_[i + 15], 36);
        EXPECT_EQ(outputs_data_[i + 16], 36);
        EXPECT_EQ(outputs_data_[i + 17], 34);
        EXPECT_EQ(outputs_data_[i + 18], 18);
        EXPECT_EQ(outputs_data_[i + 19], 18);
      }
      // [ 1     8     4    17    17]
      // [ 8    21    21    17    17]
      // [13    27    27    17    17]
      // [32    32    27    35    35]
#if 0
      for (int i = 0; i < 20 * kNum * kChannels; i += 20) {
        EXPECT_EQ(idx_[i + 0], 0);
        EXPECT_EQ(idx_[i + 1], 7);
        EXPECT_EQ(idx_[i + 2], 3);
        EXPECT_EQ(idx_[i + 3], 16);
        EXPECT_EQ(idx_[i + 4], 16);
        EXPECT_EQ(idx_[i + 5], 7);
        EXPECT_EQ(idx_[i + 6], 20);
        EXPECT_EQ(idx_[i + 7], 20);
        EXPECT_EQ(idx_[i + 8], 16);
        EXPECT_EQ(idx_[i + 9], 16);
        EXPECT_EQ(idx_[i + 10], 12);
        EXPECT_EQ(idx_[i + 11], 26);
        EXPECT_EQ(idx_[i + 12], 26);
        EXPECT_EQ(idx_[i + 13], 16);
        EXPECT_EQ(idx_[i + 14], 16);
        EXPECT_EQ(idx_[i + 15], 31);
        EXPECT_EQ(idx_[i + 16], 31);
        EXPECT_EQ(idx_[i + 17], 26);
        EXPECT_EQ(idx_[i + 18], 34);
        EXPECT_EQ(idx_[i + 19], 34);
      }
#endif
    }
    // Test for rectangular pooling layer with kernel_w > kernel_h
    void TestForwardRectWide() {
      const int kNum = 2;
      const int kChannels = 2;
      std::shared_ptr<PoolingProto> proto_param =
        std::make_shared<PoolingProto>();
      proto_param->set_in(pooling_proto->in());
      proto_param->set_out(pooling_proto->out());
      proto_param->set_kernel_h(2);
      proto_param->set_kernel_w(3);
      proto_param->set_pool(PoolingProto_PoolMethod_MAX);

      pooling_layer = std::make_shared<CuDNNPoolingLayer<Dtype>>(
        layer_name, proto_param->DebugString());
      pooling_layer->InitParam();
      pooling_data = std::shared_ptr<CuDNNPoolingData<Dtype>>(
        dynamic_cast<CuDNNPoolingData<Dtype>*>(
        pooling_layer->CreateDataParam()));
      pooling_data->AllocateEmptyBlobs();
      pooling_data->in->set_shape(Shape(kNum, kChannels, 6, 6));
      pooling_layer->InitFromInputShape(pooling_data.get());

      InitParameters();

      EXPECT_EQ(pooling_data->out->shape().num(), kNum);
      EXPECT_EQ(pooling_data->out->shape().channels(), kChannels);
      EXPECT_EQ(pooling_data->out->shape().height(), 5);
      EXPECT_EQ(pooling_data->out->shape().width(), 4);
      EXPECT_EQ(pooling_data->out_diff->shape().num(), kNum);
      EXPECT_EQ(pooling_data->out_diff->shape().channels(), kChannels);
      EXPECT_EQ(pooling_data->out_diff->shape().height(), 5);
      EXPECT_EQ(pooling_data->out_diff->shape().width(), 4);

      Dtype inputs_data_[kNum * kChannels * 6 * 6];
      // Input: 2x 2 channels of:
      // [35     1     6    26    19    24]
      // [ 3    32     7    21    23    25]
      // [31     9     2    22    27    20]
      // [ 8    28    33    17    10    15]
      // [30     5    34    12    14    16]
      // [ 4    36    29    13    18    11]
      //// (this is generated by magic(6) in MATLAB)
      for (int i = 0; i < 36 * kNum * kChannels; i += 36) {
        inputs_data_[i + 0] = 35;
        inputs_data_[i + 1] = 1;
        inputs_data_[i + 2] = 6;
        inputs_data_[i + 3] = 26;
        inputs_data_[i + 4] = 19;
        inputs_data_[i + 5] = 24;
        inputs_data_[i + 6] = 3;
        inputs_data_[i + 7] = 32;
        inputs_data_[i + 8] = 7;
        inputs_data_[i + 9] = 21;
        inputs_data_[i + 10] = 23;
        inputs_data_[i + 11] = 25;
        inputs_data_[i + 12] = 31;
        inputs_data_[i + 13] = 9;
        inputs_data_[i + 14] = 2;
        inputs_data_[i + 15] = 22;
        inputs_data_[i + 16] = 27;
        inputs_data_[i + 17] = 20;
        inputs_data_[i + 18] = 8;
        inputs_data_[i + 19] = 28;
        inputs_data_[i + 20] = 33;
        inputs_data_[i + 21] = 17;
        inputs_data_[i + 22] = 10;
        inputs_data_[i + 23] = 15;
        inputs_data_[i + 24] = 30;
        inputs_data_[i + 25] = 5;
        inputs_data_[i + 26] = 34;
        inputs_data_[i + 27] = 12;
        inputs_data_[i + 28] = 14;
        inputs_data_[i + 29] = 16;
        inputs_data_[i + 30] = 4;
        inputs_data_[i + 31] = 36;
        inputs_data_[i + 32] = 29;
        inputs_data_[i + 33] = 13;
        inputs_data_[i + 34] = 18;
        inputs_data_[i + 35] = 11;
      }
      CUDA_CHECK(cudaMemcpy(pooling_data->in->mutable_data(), inputs_data_,
        pooling_data->in->shape().count()*sizeof(Dtype), cudaMemcpyHostToDevice));
      pooling_layer->Forward(ctx, pooling_data.get(), pooling_model.get());

      Dtype outputs_data_[kNum * kChannels * 20];
      // Dtype idx_[kNum * kChannels * 20];
      CUDA_CHECK(cudaMemcpy(outputs_data_, pooling_data->out->data(),
        pooling_data->out->shape().count()*sizeof(Dtype),
        cudaMemcpyDeviceToHost));
      // CUDA_CHECK(cudaMemcpy(idx_, pooling_data->idx->data(),
      //   pooling_data->idx->shape().count()*sizeof(Dtype),
      //   cudaMemcpyDeviceToHost));
      // Expected output: 2x 2 channels of:
      // [35    32    26    26]
      // [32    32    27    27]
      // [33    33    33    27]
      // [34    34    34    17]
      // [36    36    34    18]
      for (int i = 0; i < 20 * kNum * kChannels; i += 20) {
        EXPECT_EQ(outputs_data_[i + 0], 35);
        EXPECT_EQ(outputs_data_[i + 1], 32);
        EXPECT_EQ(outputs_data_[i + 2], 26);
        EXPECT_EQ(outputs_data_[i + 3], 26);
        EXPECT_EQ(outputs_data_[i + 4], 32);
        EXPECT_EQ(outputs_data_[i + 5], 32);
        EXPECT_EQ(outputs_data_[i + 6], 27);
        EXPECT_EQ(outputs_data_[i + 7], 27);
        EXPECT_EQ(outputs_data_[i + 8], 33);
        EXPECT_EQ(outputs_data_[i + 9], 33);
        EXPECT_EQ(outputs_data_[i + 10], 33);
        EXPECT_EQ(outputs_data_[i + 11], 27);
        EXPECT_EQ(outputs_data_[i + 12], 34);
        EXPECT_EQ(outputs_data_[i + 13], 34);
        EXPECT_EQ(outputs_data_[i + 14], 34);
        EXPECT_EQ(outputs_data_[i + 15], 17);
        EXPECT_EQ(outputs_data_[i + 16], 36);
        EXPECT_EQ(outputs_data_[i + 17], 36);
        EXPECT_EQ(outputs_data_[i + 18], 34);
        EXPECT_EQ(outputs_data_[i + 19], 18);
      }
      // [ 1     8     4     4]
      // [ 8     8    17    17]
      // [21    21    21    17]
      // [27    27    27    22]
      // [32    32    27    35]
#if 0
      for (int i = 0; i < 20 * kNum * kChannels; i += 20) {
        EXPECT_EQ(idx_[i + 0], 0);
        EXPECT_EQ(idx_[i + 1], 7);
        EXPECT_EQ(idx_[i + 2], 3);
        EXPECT_EQ(idx_[i + 3], 3);
        EXPECT_EQ(idx_[i + 4], 7);
        EXPECT_EQ(idx_[i + 5], 7);
        EXPECT_EQ(idx_[i + 6], 16);
        EXPECT_EQ(idx_[i + 7], 16);
        EXPECT_EQ(idx_[i + 8], 20);
        EXPECT_EQ(idx_[i + 9], 20);
        EXPECT_EQ(idx_[i + 10], 20);
        EXPECT_EQ(idx_[i + 11], 16);
        EXPECT_EQ(idx_[i + 12], 26);
        EXPECT_EQ(idx_[i + 13], 26);
        EXPECT_EQ(idx_[i + 14], 26);
        EXPECT_EQ(idx_[i + 15], 21);
        EXPECT_EQ(idx_[i + 16], 31);
        EXPECT_EQ(idx_[i + 17], 31);
        EXPECT_EQ(idx_[i + 18], 26);
        EXPECT_EQ(idx_[i + 19], 34);
      }
#endif
    }
  };

  TYPED_TEST_CASE(CuDNNPoolingLayerTest, TestDtypes);

  TYPED_TEST(CuDNNPoolingLayerTest, TestSetup) {
    typedef typename TypeParam Dtype;

    pooling_proto->set_kernel_size(3);
    pooling_proto->set_stride(2);

    pooling_layer = std::make_shared<CuDNNPoolingLayer<Dtype>>(
      layer_name, pooling_proto->DebugString());
    pooling_layer->InitParam();
    pooling_data = std::shared_ptr<CuDNNPoolingData<Dtype>>(
      dynamic_cast<CuDNNPoolingData<Dtype>*>(
      pooling_layer->CreateDataParam()));
    pooling_data->AllocateEmptyBlobs();
    pooling_data->in->set_shape(Shape(2, 3, 6, 5));
    pooling_layer->InitFromInputShape(pooling_data.get());

    EXPECT_EQ(pooling_data->out->shape().num(), pooling_data->in->shape().num());
    EXPECT_EQ(pooling_data->out->shape().channels(),
      pooling_data->in->shape().channels());
    EXPECT_EQ(pooling_data->out->shape().height(), 3);
    EXPECT_EQ(pooling_data->out->shape().width(), 2);;
    EXPECT_EQ(pooling_data->out_diff->shape().num(), pooling_data->in->shape().num());
    EXPECT_EQ(pooling_data->out_diff->shape().channels(),
      pooling_data->in->shape().channels());
    EXPECT_EQ(pooling_data->out_diff->shape().height(), 3);
    EXPECT_EQ(pooling_data->out_diff->shape().width(), 2);;
  }

  TYPED_TEST(CuDNNPoolingLayerTest, TestSetupPadded) {
    typedef typename TypeParam Dtype;

    pooling_proto->set_kernel_size(3);
    pooling_proto->set_stride(2);
    pooling_proto->set_pad(1);
    pooling_proto->set_pool(PoolingProto_PoolMethod_AVE);

    pooling_layer = std::make_shared<CuDNNPoolingLayer<Dtype>>(
      layer_name, pooling_proto->DebugString());
    pooling_layer->InitParam();
    pooling_data = std::shared_ptr<CuDNNPoolingData<Dtype>>(
      dynamic_cast<CuDNNPoolingData<Dtype>*>(
      pooling_layer->CreateDataParam()));
    pooling_data->AllocateEmptyBlobs();
    pooling_data->in->set_shape(Shape(2, 3, 6, 5));
    pooling_layer->InitFromInputShape(pooling_data.get());


    EXPECT_EQ(pooling_data->out->shape().num(), pooling_data->in->shape().num());
    EXPECT_EQ(pooling_data->out->shape().channels(),
      pooling_data->in->shape().channels());
    EXPECT_EQ(pooling_data->out->shape().height(), 4);
    EXPECT_EQ(pooling_data->out->shape().width(), 3);
    EXPECT_EQ(pooling_data->out_diff->shape().num(), pooling_data->in->shape().num());
    EXPECT_EQ(pooling_data->out_diff->shape().channels(),
      pooling_data->in->shape().channels());
    EXPECT_EQ(pooling_data->out_diff->shape().height(), 4);
    EXPECT_EQ(pooling_data->out_diff->shape().width(), 3);
  }

  TYPED_TEST(CuDNNPoolingLayerTest, TestSetupGlobalPooling) {
    typedef typename TypeParam Dtype;
    std::shared_ptr<PoolingProto> proto_param = std::make_shared<PoolingProto>();
    proto_param->set_in(pooling_proto->in());
    proto_param->set_out(pooling_proto->out());
    proto_param->set_global_pooling(true);
    proto_param->set_pool(PoolingProto_PoolMethod_AVE);

    pooling_layer = std::make_shared<CuDNNPoolingLayer<Dtype>>(
      layer_name, proto_param->DebugString());
    pooling_layer->InitParam();
    pooling_data = std::shared_ptr<CuDNNPoolingData<Dtype>>(
      dynamic_cast<CuDNNPoolingData<Dtype>*>(
      pooling_layer->CreateDataParam()));
    pooling_data->AllocateEmptyBlobs();
    pooling_data->in->set_shape(Shape(2, 3, 6, 5));
    pooling_layer->InitFromInputShape(pooling_data.get());

    EXPECT_EQ(pooling_data->out->shape().num(), pooling_data->in->shape().num());
    EXPECT_EQ(pooling_data->out->shape().channels(),
      pooling_data->in->shape().channels());
    EXPECT_EQ(pooling_data->out->shape().height(), 1);
    EXPECT_EQ(pooling_data->out->shape().width(), 1);
    EXPECT_EQ(pooling_data->out_diff->shape().num(), pooling_data->in->shape().num());
    EXPECT_EQ(pooling_data->out_diff->shape().channels(),
      pooling_data->in->shape().channels());
    EXPECT_EQ(pooling_data->out_diff->shape().height(), 1);
    EXPECT_EQ(pooling_data->out_diff->shape().width(), 1);
  }

  TYPED_TEST(CuDNNPoolingLayerTest, TestForwardMax) {
    TestForwardSquare();
    TestForwardRectHigh();
    TestForwardRectWide();
  }

  TYPED_TEST(CuDNNPoolingLayerTest, TestGradientMax) {
    typedef typename TypeParam Dtype;

    const int kNum = 2;
    const int kChannels = 2;
    std::shared_ptr<PoolingProto> proto_param =
      std::make_shared<PoolingProto>();
    proto_param->set_in(pooling_proto->in());
    proto_param->set_out(pooling_proto->out());
    proto_param->set_kernel_size(2);
    proto_param->set_pool(PoolingProto_PoolMethod_MAX);

    pooling_layer = std::make_shared<CuDNNPoolingLayer<Dtype>>(
      layer_name, proto_param->DebugString());
    pooling_layer->InitParam();
    pooling_data = std::shared_ptr<CuDNNPoolingData<Dtype>>(
      dynamic_cast<CuDNNPoolingData<Dtype>*>(
      pooling_layer->CreateDataParam()));
    pooling_data->AllocateEmptyBlobs();
    pooling_data->in->set_shape(Shape(kNum, kChannels, 3, 5));
    pooling_layer->InitFromInputShape(pooling_data.get());

    InitParameters();

    Dtype inputs_data_[kNum * kChannels * 3 * 5];
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < 15 * kNum * kChannels; i += 15) {
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

    CUDA_CHECK(cudaMemcpy(pooling_data->in->mutable_data(), inputs_data_,
      pooling_data->in->shape().count()*sizeof(Dtype), cudaMemcpyHostToDevice));

    GradientChecker<Dtype> gradient_checker(1e-2, 1e-2);
    gradient_checker.CheckGradientExhaustive(ctx, pooling_layer.get(),
      pooling_data.get(), pooling_model.get());
  }
  // TODO(xcdu) : setup forward and backward need to be tested

}  // namespace caffe
