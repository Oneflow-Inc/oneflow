#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "dag/blob_meta.h"
#include "layers/base_layer.h"
#include "layers/convolution_layer.h"
#include "memory/blob.h"

#include "test/test_gradient_check_util.h"
#include "test/test_job.h"
#include "test/test_proto_engine.h"

namespace caffe {
// Simulate the computing of convolution on CPU to compare with the result that
// GPU returns.
template <typename Dtype>
void caffe_conv(Blob<Dtype>* in,
  std::shared_ptr<ConvolutionProto> conv_param, Blob<Dtype>* weight,
  Blob<Dtype>* bias, Blob<Dtype>* out) {
  const bool has_depth = (out->shape().num_axes() == 5);
  if (!has_depth) { CHECK_EQ(4, out->shape().num_axes()); }
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_h() || conv_param->has_kernel_w()) {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  }
  else {
    kernel_h = kernel_w = conv_param->kernel_size();
  }
  int pad_h, pad_w;
  if (conv_param->has_pad_h() || conv_param->has_pad_w()) {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  }
  else {
    pad_h = pad_w = conv_param->pad() ? conv_param->pad() : 0;
  }
  int stride_h, stride_w;
  if (conv_param->has_stride_h() || conv_param->has_stride_w()) {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  }
  else {
    stride_h = stride_w = conv_param->stride() ? conv_param->stride() : 1;
  }
  int kernel_d, pad_d, stride_d;
  if (has_depth) {
    kernel_d = kernel_h;
    stride_d = stride_h;
    pad_d = pad_h;
  }
  else {
    kernel_d = stride_d = 1;
    pad_d = 0;
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->shape().shape(1) / groups;
  int k_g = in->shape().shape(1) / groups;
  int o_head, k_head;
  // Convolution
  std::vector<int> weight_offset(4 + has_depth);
  std::vector<int> in_offset(4 + has_depth);
  std::vector<int> out_offset(4 + has_depth);
  Dtype* out_data = out->mutable_data();
  for (int n = 0; n < out->shape().shape(0); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int z = 0; z < (has_depth ? out->shape().shape(2) : 1); z++) {
            for (int y = 0; y < out->shape().shape(2 + has_depth); y++) {
              for (int x = 0; x < out->shape().shape(3 + has_depth); x++) {
                for (int r = 0; r < kernel_d; r++) {
                  for (int p = 0; p < kernel_h; p++) {
                    for (int q = 0; q < kernel_w; q++) {
                      int in_z = z * stride_d - pad_d + r;
                      int in_y = y * stride_h - pad_h + p;
                      int in_x = x * stride_w - pad_w + q;
                      if (in_z >= 0
                        && in_z < (has_depth ? in->shape().shape(2) : 1)
                        && in_y >= 0 && in_y < in->shape().shape(2 + has_depth)
                        && in_x >= 0
                        && in_x < in->shape().shape(3 + has_depth)) {
                        weight_offset[0] = o + o_head;
                        weight_offset[1] = k;
                        if (has_depth) { weight_offset[2] = r; }
                        weight_offset[2 + has_depth] = p;
                        weight_offset[3 + has_depth] = q;
                        in_offset[0] = n;
                        in_offset[1] = k + k_head;
                        if (has_depth) { in_offset[2] = in_z; }
                        in_offset[2 + has_depth] = in_y;
                        in_offset[3 + has_depth] = in_x;
                        out_offset[0] = n;
                        out_offset[1] = o + o_head;
                        if (has_depth) { out_offset[2] = z; }
                        out_offset[2 + has_depth] = y;
                        out_offset[3 + has_depth] = x;
                        out_data[out->shape().offset(out_offset[0],
                          out_offset[1], out_offset[2], out_offset[3])] +=
                          in->data()[in->shape().offset(in_offset[0],
                          in_offset[1], in_offset[2], in_offset[3])] *
                          weight->data()[weight->shape().offset(
                          weight_offset[0], weight_offset[1], weight_offset[2],
                          weight_offset[3])];
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = bias->data();
    for (int n = 0; n < out->shape().shape(0); n++) {
      for (int o = 0; o < out->shape().shape(1); o++) {
        for (int z = 0; z < (has_depth ? out->shape().shape(2) : 1); z++) {
          for (int y = 0; y < out->shape().shape(2 + has_depth); y++) {
            for (int x = 0; x < out->shape().shape(3 + has_depth); x++) {
              out_offset[0] = n;
              out_offset[1] = o;
              if (has_depth) { out_offset[2] = z; }
              out_offset[2 + has_depth] = y;
              out_offset[3 + has_depth] = x;
              out_data[out->shape().offset(out_offset[0], out_offset[1],
                out_offset[2], out_offset[3])] += bias_data[o];
            }
          }
        }
      }
    }
  }
}
template void caffe_conv(Blob<float>* in,
  std::shared_ptr<ConvolutionProto> conv_param,
  Blob<float>* weight, Blob<float>* bias,
  Blob<float>* out);
template void caffe_conv(Blob<double>* in,
  std::shared_ptr<ConvolutionProto> conv_param,
  Blob<double>* weight, Blob<double>* bias,
  Blob<double>* out);

template<typename Dtype>
class ConvolutionLayerTest :public ::testing::Test{
protected:
  ConvolutionLayerTest() {
    RNG::set_seed(kSeed);
    cudaSetDevice(kDeviceId);
    cublasCreate(&ctx.cublas_handle);
    cudaStreamCreate(&ctx.cuda_stream);
    cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);
    layer_name = "conv1";
    convolution_proto = std::make_shared<ConvolutionProto>(
      *dynamic_cast<const ConvolutionProto*>(
      ProtoEngine<Dtype>::proto_param(layer_name)));
  }
  ~ConvolutionLayerTest() {
    cublasDestroy(ctx.cublas_handle);
    cudaStreamDestroy(ctx.cuda_stream);
  }

  void InitParameters() {
    for (int i = 0; i < convolution_data->blob_names().size(); ++i) {
      (*convolution_data->name_to_blob_pptr().find(
        convolution_data->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
    convolution_model = std::shared_ptr<ConvolutionModel<Dtype>>(
      dynamic_cast<ConvolutionModel<Dtype>*>(
      convolution_layer->CreateModelParam()));
    convolution_model->AllocateEmptyBlobs();
    convolution_model->AlignBlobShapes(
      *dynamic_cast<const ConvolutionModel<Dtype>*>(
      convolution_layer->GetModelParam()));

    for (int i = 0; i < convolution_model->blob_names().size(); ++i) {
      (*convolution_model->name_to_blob_pptr().find(
        convolution_model->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }

    FillerParameter gaussian_filler_param;
    gaussian_filler_param.set_mean(1);
    gaussian_filler_param.set_std(0.5);
    std::shared_ptr<GaussianFiller<Dtype>> gaussian_filler =
      std::make_shared<GaussianFiller<Dtype>>(gaussian_filler_param);
    gaussian_filler->fill(convolution_data->in);
    gaussian_filler->fill(convolution_model->weight);
    gaussian_filler->fill(convolution_model->bias);

    FillerParameter constant_filler_param;
    constant_filler_param.set_value(1);
    std::shared_ptr<ConstantFiller<Dtype>> constant_filler =
      std::make_shared<ConstantFiller<Dtype>>(constant_filler_param);
    constant_filler->fill(convolution_model->bias_multiplier);
  }
  ContextParam ctx;
  std::string layer_name;
  std::shared_ptr<ConvolutionProto> convolution_proto;
  std::shared_ptr<ConvolutionLayer<Dtype>> convolution_layer;
  std::shared_ptr<ConvolutionData<Dtype>> convolution_data;
  std::shared_ptr<ConvolutionModel<Dtype>> convolution_model;
};

TYPED_TEST_CASE(ConvolutionLayerTest, TestDtypes);

TYPED_TEST(ConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam Dtype;

  convolution_proto->set_kernel_size(3);
  convolution_proto->set_stride(2);
  convolution_proto->set_num_output(4);

  convolution_layer = std::make_shared<ConvolutionLayer<Dtype>>(
    layer_name, convolution_proto->DebugString());
  convolution_layer->InitParam();
  convolution_data = std::shared_ptr<ConvolutionData<Dtype>>(
    dynamic_cast<ConvolutionData<Dtype>*>(
    convolution_layer->CreateDataParam()));
  convolution_data->AllocateEmptyBlobs();
  convolution_data->in->set_shape(Shape(2, 3, 6, 4));
  convolution_data->in_diff->set_shape(Shape(2, 3, 6, 4));
  convolution_layer->InitFromInputShape(convolution_data.get());

  EXPECT_EQ(convolution_data->out->shape().num(), 2);
  EXPECT_EQ(convolution_data->out->shape().channels(), 4);
  EXPECT_EQ(convolution_data->out->shape().height(), 2);
  EXPECT_EQ(convolution_data->out->shape().width(), 1);

  convolution_proto->set_num_output(3);
  convolution_proto->set_group(3);
  convolution_layer = std::make_shared<ConvolutionLayer<Dtype>>(
    layer_name, convolution_proto->DebugString());
  convolution_layer->InitParam();
  convolution_layer->InitFromInputShape(convolution_data.get());

  EXPECT_EQ(convolution_data->out->shape().num(), 2);
  EXPECT_EQ(convolution_data->out->shape().channels(), 3);
  EXPECT_EQ(convolution_data->out->shape().height(), 2);
  EXPECT_EQ(convolution_data->out->shape().width(), 1);
  EXPECT_EQ(convolution_data->out_diff->shape().num(), 2);
  EXPECT_EQ(convolution_data->out_diff->shape().channels(), 3);
  EXPECT_EQ(convolution_data->out_diff->shape().height(), 2);
  EXPECT_EQ(convolution_data->out_diff->shape().width(), 1);
}
TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolution) {
  typedef typename TypeParam Dtype;

  convolution_proto = std::make_shared<ConvolutionProto>(
    *dynamic_cast<const ConvolutionProto*>(
    ProtoEngine<Dtype>::proto_param(layer_name)));
  convolution_proto->set_kernel_size(2);
  convolution_proto->set_stride(2);
  convolution_proto->set_num_output(3);

  convolution_layer = std::make_shared<ConvolutionLayer<Dtype>>(
    layer_name, convolution_proto->DebugString());
  convolution_layer->InitParam();
  convolution_data = std::shared_ptr<ConvolutionData<Dtype>>(
    dynamic_cast<ConvolutionData<Dtype>*>(
    convolution_layer->CreateDataParam()));
  convolution_data->AllocateEmptyBlobs();
  convolution_data->in->set_shape(Shape(2, 3, 6, 4));
  convolution_layer->InitFromInputShape(convolution_data.get());

  InitParameters();

  convolution_layer->Forward(ctx, convolution_data.get(),
    convolution_model.get());

  std::shared_ptr<Dtype> in_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->in->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> out_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> refer_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> weight_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_model->weight->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> bias_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_model->bias->shape().count(),
    sizeof(Dtype))));
  // copy from device to host
  CUDA_CHECK(cudaMemcpy(in_data_.get(), convolution_data->in->data(),
    convolution_data->in->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_data_.get(), convolution_data->out->data(),
    convolution_data->out->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(weight_data_.get(), convolution_model->weight->data(),
    convolution_model->weight->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(bias_data_.get(), convolution_model->bias->data(),
    convolution_model->bias->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  // encapsulation
  std::shared_ptr<Blob<Dtype>> in_ = std::make_shared<Blob<Dtype>>(
    in_data_.get(), convolution_data->in->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> refer_ = std::make_shared<Blob<Dtype>>(
    refer_data_.get(), convolution_data->out->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> wei_ = std::make_shared<Blob<Dtype>>(
    weight_data_.get(), convolution_model->weight->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> bi_ = std::make_shared<Blob<Dtype>>(
    bias_data_.get(), convolution_model->bias->shape(),
    MemoryType::kHostPageableMemory);
  // convolution
  caffe_conv(in_.get(), convolution_proto, wei_.get(), bi_.get(), refer_.get());
  // check
  for (int k = 0; k < convolution_data->out->shape().count(); ++k) {
    EXPECT_NEAR(out_data_.get()[k], refer_data_.get()[k], 1e-5);
  }
}
TYPED_TEST(ConvolutionLayerTest, Test1x1Convolution) {
  typedef typename TypeParam Dtype;

  convolution_proto = std::make_shared<ConvolutionProto>(
    *dynamic_cast<const ConvolutionProto*>(
    ProtoEngine<Dtype>::proto_param(layer_name)));
  convolution_proto->set_kernel_size(1);
  convolution_proto->set_stride(1);
  convolution_proto->set_num_output(4);

  convolution_layer = std::make_shared<ConvolutionLayer<Dtype>>(
    layer_name, convolution_proto->DebugString());
  convolution_layer->InitParam();
  convolution_data = std::shared_ptr<ConvolutionData<Dtype>>(
    dynamic_cast<ConvolutionData<Dtype>*>(
    convolution_layer->CreateDataParam()));
  convolution_data->AllocateEmptyBlobs();
  convolution_data->in->set_shape(Shape(2, 3, 6, 4));
  convolution_layer->InitFromInputShape(convolution_data.get());

  InitParameters();

  convolution_layer->Forward(ctx, convolution_data.get(),
    convolution_model.get());

  std::shared_ptr<Dtype> in_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->in->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> out_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> refer_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> weight_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_model->weight->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> bias_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_model->bias->shape().count(),
    sizeof(Dtype))));
  // copy from device to host
  CUDA_CHECK(cudaMemcpy(in_data_.get(), convolution_data->in->data(),
    convolution_data->in->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_data_.get(), convolution_data->out->data(),
    convolution_data->out->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(weight_data_.get(), convolution_model->weight->data(),
    convolution_model->weight->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(bias_data_.get(), convolution_model->bias->data(),
    convolution_model->bias->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  // encapsulation
  std::shared_ptr<Blob<Dtype>> in_ = std::make_shared<Blob<Dtype>>(
    in_data_.get(), convolution_data->in->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> refer_ = std::make_shared<Blob<Dtype>>(
    refer_data_.get(), convolution_data->out->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> wei_ = std::make_shared<Blob<Dtype>>(
    weight_data_.get(), convolution_model->weight->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> bi_ = std::make_shared<Blob<Dtype>>(
    bias_data_.get(), convolution_model->bias->shape(),
    MemoryType::kHostPageableMemory);
  // convolution
  caffe_conv(in_.get(), convolution_proto, wei_.get(), bi_.get(), refer_.get());
  // check
  for (int k = 0; k < convolution_data->out->shape().count(); ++k) {
    EXPECT_NEAR(out_data_.get()[k], refer_data_.get()[k], 1e-5);
  }
}
TYPED_TEST(ConvolutionLayerTest, TestSimpleConvolutionGruop) {
  typedef typename TypeParam Dtype;

  convolution_proto = std::make_shared<ConvolutionProto>(
    *dynamic_cast<const ConvolutionProto*>(
    ProtoEngine<Dtype>::proto_param(layer_name)));
  convolution_proto->set_kernel_size(2);
  convolution_proto->set_stride(2);
  convolution_proto->set_num_output(3);
  convolution_proto->set_group(3);

  convolution_layer = std::make_shared<ConvolutionLayer<Dtype>>(
    layer_name, convolution_proto->DebugString());
  convolution_layer->InitParam();
  convolution_data = std::shared_ptr<ConvolutionData<Dtype>>(
    dynamic_cast<ConvolutionData<Dtype>*>(
    convolution_layer->CreateDataParam()));
  convolution_data->AllocateEmptyBlobs();
  convolution_data->in->set_shape(Shape(2, 3, 6, 4));
  convolution_layer->InitFromInputShape(convolution_data.get());

  InitParameters();

  convolution_layer->Forward(ctx, convolution_data.get(),
    convolution_model.get());

  std::shared_ptr<Dtype> in_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->in->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> out_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> refer_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_data->out->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> weight_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_model->weight->shape().count(),
    sizeof(Dtype))));
  std::shared_ptr<Dtype> bias_data_ = std::shared_ptr<Dtype>(
    reinterpret_cast<Dtype*>(calloc(convolution_model->bias->shape().count(),
    sizeof(Dtype))));
  // copy from device to host
  CUDA_CHECK(cudaMemcpy(in_data_.get(), convolution_data->in->data(),
    convolution_data->in->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out_data_.get(), convolution_data->out->data(),
    convolution_data->out->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(weight_data_.get(), convolution_model->weight->data(),
    convolution_model->weight->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(bias_data_.get(), convolution_model->bias->data(),
    convolution_model->bias->shape().count()*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  // encapsulation
  std::shared_ptr<Blob<Dtype>> in_ = std::make_shared<Blob<Dtype>>(
    in_data_.get(), convolution_data->in->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> refer_ = std::make_shared<Blob<Dtype>>(
    refer_data_.get(), convolution_data->out->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> wei_ = std::make_shared<Blob<Dtype>>(
    weight_data_.get(), convolution_model->weight->shape(),
    MemoryType::kHostPageableMemory);
  std::shared_ptr<Blob<Dtype>> bi_ = std::make_shared<Blob<Dtype>>(
    bias_data_.get(), convolution_model->bias->shape(),
    MemoryType::kHostPageableMemory);
  // convolution
  caffe_conv(in_.get(), convolution_proto, wei_.get(), bi_.get(), refer_.get());
  // check
  for (int k = 0; k < convolution_data->out->shape().count(); ++k) {
    EXPECT_NEAR(out_data_.get()[k], refer_data_.get()[k], 1e-5);
  }
}

TYPED_TEST(ConvolutionLayerTest, TestGradient) {
  typedef typename TypeParam Dtype;

  convolution_proto->set_kernel_size(2);
  convolution_proto->set_stride(2);
  convolution_proto->set_num_output(3);

  convolution_layer = std::make_shared<ConvolutionLayer<Dtype>>(
    layer_name, convolution_proto->DebugString());
  convolution_layer->InitParam();
  convolution_data = std::shared_ptr<ConvolutionData<Dtype>>(
    dynamic_cast<ConvolutionData<Dtype>*>(
    convolution_layer->CreateDataParam()));
  convolution_data->AllocateEmptyBlobs();
  convolution_data->in->set_shape(Shape(2, 3, 6, 4));

  convolution_layer->InitFromInputShape(convolution_data.get());

  InitParameters();

  GradientChecker<Dtype> gradient_checker(1e-2, 1e-2);
  gradient_checker.CheckGradientExhaustive(ctx, convolution_layer.get(),
    convolution_data.get(), convolution_model.get());
}

TYPED_TEST(ConvolutionLayerTest, Test1x1Gradient) {
  typedef typename TypeParam Dtype;

  convolution_proto->set_kernel_size(1);
  convolution_proto->set_stride(1);
  convolution_proto->set_num_output(2);

  convolution_layer = std::make_shared<ConvolutionLayer<Dtype>>(
    layer_name, convolution_proto->DebugString());
  convolution_layer->InitParam();
  convolution_data = std::shared_ptr<ConvolutionData<Dtype>>(
    dynamic_cast<ConvolutionData<Dtype>*>(
    convolution_layer->CreateDataParam()));
  convolution_data->AllocateEmptyBlobs();
  convolution_data->in->set_shape(Shape(2, 3, 6, 4));

  convolution_layer->InitFromInputShape(convolution_data.get());

  InitParameters();

  GradientChecker<Dtype> gradient_checker(1e-2, 1e-2);
  gradient_checker.CheckGradientExhaustive(ctx, convolution_layer.get(),
    convolution_data.get(), convolution_model.get());
}


TYPED_TEST(ConvolutionLayerTest, TestGradientGroup) {
  typedef typename TypeParam Dtype;

  convolution_proto->set_kernel_size(2);
  convolution_proto->set_stride(2);
  convolution_proto->set_num_output(3);
  convolution_proto->set_group(3);

  convolution_layer = std::make_shared<ConvolutionLayer<Dtype>>(
    layer_name, convolution_proto->DebugString());
  convolution_layer->InitParam();
  convolution_data = std::shared_ptr<ConvolutionData<Dtype>>(
    dynamic_cast<ConvolutionData<Dtype>*>(
    convolution_layer->CreateDataParam()));
  convolution_data->AllocateEmptyBlobs();
  convolution_data->in->set_shape(Shape(2, 3, 6, 4));

  convolution_layer->InitFromInputShape(convolution_data.get());

  InitParameters();

  GradientChecker<Dtype> gradient_checker(1e-2, 1e-2);
  gradient_checker.CheckGradientExhaustive(ctx, convolution_layer.get(),
    convolution_data.get(), convolution_model.get());
}
}  // namespace caffe
