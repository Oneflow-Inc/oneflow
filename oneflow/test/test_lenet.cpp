#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "layers/base_layer.h"
#include "layers/convolution_layer.h"
#include "layers/innerproduct_layer.h"
#include "layers/multinomial_logistic_loss_layer.h"
#include "layers/pooling_layer.h"
#include "layers/relu_layer.h"
#include "layers/softmax_layer.h"

#include "test/test_job.h"
#include "test/test_lenet.h"
#include "test/test_proto_engine.h"

#define PRINT_LOG

namespace caffe {
#define DECLARE_LAYER(layer_type, layer_name) \
  std::string layer_name##_name = #layer_name;\
  std::shared_ptr<layer_type##Proto> layer_name##_proto;\
  std::shared_ptr<layer_type##Layer<Dtype>> layer_name##_layer;\
  std::shared_ptr<layer_type##Data<Dtype>> layer_name##_data;\
  std::shared_ptr<layer_type##Model<Dtype>> layer_name##_model

#define INIT_LAYER(layer_type, layer_name, in_blob_name, in_shape) \
    layer_name##_proto = std::make_shared<layer_type##Proto>(\
      *dynamic_cast<const layer_type##Proto*>(\
      ProtoEngine<Dtype>::proto_param(layer_name##_name)));\
    layer_name##_layer = std::make_shared<layer_type##Layer<Dtype>>(\
      layer_name##_name, layer_name##_proto->DebugString());\
    layer_name##_layer->InitParam();\
    layer_name##_data = std::shared_ptr<layer_type##Data<Dtype>>(\
      dynamic_cast<layer_type##Data<Dtype>*>(\
      layer_name##_layer->CreateDataParam()));\
    layer_name##_data->AllocateEmptyBlobs();\
    layer_name##_data->in_blob_name->set_shape(in_shape);\
    layer_name##_layer->InitFromInputShape(layer_name##_data.get())
#define ALOCATE_GPU_MEMORY(layer_type, layer_name) \
  do {\
    for (int i = 0; i < layer_name##_data->blob_names().size(); ++i) {\
      (*layer_name##_data->name_to_blob_pptr().find(\
        layer_name##_data->blob_names()[i])->second)->Reallocate(\
        MemoryType::kDeviceMemory);}\
    layer_name##_model = std::shared_ptr<layer_type##Model<Dtype>>(\
      dynamic_cast<layer_type##Model<Dtype>*>(\
      layer_name##_layer->CreateModelParam()));\
    layer_name##_model->AllocateEmptyBlobs();\
    layer_name##_model->AlignBlobShapes(\
      *dynamic_cast<const layer_type##Model<Dtype>*>(\
      layer_name##_layer->GetModelParam()));\
    for (int i = 0; i < layer_name##_model->blob_names().size(); ++i) {\
      (*layer_name##_model->name_to_blob_pptr().find(\
        layer_name##_model->blob_names()[i])->second)->Reallocate(\
        MemoryType::kDeviceMemory);}} while(0)

#define LINK_LAYERS(s_layer_name, s_blob_name, d_layer_name, d_blob_name) \
  do{\
    d_layer_name##_data->d_blob_name->set_data_ptr(\
      static_cast<void*>(const_cast<Dtype*>(\
      s_layer_name##_data->s_blob_name->data())), false);} while(0)

#define FORWARD(layer_name) \
  layer_name##_layer->Forward(ctx, layer_name##_data.get(),\
    layer_name##_model.get())

#define BACKWARD(layer_name) \
  layer_name##_layer->Backward(ctx, layer_name##_data.get(),\
    layer_name##_model.get())

#define ALLOCATE_HOST_MEMORY(layer_name) \
  Dtype* layer_name##_weight = (Dtype*) calloc(\
    layer_name##_model->weight->shape().count(),sizeof(Dtype));\
  Dtype* layer_name##_weight_tmp = (Dtype*) calloc(\
    layer_name##_model->weight->shape().count(),sizeof(Dtype));\
  Dtype* layer_name##_bias = (Dtype*) calloc(\
    layer_name##_model->bias->shape().count(),sizeof(Dtype));\
  Dtype* layer_name##_bias_tmp = (Dtype*) calloc(\
    layer_name##_model->bias->shape().count(),sizeof(Dtype))

#define ACCUMULATE_DIFF(layer_name) \
  do{\
    int32_t weight_count = layer_name##_model->weight->shape().count();\
    int32_t bias_count = layer_name##_model->bias->shape().count();\
    CUDA_CHECK(cudaMemcpy(layer_name##_weight_tmp,\
      layer_name##_model->weight_diff->data(),\
      weight_count*sizeof(Dtype), cudaMemcpyDeviceToHost));\
    CUDA_CHECK(cudaMemcpy(layer_name##_bias_tmp,\
      layer_name##_model->bias_diff->data(),\
      bias_count*sizeof(Dtype), cudaMemcpyDeviceToHost));\
    for(int i=0;i<weight_count;++i) {\
      layer_name##_weight[i] += layer_name##_weight_tmp[i]/weight_count;\
        }\
    for(int i=0;i<bias_count;++i) {\
      layer_name##_bias[i] += layer_name##_bias_tmp[i]/bias_count;\
        }\
    }while(0)

#define UPDATE_DIFF(layer_name) \
  do{\
    int32_t weight_count = layer_name##_model->weight->shape().count();\
    int32_t bias_count = layer_name##_model->bias->shape().count();\
    CUDA_CHECK(cudaMemcpy(layer_name##_weight_tmp,\
      layer_name##_model->weight->data(),\
      weight_count*sizeof(Dtype), cudaMemcpyDeviceToHost));\
    CUDA_CHECK(cudaMemcpy(layer_name##_bias_tmp, \
      layer_name##_model->bias->data(),\
      bias_count*sizeof(Dtype), cudaMemcpyDeviceToHost));\
    for(int i=0;i<weight_count;++i) {\
      layer_name##_weight_tmp[i] -= lr*layer_name##_weight[i];\
    }\
    for(int i=0;i<bias_count;++i) {\
      layer_name##_bias_tmp[i] -= lr*layer_name##_bias[i];\
     }\
    CUDA_CHECK(cudaMemcpy(layer_name##_model->weight->mutable_data(),\
      layer_name##_weight_tmp, weight_count*sizeof(Dtype),\
      cudaMemcpyHostToDevice));\
    CUDA_CHECK(cudaMemcpy(layer_name##_model->bias->mutable_data(),\
      layer_name##_bias_tmp, bias_count*sizeof(Dtype),\
      cudaMemcpyHostToDevice));\
    memset(layer_name##_weight,0,weight_count*sizeof(Dtype));\
    memset(layer_name##_bias,0,bias_count*sizeof(Dtype));\
  }while(0)

template <typename Dtype>
class LeNetTest : public ::testing::Test{};

template <typename Dtype>
class TrainLeNetTest : public LeNetTest<Dtype> {
protected:
  TrainLeNetTest() {

    Mnist<Dtype>::test();

    cudaSetDevice(kDeviceId);
    cublasCreate(&ctx.cublas_handle);
    cudaStreamCreate(&ctx.cuda_stream);
    cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);

    batch_size = 100;
    channel = 1;
    height = 28;
    width = 28;

    solver_proto = std::make_shared<SolverProto>(
      ProtoEngine<Dtype>::solver_proto());
    test_iter = solver_proto->test_iter().Get(0);
    test_interval = solver_proto->test_interval();
    max_iter = solver_proto->max_iter();
    base_lr = solver_proto->base_lr();
    momentum = solver_proto->momentum();
    weight_decay = solver_proto->weight_decay();
    gamma = solver_proto->gamma();
    power = solver_proto->power();

    //std::cout << test_iter << std::endl;
    //std::cout << test_interval << std::endl;
    //std::cout << max_iter << std::endl;
    //std::cout << base_lr << std::endl;
    //std::cout << momentum << std::endl;
    //std::cout << weight_decay << std::endl;
    //std::cout << gamma << std::endl;
    //std::cout << power << std::endl;
    
    //layer_names = ProtoEngine<Dtype>::layer_names();
    //layer_names = { "mnist", "conv1", "pool1", "conv2", "pool2", "ip1",
    //  "relu1", "ip2", "softmax", "loss" };

    // Initiate data's and model's shapes of each layers
    INIT_LAYER(Convolution, conv1, in,
      Shape(batch_size, channel, height, width));
    INIT_LAYER(Pooling, pool1, in, conv1_data->out->shape());
    INIT_LAYER(Convolution, conv2, in, pool1_data->out->shape());
    INIT_LAYER(Pooling, pool2, in, conv2_data->out->shape());
    INIT_LAYER(InnerProduct, ip1, in, pool2_data->out->shape());
    INIT_LAYER(ReLU, relu1, in, ip1_data->out->shape());
    INIT_LAYER(InnerProduct, ip2, in, relu1_data->out->shape());
    INIT_LAYER(Softmax, softmax, in, ip2_data->out->shape());
    loss_proto = std::make_shared<MultinomialLogisticLossProto>(
      *dynamic_cast<const MultinomialLogisticLossProto*>(
      ProtoEngine<Dtype>::proto_param(loss_name)));
    loss_layer = std::make_shared<MultinomialLogisticLossLayer<Dtype>>(
      loss_name, loss_proto->DebugString());
    loss_layer->InitParam();
    loss_data = std::shared_ptr<MultinomialLogisticLossData<Dtype>>(
      dynamic_cast<MultinomialLogisticLossData<Dtype>*>(
      loss_layer->CreateDataParam()));
    loss_data->AllocateEmptyBlobs();
    loss_data->data->set_shape(softmax_data->out->shape());
    loss_data->label->set_shape(Shape(softmax_data->out->shape().num(),1));
    loss_layer->InitFromInputShape(loss_data.get());

    // allocate the memory of each layers
    ALOCATE_GPU_MEMORY(Convolution, conv1);
    ALOCATE_GPU_MEMORY(Pooling, pool1);
    ALOCATE_GPU_MEMORY(Convolution, conv2);
    ALOCATE_GPU_MEMORY(Pooling, pool2);
    ALOCATE_GPU_MEMORY(InnerProduct, ip1);
    ALOCATE_GPU_MEMORY(ReLU, relu1);
    ALOCATE_GPU_MEMORY(InnerProduct, ip2);
    ALOCATE_GPU_MEMORY(Softmax, softmax);
    ALOCATE_GPU_MEMORY(MultinomialLogisticLoss, loss);
    //for (auto a : conv1_data->blob_names()) {
    //  std::cout << a << std::endl;
    //}
    LINK_LAYERS(conv1, out, pool1, in);
    LINK_LAYERS(pool1, out, conv2, in);
    LINK_LAYERS(conv2, out, pool2, in);
    LINK_LAYERS(pool2, out, ip1, in);
    LINK_LAYERS(ip1, out, relu1, in);
    LINK_LAYERS(relu1, out, ip2, in);
    LINK_LAYERS(ip2, out, softmax, in);
    LINK_LAYERS(softmax, out, loss, data);

    // Initiate the model
    FillerParameter xavier_filler_param;
    FillerParameter constant_filler_0_param;
    FillerParameter constant_filler_1_param;
    constant_filler_0_param.set_value(0);
    constant_filler_1_param.set_value(1);
    std::shared_ptr<XavierFiller<Dtype>> xavier_filler =
      std::make_shared<XavierFiller<Dtype>>(xavier_filler_param);
    std::shared_ptr<ConstantFiller<Dtype>> constant_filler_0 =
      std::make_shared<ConstantFiller<Dtype>>(constant_filler_0_param);
    std::shared_ptr<ConstantFiller<Dtype>> constant_filler_1 =
      std::make_shared<ConstantFiller<Dtype>>(constant_filler_1_param);

    xavier_filler->fill(conv1_model->weight);
    constant_filler_0->fill(conv1_model->bias);
    constant_filler_1->fill(conv1_model->bias_multiplier);

    xavier_filler->fill(conv2_model->weight);
    constant_filler_0->fill(conv2_model->bias);
    constant_filler_1->fill(conv2_model->bias_multiplier);

    xavier_filler->fill(ip1_model->weight);
    constant_filler_0->fill(ip1_model->bias);
    constant_filler_1->fill(ip1_model->bias_multiplier);

    xavier_filler->fill(ip2_model->weight);
    constant_filler_0->fill(ip2_model->bias);
    constant_filler_1->fill(ip2_model->bias_multiplier);

    constant_filler_1->fill(loss_model->loss_multiplier);

  }
  ~TrainLeNetTest() {
    cublasDestroy(ctx.cublas_handle);
    cudaStreamDestroy(ctx.cuda_stream);
  }
  int32_t test_iter;
  int32_t test_interval;
  int32_t max_iter;
  Dtype base_lr;
  Dtype momentum;
  Dtype weight_decay;
  Dtype gamma;
  Dtype power;

  size_t batch_size;
  size_t channel;
  size_t height;
  size_t width;

  ContextParam ctx;
  std::shared_ptr<SolverProto> solver_proto;
  //std::vector<std::string> layer_names;
  //std::vector<std::shared_ptr<BaseLayer<Dtype>>> layers;
  DECLARE_LAYER(Convolution, conv1);
  DECLARE_LAYER(Pooling, pool1);
  DECLARE_LAYER(Convolution, conv2);
  DECLARE_LAYER(Pooling, pool2);
  DECLARE_LAYER(InnerProduct, ip1);
  DECLARE_LAYER(ReLU, relu1);
  DECLARE_LAYER(InnerProduct, ip2);
  DECLARE_LAYER(Softmax, softmax);
  DECLARE_LAYER(MultinomialLogisticLoss, loss);

};

template <typename Dtype>
class TestLeNetTest : public LeNetTest<Dtype> {

};

TYPED_TEST_CASE(TrainLeNetTest, TestDtypes);
TYPED_TEST_CASE(TestLeNetTest, TestDtypes);

TYPED_TEST(TrainLeNetTest, TestRead) {
  typedef typename TypeParam Dtype;
  int iter = 0;
  //max_iter = 20;
  for (int iter = 0; iter < max_iter; ++iter) {
    Dtype lr = base_lr*std::pow(1 + gamma*iter, -power);
    ALLOCATE_HOST_MEMORY(conv1);
    ALLOCATE_HOST_MEMORY(conv2);
    ALLOCATE_HOST_MEMORY(ip1);
    ALLOCATE_HOST_MEMORY(ip2);

    int32_t softmax_out_count = softmax_data->out->shape().count();
    int32_t softmax_out_num = softmax_data->out->shape().num();
    int32_t softmax_out_dim = softmax_out_count / softmax_out_num;
    Dtype* softmax_out = (Dtype*)calloc(softmax_out_count, sizeof(Dtype));
#ifdef PRINT_LOG
    printf("Iteration:%d begining... Learing rate:%f\n\n", iter, lr);
#endif
    for (int batch_num = 0; batch_num <
      Mnist<Dtype>::train_image()->image_number / batch_size; ++batch_num) {
      CUDA_CHECK(cudaMemcpy(conv1_data->in->mutable_data(),
        Mnist<Dtype>::train_image()->memory_pool +
        batch_num*batch_size*channel*height*width,
        conv1_data->in->shape().count()*sizeof(Dtype), cudaMemcpyHostToDevice));
      //for (int i = 0; i < batch_size*channel*height*width; ++i){
      //  printf("%f ", Mnist<Dtype>::train_image_memory()[i]);
      //}
      //printf("\n");
      CUDA_CHECK(cudaMemcpy(loss_data->label->mutable_data(),
        Mnist<Dtype>::train_label()->memory_pool + batch_num*batch_size,
        loss_data->label->shape().count()*sizeof(Dtype),
        cudaMemcpyHostToDevice));
      //for (int i = 0; i < batch_size; i++) {
      //  printf("%f ", Mnist<Dtype>::train_label_memory()[i]);
      //}
      //printf("\n");
      //printf("conv1_data:%d %d %d %d\nconv1_weight:%d %d %d %d\n",
      //  conv1_data->in->shape().num(), conv1_data->in->shape().channels(), conv1_data->in->shape().height(), conv1_data->in->shape().width(),
      //  conv1_model->weight->shape().num(), conv1_model->weight->shape().channels(), conv1_model->weight->shape().height(), conv1_model->weight->shape().width());
      //printf("softmax_out:%d %d\n", softmax_data->out->shape().num(), softmax_data->out->shape().dim());
      //printf("loss_data:%d %d\nloss_label:%d %d\nloss_buffer:%d %d\nloss_multiplier:%d %d\nloss_out:%d\n",
      //  loss_data->data->shape().num(), loss_data->data->shape().dim(),
      //  loss_data->label->shape().num(), loss_data->label->shape().dim(),
      //  loss_data->loss_buffer->shape().num(),loss_data->loss_buffer->shape().dim(),
      //  loss_model->loss_multiplier->shape().num(),loss_model->loss_multiplier->shape().dim(),
      //  loss_data->loss->shape().count());
#ifdef PRINT_LOG
      printf("Training fowarding...\n");
#endif
      FORWARD(conv1);
      FORWARD(pool1);
      FORWARD(conv2);
      FORWARD(pool2);
      FORWARD(ip1);
      FORWARD(relu1);
      FORWARD(ip2);
      FORWARD(softmax);
      cublasSetPointerMode(ctx.cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
      FORWARD(loss);

#ifdef PRINT_LOG
      printf("Train backwarding...\n");
#endif
      Dtype loss;
      CUDA_CHECK(cudaMemcpy(&loss, loss_data->loss->data(),
        loss_data->loss->shape().count()*sizeof(Dtype),
        cudaMemcpyDeviceToHost));
#ifdef PRINT_LOG
      printf("\nIteration:%d Batch_num:%d Loss:%f\n\n",iter, batch_num, loss);
#endif
      BACKWARD(loss);
      cublasSetPointerMode(ctx.cublas_handle, CUBLAS_POINTER_MODE_HOST);
      BACKWARD(softmax);
      BACKWARD(ip2);
      BACKWARD(relu1);
      BACKWARD(ip1);
      BACKWARD(pool2);
      BACKWARD(conv2);
      BACKWARD(pool1);
      BACKWARD(conv1);
#ifdef PRINT_LOG
      printf("Accumulating error signal...\n");
#endif
      ACCUMULATE_DIFF(conv1);
      ACCUMULATE_DIFF(conv2);
      ACCUMULATE_DIFF(ip1);
      ACCUMULATE_DIFF(ip2);
    }
#ifdef PRINT_LOG
    printf("Updating error signal...\n");
#endif
    UPDATE_DIFF(conv1);
    UPDATE_DIFF(conv2);
    UPDATE_DIFF(ip1);
    UPDATE_DIFF(ip2);

    int cnt = 0;
    //printf("%d %d\n", Mnist<Dtype>::test_image()->image_number,Mnist<Dtype>::test_label()->items);
    for (int batch_num = 0; batch_num < Mnist<Dtype>::test_image()->image_number/batch_size;
      ++batch_num) {

      CUDA_CHECK(cudaMemcpy(conv1_data->in->mutable_data(),
        Mnist<Dtype>::test_image()->memory_pool +
        batch_num*batch_size*channel*height*width,
        conv1_data->in->shape().count()*sizeof(Dtype), cudaMemcpyHostToDevice));

      CUDA_CHECK(cudaMemcpy(loss_data->label->mutable_data(),
        Mnist<Dtype>::test_label()->memory_pool + batch_num*batch_size,
        loss_data->label->shape().count()*sizeof(Dtype),
        cudaMemcpyHostToDevice));
#ifdef PRINT_LOG
      printf("Test_images batch_number:%d testing...\n", batch_num);
#endif
      FORWARD(conv1);
      FORWARD(pool1);
      FORWARD(conv2);
      FORWARD(pool2);
      FORWARD(ip1);
      FORWARD(relu1);
      FORWARD(ip2);
      FORWARD(softmax);

      CUDA_CHECK(cudaMemcpy(softmax_out, softmax_data->out->data(),
        softmax_out_count*sizeof(Dtype), cudaMemcpyDeviceToHost));


      for (int i = 0; i < softmax_out_num; ++i) {
        int32_t idx = -1;
        Dtype maxval = -FLT_MAX;
        for (int j = 0; j<softmax_out_dim; ++j) {
          if (softmax_out[i*softmax_out_dim+j] > maxval){
            maxval = softmax_out[i*softmax_out_dim + j];
            idx = j;
          }
        }
        if (Mnist<Dtype>::test_label()->memory_pool[batch_num*batch_size + i] == idx) {
          cnt++;
        }
      }
    }
#ifdef PRINT_LOG
    printf("\nIteration:%d Accuracy:%f\n\n", iter, (Dtype)cnt / Mnist<Dtype>::test_label()->items);
#endif
    free(softmax_out);
  }
}

TYPED_TEST(TrainLeNetTest, Test) {
  typedef typename TypeParam Dtype;
}

}  // namespace caffe
