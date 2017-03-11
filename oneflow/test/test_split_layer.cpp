#include <gtest/gtest.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "common/common.h"
#include "common/filler.h"
#include "common/shape.h"
#include "dag/blob_meta.h"
#include "layers/base_layer.h"
#include "layers/split_layer.h"
#include "memory/blob.h"

#include "test/test_job.h"
#include "test/test_gradient_check_util.h"
#include "test/test_proto_engine.h"

namespace caffe {
template <typename Dtype>
class SplitLayerTest : public ::testing::Test{
protected:
  SplitLayerTest() {
    RNG::set_seed(kSeed);
    cudaSetDevice(kDeviceId);
    cublasCreate(&ctx.cublas_handle);
    cudaStreamCreate(&ctx.cuda_stream);
    cublasSetStream(ctx.cublas_handle, ctx.cuda_stream);

    layer_name = "split1";
    split_proto = std::make_shared<SplitProto>(
      *dynamic_cast<const SplitProto*>(
      ProtoEngine<Dtype>::proto_param(layer_name)));
  }
  ~SplitLayerTest() {
    cublasDestroy(ctx.cublas_handle);
    cudaStreamDestroy(ctx.cuda_stream);
  }

  void InitParameters() {
    for (int i = 0; i < split_data->blob_names().size(); ++i) {
      (*split_data->name_to_blob_pptr().find(
        split_data->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
    split_model = std::shared_ptr<SplitModel<Dtype>>(
      dynamic_cast<splitModel<Dtype>*>(
      split_layer->CreateModelParam()));
    split_model->AllocateEmptyBlobs();
    split_model->AlignBlobShapes(
      *dynamic_cast<const SplitModel<Dtype>*>(
      split_layer->GetModelParam()));

    for (int i = 0; i < split_model->blob_names().size(); ++i) {
      (*split_model->name_to_blob_pptr().find(
        split_model->blob_names()[i])->second)->Reallocate(
        MemoryType::kDeviceMemory);
    }
  }

  ContextParam ctx;
  std::string layer_name;
  std::shared_ptr<SplitProto> split_proto;
  std::shared_ptr<SplitLayer<Dtype>> split_layer;
  std::shared_ptr<SplitData<Dtype>> split_data;
  std::shared_ptr<SplitModel<Dtype>> split_model;
};


TYPED_TEST_CASE(SplitLayerTest, TestDtypes);

TYPED_TEST(SplitLayerTest, TestSetup) {
  typedef typename TypeParam Dtype;

  const int out_num = 3;
  split_proto->set_out_num(out_num);
  
  split_layer = std::make_shared<SplitLayer<Dtype>>(
    layer_name, split_proto->DebugString());
  split_layer->InitParam();
  split_data = std::shared_ptr<SplitData<Dtype>>(
    dynamic_cast<SplitData<Dtype>*>(
    split_layer->CreateDataParam()));
  split_data->AllocateEmptyBlobs();
  split_data->in->set_shape(Shape(2, 3, 6, 5));
 
  split_layer->InitFromInputShape(split_data.get());


  EXPECT_EQ(split_data->out.size(), out_num);
  /*
  for (int i = 0; i < split_data->out.size(); ++i) {
    EXPECT_EQ(split_data->out->shape().num(), split_data->in->shape().num());
    EXPECT_EQ(split_data->out->shape().channels(),
      split_data->in->shape().channels());
    EXPECT_EQ(split_data->out->shape().height(), 3);
    EXPECT_EQ(split_data->out->shape().width(), 2);;
  }
  */
}

}