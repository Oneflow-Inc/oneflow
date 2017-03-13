#ifndef _LAYERS_POOLING_LAYER_H_
#define _LAYERS_POOLING_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class PoolingData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };
  Blob<Dtype>* idx{ nullptr };
  explicit PoolingData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
    DATA_REGISTER_BLOB(layer_name, idx, BlobType::kOther);
  }
};

template <typename Dtype>
class PoolingModel : public ModelParam<Dtype> {
public:
  explicit PoolingModel(const std::string& layer_name) {
  }
};

template <typename Dtype>
class PoolingParam : public LayerParam<Dtype> {
public:
  // Init from proto
  PoolingProto_PoolMethod pool_;
  bool global_pooling_;
  int32_t kernel_h_, kernel_w_;
  int32_t stride_h_, stride_w_;
  int32_t pad_h_, pad_w_;

  // Init from input shape
  int32_t channels_;
  int32_t height_, width_;
  int32_t pooled_height_, pooled_width_;

  explicit PoolingParam() {
  }
};

template <typename Dtype>
class PoolingLayer : public BaseLayer<Dtype> {
public:
  explicit PoolingLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new PoolingData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new PoolingModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  PoolingLayer(const PoolingLayer& other) = delete;
  PoolingLayer& operator=(const PoolingLayer& other) = delete;
};
}  // namespace caffe
#endif  // _LAYERS_POOLING_LAYER_H_