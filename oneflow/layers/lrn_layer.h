#ifndef _LAYERS_LRN_LAYER_H_
#define _LAYERS_LRN_LAYER_H_

#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {
template <typename Dtype>
class LRNData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };
  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
  Blob<Dtype>* scale{ nullptr };
  explicit LRNData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
    DATA_REGISTER_BLOB(layer_name, scale, BlobType::kOther);
  }
};

template <typename Dtype>
class LRNModel : public ModelParam<Dtype> {
public:
  explicit LRNModel(const std::string& layer_name) {}
};

template <typename Dtype>
class LRNParam : public LayerParam<Dtype> {
public:
  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  Dtype k_;
  int num_;
  int channels_;
  int height_;
  int width_;
  LRNProto_NormRegion norm_region_;
  explicit LRNParam() {
  }
};

template <typename Dtype>
class LRNLayer : public BaseLayer<Dtype> {
public:
  explicit LRNLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {
  }

  DataParam<Dtype>* CreateDataParam() const override {
    return new LRNData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new LRNModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  virtual void CrossChannelForward(const ContextParam& ctx,
    DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const;
  virtual void CrossChannelBackward(const ContextParam& ctx,
    DataParam<Dtype>* data_param, ModelParam<Dtype>* model_param) const;

  LRNLayer(const LRNLayer& other) = delete;
  LRNLayer& operator=(const LRNLayer& other) = delete;


};
#if 0
  explicit LRNLayer(const LayerParameter& param)
    : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LRN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void CrossChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void WithinChannelForward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void CrossChannelBackward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void CrossChannelBackward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void WithinChannelBackward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int size_;
  int pre_pad_;
  Dtype alpha_;
  Dtype beta_;
  Dtype k_;
  int num_;
  int channels_;
  int height_;
  int width_;

  // Fields used for normalization ACROSS_CHANNELS
  // scale_ stores the intermediate summing results
  Blob<Dtype> scale_;
#endif 
}  // namespace caffe

#endif  // _LAYERS_LRN_LAYER_HPP_
