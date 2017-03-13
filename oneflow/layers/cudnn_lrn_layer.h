#ifndef _LAYERS_CUDNN_LRN_LAYER_H_
#define _LAYERS_CUDNN_LRN_LAYER_H_


#include "common/cudnn_utils.h"
#include "layers/lrn_layer.h"

namespace caffe {

//#ifdef USE_CUDNN


template <typename Dtype>
class CuDNNLRNParam : public LRNParam<Dtype> {
public:

  cudnnLRNDescriptor_t norm_desc_;
  cudnnTensorDescriptor_t in_desc_, out_desc_;
  int size_;
  Dtype alpha_, beta_, k_;

  explicit CuDNNLRNParam() {}

  virtual  ~CuDNNLRNParam() {
    cudnnDestroyTensorDescriptor(in_desc_);
    cudnnDestroyTensorDescriptor(out_desc_);
    cudnnDestroyLRNDescriptor(norm_desc_);
  }
};

template <typename Dtype>
class CuDNNLRNLayer : public BaseLayer<Dtype> {
public:
  explicit CuDNNLRNLayer(const std::string& layer_name,
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
  CuDNNLRNLayer(const CuDNNLRNLayer& other) = delete;
  CuDNNLRNLayer& operator=(const CuDNNLRNLayer& other) = delete;
};

//#endif

}  // namespace caffe

#endif  // _LAYERS_CUDNN_LRN_LAYER_H_
