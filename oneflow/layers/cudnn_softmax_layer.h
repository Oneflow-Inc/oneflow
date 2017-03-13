#ifndef _LAYERS_CUDNN_SOFTMAX_LAYER_H_
#define _LAYERS_CUDNN_SOFTMAX_LAYER_H_


#include "layers/softmax_layer.h"
#include "common/cudnn_utils.h"
namespace caffe {

//#ifdef USE_CUDNN

template <typename Dtype>
class CuDNNSoftmaxModel : public ModelParam<Dtype> {
public:
  explicit CuDNNSoftmaxModel(const std::string& layer_name) {}
};

template <typename Dtype>
class CuDNNSoftmaxParam : public SoftmaxParam<Dtype> {
public:
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  explicit CuDNNSoftmaxParam(){
  }
  virtual ~CuDNNSoftmaxParam() {
    cudnnDestroyTensorDescriptor(this->in_desc_);
    cudnnDestroyTensorDescriptor(this->out_desc_);
  }
};

template <typename Dtype>
class CuDNNSoftmaxLayer : public BaseLayer<Dtype> {
public:
  explicit CuDNNSoftmaxLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {
    is_elem_wise_ = true;
  }
  DataParam<Dtype>* CreateDataParam() const override {
    return new SoftmaxData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new CuDNNSoftmaxModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  CuDNNSoftmaxLayer(const CuDNNSoftmaxLayer& other) = delete;
  CuDNNSoftmaxLayer& operator=(const CuDNNSoftmaxLayer& other) = delete;

};
//#endif

}  // namespace caffe

#endif  // _LAYERS_CUDNN_SOFTMAX_LAYER_H_
