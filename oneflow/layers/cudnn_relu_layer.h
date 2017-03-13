#ifndef _LAYERS_CUDNN_RELU_LAYER_H_
#define _LAYERS_CUDNN_RELU_LAYER_H_


#include "layers/relu_layer.h"
#include "common/cudnn_utils.h"
namespace caffe {

//#ifdef USE_CUDNN
/**
* @brief CuDNN acceleration of ReLULayer.
*/
template <typename Dtype>
class CuDNNReLUParam : public ReLUParam<Dtype> {
public:
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  explicit CuDNNReLUParam(){
  }
  virtual ~CuDNNReLUParam() {
    cudnnDestroyTensorDescriptor(this->in_desc_);
    cudnnDestroyTensorDescriptor(this->out_desc_);
  }
};

template <typename Dtype>
class CuDNNReLULayer : public BaseLayer<Dtype> {
public:
  explicit CuDNNReLULayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {
    is_elem_wise_ = true;
  }
  DataParam<Dtype>* CreateDataParam() const override {
    return new ReLUData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new ReLUModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  CuDNNReLULayer(const CuDNNReLULayer& other) = delete;
  CuDNNReLULayer& operator=(const CuDNNReLULayer& other) = delete;

};
//#endif

}  // namespace caffe

#endif  // _LAYERS_CUDNN_RELU_LAYER_H_
