#ifndef _LAYERS_CUDNN_POOLING_LAYER_H_
#define _LAYERS_CUDNN_POOLING_LAYER_H_

#include "layers/pooling_layer.h"
#include "common/cudnn_utils.h"
namespace caffe {

template <typename Dtype>
class CuDNNPoolingData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };
  explicit CuDNNPoolingData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
  }
};

template <typename Dtype>
class CuDNNPoolingParam : public PoolingParam<Dtype> {
public:
  cudnnTensorDescriptor_t in_desc_, out_desc_;
  cudnnPoolingDescriptor_t  pooling_desc_;
  cudnnPoolingMode_t        mode_;

  explicit CuDNNPoolingParam() {}

  virtual  ~CuDNNPoolingParam() {
    cudnnDestroyTensorDescriptor(in_desc_);
    cudnnDestroyTensorDescriptor(out_desc_);
    cudnnDestroyPoolingDescriptor(pooling_desc_);
  }
};

#if 0
bool handles_setup_;
cudnnHandle_t             handle_;
cudnnTensorDescriptor_t in_desc_, out_desc_;
cudnnPoolingDescriptor_t  pooling_desc_;
cudnnPoolingMode_t        mode_;

#endif



//#ifdef USE_CUDNN
/*
* @brief cuDNN implementation of PoolingLayer.
*        Fallback to PoolingLayer for CPU mode.
*/
template <typename Dtype>
class CuDNNPoolingLayer : public BaseLayer<Dtype> {
public:
  explicit CuDNNPoolingLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new CuDNNPoolingData<Dtype>(layer_name_);
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

  CuDNNPoolingLayer(const CuDNNPoolingLayer& other) = delete;
  CuDNNPoolingLayer& operator=(const CuDNNPoolingLayer& other) = delete;
};
//#endif

}  // namespace caffe

#endif  // _LAYERS_CUDNN_POOLING_LAYER_H_
