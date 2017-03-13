#ifndef _LAYERS_CUDNN_CONVOLUTION_LAYER_H_
#define _LAYERS_CUDNN_CONVOLUTION_LAYER_H_
#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/convolution_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"
#include "common/cudnn_utils.h"

namespace caffe {

template <typename Dtype>
class CuDNNConvolutionData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };

  explicit CuDNNConvolutionData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
  }
};


template <typename Dtype>
class CuDNNConvolutionParam : public ConvolutionParam<Dtype> {
public:

  cudnnTensorDescriptor_t in_descs_, out_descs_;
  cudnnTensorDescriptor_t bias_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_descs_;
  int in_offset_, out_offset_, bias_offset_;

  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;

  explicit CuDNNConvolutionParam() {
    fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  }

  virtual ~CuDNNConvolutionParam() {
    cudnnDestroyTensorDescriptor(in_descs_);
    cudnnDestroyTensorDescriptor(out_descs_);
    cudnnDestroyConvolutionDescriptor(conv_descs_);
    if (this->bias_term_) {
      cudnnDestroyTensorDescriptor(bias_desc_);
    }
    cudnnDestroyFilterDescriptor(filter_desc_);
  }
};

//#ifdef USE_CUDNN
template <typename Dtype>
class CuDNNConvolutionLayer : public BaseLayer<Dtype> {
public:
  explicit CuDNNConvolutionLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {}

  DataParam<Dtype>* CreateDataParam() const override {
    return new CuDNNConvolutionData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new ConvolutionModel<Dtype>(layer_name_);
  }

  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;

private:

  CuDNNConvolutionLayer(const CuDNNConvolutionLayer& other) = delete;
  CuDNNConvolutionLayer& operator=(const CuDNNConvolutionLayer& other) = delete;

};
//#endif


}  // namespace caffe
#endif  // _LAYERS_CUDNN_CONVOLUTION_LAYER_H_