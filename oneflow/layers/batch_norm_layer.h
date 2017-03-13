#ifndef _LAYER_BATCHNORM_LAYER_H_
#define _LAYER_BATCHNORM_LAYER_H_

#include <glog/logging.h>
#include <string>
#include <vector>
#include "layers/base_layer.h"
#include "caffe.pb.h"
#include "proto_io.h"

namespace caffe {

template <typename Dtype>
class BatchNormData : public DataParam<Dtype> {
public:
  Blob<Dtype>* in{ nullptr };
  Blob<Dtype>* out{ nullptr };
  Blob<Dtype>* in_diff{ nullptr };
  Blob<Dtype>* out_diff{ nullptr };
  Blob<Dtype>* mean_{ nullptr };
  Blob<Dtype>* temp_{ nullptr };
  Blob<Dtype>* variance_{ nullptr };
  Blob<Dtype>* x_norm_{ nullptr };
  Blob<Dtype>* num_by_chans_{ nullptr };
  // extra temporary variables is used to carry out sums/broadcasting
  // using BLAS
  Blob<Dtype>* batch_sum_multiplier_{ nullptr };
  Blob<Dtype>* spatial_sum_multiplier_{ nullptr };

  explicit BatchNormData(const std::string& layer_name) {
    DATA_REGISTER_BLOB(layer_name, in, BlobType::kInput);
    DATA_REGISTER_BLOB(layer_name, out, BlobType::kOutput);
    DATA_REGISTER_BLOB(layer_name, in_diff, BlobType::kInDiff);
    DATA_REGISTER_BLOB(layer_name, out_diff, BlobType::kOutDiff);
    DATA_REGISTER_BLOB(layer_name, mean_, BlobType::kOther);
    DATA_REGISTER_BLOB(layer_name, temp_, BlobType::kOther);
    DATA_REGISTER_BLOB(layer_name, variance_, BlobType::kOther);
    DATA_REGISTER_BLOB(layer_name, x_norm_, BlobType::kOther);
    DATA_REGISTER_BLOB(layer_name, num_by_chans_, BlobType::kOther);
    DATA_REGISTER_BLOB(layer_name, batch_sum_multiplier_, BlobType::kOther);
    DATA_REGISTER_BLOB(layer_name, spatial_sum_multiplier_, BlobType::kOther);
  }
};

template <typename Dtype>
class BatchNormModel : public ModelParam<Dtype> {
public:
  std::vector<Blob<Dtype>* > blobs_;
  explicit BatchNormModel(const std::string& layer_name) {
    blobs_.resize(3, nullptr);
    for (int32_t idx = 0; idx < 3; ++idx) {
      MODEL_REGISTER_ARRAY_BLOB(layer_name, blobs_, idx, BlobType::kModel);
    }
  }
};

template <typename Dtype>
class BatchNormParam : public LayerParam<Dtype> {
public:
  bool use_global_stats_;
  Dtype moving_average_fraction_;
  int channels_;
  Dtype eps_;
  BatchNormParam(){}
};

template <typename Dtype>
class BatchNormLayer : public BaseLayer<Dtype> {
public:
  explicit BatchNormLayer(const std::string& layer_name,
    const std::string& proto_param) : BaseLayer(layer_name, proto_param) {
  }

  // TODO(jiyuan): use covariant return type
  DataParam<Dtype>* CreateDataParam() const override {
    return new BatchNormData<Dtype>(layer_name_);
  }
  ModelParam<Dtype>* CreateModelParam() const override {
    return new BatchNormModel<Dtype>(layer_name_);
  }
  void InitParamFromProto() override;
  void InitFromInputShape(DataParam<Dtype>* data_param) override;
  void Forward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
  void Backward(const ContextParam& ctx, DataParam<Dtype>* data_param,
    ModelParam<Dtype>* model_param) const override;
private:
  BatchNormLayer(const BatchNormLayer& other) = delete;
  BatchNormLayer& operator=(const BatchNormLayer& other) = delete;
};

}  // namespace caffe

#endif  // _LAYER_BATCHNORM_LAYER_HPP_
