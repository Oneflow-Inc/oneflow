#ifndef TEST_TEST_GRADIENT_CHECK_UTIL_H_
#define TEST_TEST_GRADIENT_CHECK_UTIL_H_

#include <cublas_v2.h>
#include <cuda.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "common/common.h"
#include "common/shape.h"
#include "layers/base_layer.h"
#include "math/math_util.h"
#include "memory/blob.h"
namespace caffe {
// The gradient checker adds a L2 normalization loss function on outputs of the
// outputs blobs, and checks the gradient.
template <typename Dtype>
class GradientChecker {
public:
  // kink and kink_range specify an ignored nonsmooth segment of the form
  // kink - kink_range <= |feature value| <= kink + kink_range,
  // which accounts for all nonsmoothness in use by caffe
  GradientChecker(const Dtype stepsize, const Dtype threshold,
    const size_t seed = kSeed, const Dtype kink = 0.,
    const Dtype kink_range = -1)
    : stepsize_(stepsize), threshold_(threshold), seed_(seed),
    kink_(kink), kink_range_(kink_range) {
    RNG::set_seed(seed_);
  }

  void CheckGradientExhaustive(ContextParam ctx, BaseLayer<Dtype>* layer,
    DataParam<Dtype>* data, ModelParam<Dtype>* model);

  // Checks the gradient of a single output with respect to particular input
  // blob(s). 
  void CheckGradientSingle(ContextParam ctx, BaseLayer<Dtype>* layer,
    DataParam<Dtype>* data, ModelParam<Dtype>* model,
    int outputs_id, int outputs_data_id);

protected:
  // Set the error signal to the out blob and return the loss
  Dtype GetObjAndGradient(ContextParam ctx, BaseLayer<Dtype>* layer,
    DataParam<Dtype>* data, int outputs_id = -1, int outputs_data_id = -1);

  Dtype stepsize_;
  Dtype threshold_;
  Dtype kink_;
  Dtype kink_range_;
  unsigned int seed_;
};

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientSingle(
  ContextParam ctx, BaseLayer<Dtype>* layer, DataParam<Dtype>* data,
  ModelParam<Dtype>* model, int outputs_id, int outputs_data_id) {
  // First, figure out what blobs we need to check against, and zero init
  // parameter blobs.
  std::vector<Blob<Dtype>*> blobs_to_change;
  std::vector<Blob<Dtype>*> blobs_to_check;
  std::vector<std::string> strings_to_check;
  std::vector<std::string> data_strings_to_check;
  
  for (auto& it = model->name_to_blob_pptr().begin();
    it != model->name_to_blob_pptr().end(); ++it) {
    if (it->first.find("diff") != std::string::npos) {
      caffe_gpu_set((*it->second)->shape().count(), static_cast<Dtype>(0),
        (*it->second)->mutable_data(), ctx.cuda_stream);
      blobs_to_check.push_back(*it->second);
      strings_to_check.push_back(it->first.substr(0,
        it->first.find("diff") - 1));
    }
  }
  for (size_t i = 0; i < strings_to_check.size(); ++i) {
    blobs_to_change.push_back(*model->name_to_blob_pptr().find(
      strings_to_check[i])->second);
  }
#if 0  
  for (auto& it = data->name_to_blob_pptr().begin();
    it != data->name_to_blob_pptr().end(); ++it) {
    if (it->first.find("diff") != std::string::npos) {
      blobs_to_check.push_back(*it->second);
      data_strings_to_check.push_back(it->first.substr(0,
        it->first.find("diff") - 1));
    }
  }
  for (size_t i = 0; i < data_strings_to_check.size(); ++i) {
    blobs_to_change.push_back(*data->name_to_blob_pptr().find(
      data_strings_to_check[i])->second);
  }
#endif 
  for (int i = 0; i < data->GetInputDiffs().size(); ++i) {
    std::string inputs_name =
      layer->layer_name() + "/" + data->GetInputDiffs()[i];
    data_strings_to_check.push_back(inputs_name.substr(0,
      inputs_name.find("diff") - 1));
    blobs_to_check.push_back(*data->name_to_blob_pptr().find(
      inputs_name)->second);
  }
  for (size_t i = 0; i < data_strings_to_check.size(); ++i) {
    blobs_to_change.push_back(*data->name_to_blob_pptr().find(
      data_strings_to_check[i])->second);
  }


  CHECK_GT(blobs_to_check.size(), 0) << "No blobs to check.";
#if 0
  // Backup the data in input blob
  std::vector<Dtype*> inputs_gpu_data(data->GetInputVars().size());
  for (int i = 0; i < data->GetInputVars().size(); ++i) {
    std::string inputs_name =
      layer->layer_name() + "/" + data->GetInputVars()[i];
    CUDA_CHECK(cudaMalloc(&inputs_gpu_data[i],
      data->GetShape(inputs_name).count()*sizeof(Dtype)));
    CUDA_CHECK(cudaMemcpy(inputs_gpu_data[i], (*data->name_to_blob_pptr().find(
      inputs_name)->second)->data(),
      data->GetShape(inputs_name).count() * sizeof(Dtype),
      cudaMemcpyDeviceToDevice));
  }
#endif

  // Compute the loss.
  layer->Forward(ctx, data, model);
  GetObjAndGradient(ctx, layer, data, outputs_id, outputs_data_id);
  layer->Backward(ctx, data, model);
  std::vector<Dtype*>
    computed_gradient_blobs(blobs_to_check.size());


  for (int blob_id = 0; blob_id < blobs_to_check.size(); ++blob_id) {
    //std::cout << strings_to_check[blob_id] << std::endl;
    size_t count = blobs_to_check[blob_id]->shape().count();
    Dtype* diff = reinterpret_cast<Dtype*>(malloc(count*sizeof(Dtype)));
    CUDA_CHECK(cudaMemcpy(diff, blobs_to_check[blob_id]->data(),
      count*sizeof(Dtype), cudaMemcpyDeviceToHost));
    computed_gradient_blobs[blob_id] = diff;
  }

#if 0
  for (int i = 0; i < data->GetInputVars().size(); ++i) {
    std::string inputs_name =
      layer->layer_name() + "/" + data->GetInputVars()[i];
    CUDA_CHECK(cudaMemcpy((*data->name_to_blob_pptr().find(
      inputs_name)->second)->mutable_data(), inputs_gpu_data[i],
      data->GetShape(inputs_name).count()*sizeof(Dtype),
      cudaMemcpyDeviceToDevice));
  }
#endif

  // Compute derivative of outputs w.r.t. each inputs and parameter input using
  // finite differencing.
  for (int blob_id = 0; blob_id < blobs_to_change.size(); ++blob_id) {
    Blob<Dtype>* current_gpu_blob = blobs_to_change[blob_id];
    const Dtype* computed_gradients = computed_gradient_blobs[blob_id];
    for (int feat_id = 0; feat_id < current_gpu_blob->shape().count();
      ++feat_id) {

      Dtype estimated_gradient = 0;
      Dtype positive_objective = 0;
      Dtype negative_objective = 0;

      size_t current_gpu_count = current_gpu_blob->shape().count();
      Dtype* current_cpu_data = reinterpret_cast<Dtype*>(
        malloc(current_gpu_count*sizeof(Dtype)));
      CUDA_CHECK(cudaMemcpy(current_cpu_data,
        current_gpu_blob->data(), current_gpu_count*sizeof(Dtype),
        cudaMemcpyDeviceToHost));
      Dtype temp_feat = current_cpu_data[feat_id];
      current_cpu_data[feat_id] = temp_feat + stepsize_;
      CUDA_CHECK(cudaMemcpy(current_gpu_blob->mutable_data(), current_cpu_data,
        current_gpu_count*sizeof(Dtype), cudaMemcpyHostToDevice));

      layer->Forward(ctx, data, model);
      positive_objective =
        GetObjAndGradient(ctx, layer, data, outputs_id, outputs_data_id);
      // Compute loss with stepsize_ subtracted from input.
      current_cpu_data[feat_id] = temp_feat - stepsize_;
      CUDA_CHECK(cudaMemcpy(current_gpu_blob->mutable_data(),
        current_cpu_data, current_gpu_count*sizeof(Dtype),
        cudaMemcpyHostToDevice));
      layer->Forward(ctx, data, model);;
      negative_objective =
        GetObjAndGradient(ctx, layer, data, outputs_id, outputs_data_id);
      // Recover original input value.
      current_cpu_data[feat_id] = temp_feat;
      CUDA_CHECK(cudaMemcpy(current_gpu_blob->mutable_data(),
        current_cpu_data, current_gpu_count*sizeof(Dtype),
        cudaMemcpyHostToDevice));
      estimated_gradient =
        (positive_objective - negative_objective) / stepsize_ / 2.0;

      Dtype computed_gradient = computed_gradients[feat_id];

      size_t current_count = current_gpu_blob->shape().count();
      Dtype* cpu_data = reinterpret_cast<Dtype*>(
        malloc(current_count*sizeof(Dtype)));
      CUDA_CHECK(cudaMemcpy(cpu_data, current_gpu_blob->data(),
        current_count*sizeof(Dtype), cudaMemcpyDeviceToHost));
      Dtype feature = cpu_data[feat_id];

      if (kink_ - kink_range_ > fabs(feature)
        || fabs(feature) > kink_ + kink_range_) {
        // We check relative accuracy, but for too small values, we threshold
        // the scale factor by 1.
        Dtype scale = std::max(std::max(std::fabs(computed_gradient),
          std::fabs(estimated_gradient)), (Dtype)1.);
        
        EXPECT_NEAR(computed_gradient, estimated_gradient, threshold_ * scale)
          << "debug: (outputs_id, outputs_data_id, blob_id, feat_id)="
          << outputs_id << "," << outputs_data_id << "," << blob_id << ","
          << feat_id << "; feat = " << feature << "; objective+ = "
          << positive_objective << "; objective- = " << negative_objective;
      }
    }
  }
  // for (int i = 0; i < data->GetInputVars().size(); ++i) {
  //   CUDA_CHECK(cudaFree(inputs_gpu_data[i]));
  // }
  for (int i = 0; i < blobs_to_check.size(); ++i) {
    free(computed_gradient_blobs[i]);
  }
}

template <typename Dtype>
void GradientChecker<Dtype>::CheckGradientExhaustive(
  ContextParam ctx, BaseLayer<Dtype>* layer, DataParam<Dtype>* data,
  ModelParam<Dtype>* model) {
  // We assume that each one of the output may related to the loss fucntion.
  // So we check all of them.
  CHECK_GT(data->GetOutputDiffs().size(), 0);
  for (int i = 0; i < data->GetOutputDiffs().size(); ++i) {
    for (int j = 0; j < data->GetShape(
      layer->layer_name() + "/" + data->GetOutputDiffs()[i]).count(); ++j) {
      CheckGradientSingle(ctx, layer, data, model, i, j);
    }
  }
}

template <typename Dtype>
Dtype GradientChecker<Dtype>::GetObjAndGradient(ContextParam ctx,
  BaseLayer<Dtype>* layer, DataParam<Dtype>* data, int outputs_id,
  int outputs_data_id) {
  // We use 0.5 * sum(out[i]^2) / n as the loss function. So we need to fill
  // output bolb the derivative of it as sum(out[i]) / n.
  Dtype loss = 0;
  std::string outputs_diff_name =
    layer->layer_name() + "/" + data->GetOutputDiffs()[outputs_id];
  std::string outputs_name = outputs_diff_name.substr(0,
    outputs_diff_name.find("diff") - 1);


  size_t count = data->GetShape(outputs_name).count();
  Dtype* output = reinterpret_cast<Dtype*>(malloc(count*sizeof(Dtype)));
  Dtype* output_diff = reinterpret_cast<Dtype*>(malloc(count*sizeof(Dtype)));

  CUDA_CHECK(cudaMemcpy(output, (*data->name_to_blob_pptr().find(
    outputs_name)->second)->data(), count*sizeof(Dtype),
    cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i) {
    output_diff[i] = (output[i] - 0.1*(i == outputs_data_id ? 1 : 0)) / count;
    loss += output_diff[i] * output_diff[i] * count / 2.0;
  }
  CUDA_CHECK(cudaMemcpy((*data->name_to_blob_pptr().find(
    outputs_diff_name)->second)->mutable_data(), output_diff,
    count*sizeof(Dtype), cudaMemcpyHostToDevice));
  free(output);
  free(output_diff);

  return loss;
}

}  // namespace caffe
#endif  // TEST_TEST_GRADIENT_CHECK_UTIL_H_
