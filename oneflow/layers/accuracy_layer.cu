#include <cuda.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>

#include <cmath>
#include <algorithm>
#include <string>
#include <vector>

#include "common/common.h"
#include "layers/layer.h"
#include "layers/loss_layers.h"
#include "math/math_util.h"

namespace caffe {
template <typename Dtype>
void AccuracyLayer<Dtype>::Forward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& inputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* outputs) {

  //Dtype accuracy = 0;
  //const Dtype* bottom_data = bottom[0]->cpu_data();
  //const Dtype* bottom_label = bottom[1]->cpu_data();
  //const int dim = bottom[0]->count() / outer_num_;
  //const int num_labels = bottom[0]->shape(label_axis_);
  //vector<Dtype> maxval(top_k_ + 1);
  //vector<int> max_id(top_k_ + 1);
  //if (top.size() > 1) {
  //  caffe_set(nums_buffer_.count(), Dtype(0), nums_buffer_.mutable_cpu_data());
  //  caffe_set(top[1]->count(), Dtype(0), top[1]->mutable_cpu_data());
  //}
  //int count = 0;
  //for (int i = 0; i < outer_num_; ++i) {
  //  for (int j = 0; j < inner_num_; ++j) {
  //    const int label_value =
  //      static_cast<int>(bottom_label[i * inner_num_ + j]);
  //    if (has_ignore_label_ && label_value == ignore_label_) {
  //      continue;
  //    }
  //    if (top.size() > 1) ++nums_buffer_.mutable_cpu_data()[label_value];
  //    DCHECK_GE(label_value, 0);
  //    DCHECK_LT(label_value, num_labels);
  //    // Top-k accuracy
  //    std::vector<std::pair<Dtype, int> > bottom_data_vector;
  //    for (int k = 0; k < num_labels; ++k) {
  //      bottom_data_vector.push_back(std::make_pair(
  //        bottom_data[i * dim + k * inner_num_ + j], k));
  //    }
  //    std::partial_sort(
  //      bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
  //      bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
  //    // check if true label is in top k predictions
  //    for (int k = 0; k < top_k_; k++) {
  //      if (bottom_data_vector[k].second == label_value) {
  //        ++accuracy;
  //        if (top.size() > 1) ++top[1]->mutable_cpu_data()[label_value];
  //        break;
  //      }
  //    }
  //    ++count;
  //  }
  //}

  //// LOG(INFO) << "Accuracy: " << accuracy;
  //top[0]->mutable_cpu_data()[0] = accuracy / count;
  //if (top.size() > 1) {
  //  for (int i = 0; i < top[1]->count(); ++i) {
  //    top[1]->mutable_cpu_data()[i] =
  //      nums_buffer_.cpu_data()[i] == 0 ? 0
  //      : top[1]->cpu_data()[i] / nums_buffer_.cpu_data()[i];
  //  }
  //}
  //// Accuracy layer should not be used as a loss function.
}
template <typename Dtype>
void AccuracyLayer<Dtype>::Backward(
  cublasHandle_t cublas_handle, cudaStream_t cuda_stream,
  const std::vector<std::shared_ptr<Blob<Dtype>>>& outputs,
  std::unordered_map<std::string, std::shared_ptr<Blob<Dtype>>>* model,
  std::vector<std::shared_ptr<Blob<Dtype>>>* inputs) {
}

  INSTANTIATE_LAYER_FUNCS(AccuracyLayer);

}  // namespace caffe
