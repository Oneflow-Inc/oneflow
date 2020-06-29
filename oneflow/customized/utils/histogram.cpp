#include "oneflow/customized/utils/histogram.h"
#include "oneflow/customized/utils/summary.pb.h"

namespace oneflow {

namespace histogram {

static std::vector<double>* InitContainers() {
  std::vector<double>* default_containers = new std::vector<double>;
  std::vector<double> containers;
  std::vector<double> neg_containers;
  double value = MIN_VALUE;
  while (value < MAX_VLAUE) {
    containers.emplace_back(value);
    neg_containers.emplace_back(-value);
    value *= INCREASE_RATE;
  }
  containers.emplace_back(DBL_MAX);
  neg_containers.emplace_back(-DBL_MAX);
  std::reverse(neg_containers.begin(), neg_containers.end());
  default_containers->insert(default_containers->end(), neg_containers.begin(),
                             neg_containers.end());
  default_containers->emplace_back(0.0);
  default_containers->insert(default_containers->end(), containers.begin(), containers.end());
  return default_containers;
}

static std::vector<double> InitDefaultContainers() {
  static std::vector<double>* default_containers = InitContainers();
  return *default_containers;
}

Histogram::Histogram() : max_constainers_(InitDefaultContainers()) { Clear(); }

void Histogram::AppendValue(double value) {
  int idx = std::upper_bound(max_constainers_.begin(), max_constainers_.end(), value)
            - max_constainers_.begin();
  CHECK_GT(containers_.size(), idx);
  containers_[idx] += 1.0;
  if (max_value_ < value) { max_value_ = value; }
  if (min_value_ > value) { min_value_ = value; }

  value_sum_ += value;
  value_count_++;
  sum_value_squares_ += value * value;
}

void Histogram::Clear() {
  containers_.resize(max_constainers_.size());
  for (size_t idx = 0; idx < max_constainers_.size(); idx++) { containers_[idx] = 0; }
  value_sum_ = 0;
  sum_value_squares_ = 0;
  max_value_ = -DBL_MAX;
  value_count_ = 0;
  min_value_ = max_constainers_[max_constainers_.size() - 1];
}

void Histogram::AppendToProto(HistogramProto* hist_proto) {
  hist_proto->Clear();
  hist_proto->set_num(value_count_);
  hist_proto->set_sum(value_sum_);
  hist_proto->set_min(min_value_);
  hist_proto->set_max(max_value_);
  hist_proto->set_sum_squares(sum_value_squares_);
  for (size_t idx = 0; idx < containers_.size();) {
    double num = containers_[idx];
    double last = max_constainers_[idx];
    idx++;
    if (num <= 0.0) {
      while (idx < containers_.size() && containers_[idx] <= 0.0) {
        last = max_constainers_[idx];
        num = containers_[idx];
        idx++;
      }
    }
    hist_proto->add_bucket_limit(last);
    hist_proto->add_bucket(num);
  }
  if (hist_proto->bucket_size() == 0.0) {
    hist_proto->add_bucket_limit(DBL_MAX);
    hist_proto->add_bucket(0.0);
  }
}

}  // namespace histogram
}  // namespace oneflow
