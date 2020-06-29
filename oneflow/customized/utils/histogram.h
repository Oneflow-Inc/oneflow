#ifndef OF_UTIL_HISTOGRAM_H_
#define OF_UTIL_HISTOGRAM_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

class HistogramProto;

namespace histogram {

#define MIN_VALUE 1.0e-12
#define MAX_VLAUE 1.0e20
#define INCREASE_RATE 1.1

class Histogram {
 public:
  Histogram();
  explicit Histogram(const std::vector<double>& container);
  ~Histogram() {}

  void AppendValue(double value);
  void AppendToProto(HistogramProto* proto);
  void Clear();

 private:
  double value_count_;
  double value_sum_;
  double sum_value_squares_;
  double min_value_;
  double max_value_;

  std::vector<double> max_constainers_;
  std::vector<double> containers_;
  OF_DISALLOW_COPY(Histogram);
};

}  // namespace histogram
}  // namespace oneflow

#endif