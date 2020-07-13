#ifndef OF_UTIL_HISTOGRAM_H_
#define OF_UTIL_HISTOGRAM_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

class HistogramProto;

namespace histogram {

class Histogram {
 public:
  Histogram();
  ~Histogram() {}

  void AppendValue(double value);
  void AppendToProto(HistogramProto* proto);

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