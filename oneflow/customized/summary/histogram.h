#ifndef ONEFLOW_CUSTOMIZED_SUMMARY_HISTOGRAM_H_
#define ONEFLOW_CUSTOMIZED_SUMMARY_HISTOGRAM_H_

#include <vector>
#include "oneflow/customized/summary/summary.pb.h"

namespace oneflow {

namespace summary {

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
};

}  // namespace summary

}  // namespace oneflow

#endif
