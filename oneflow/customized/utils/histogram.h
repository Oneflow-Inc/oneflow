#ifndef OF_UTIL_HISTOGRAM_H_
#define OF_UTIL_HISTOGRAM_H_

#include "oneflow/core/framework/framework.h"

namespace oneflow {

class HistogramProto;

namespace histogram {

class Histogram {
 public:
  Histogram();
  explicit Histogram(std::vector<double> custom_bucket_limits);

  bool DecodeFromProto(const HistogramProto& proto);

  ~Histogram() {}

  void Clear();
  void Add(double value);

  void EncodeToProto(HistogramProto* proto, bool preserve_zero_buckets) const;

  double Median() const;

  double Percentile(double p) const;

  double Average() const;

  double StandardDeviation() const;

  std::string ToString() const;

 private:
  double min_;
  double max_;
  double num_;
  double sum_;
  double sum_squares_;

  std::vector<double> custom_bucket_limits_;
  std::vector<double> bucket_limits_;
  std::vector<double> buckets_;

  double Remap(double x, double x0, double x1, double y0, double y1) const;
  OF_DISALLOW_COPY(Histogram);
};

}  // namespace histogram
}  // namespace oneflow

#endif