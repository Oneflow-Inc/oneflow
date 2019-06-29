#include "oneflow/core/job/oneflow.h"

#include <vector>

float average(std::vector<float> v) {
  float s = 0;
  for (int i = 0; i < v.size(); i++) s += v[i];
  return s / v.size();
}

int launchWithSerialized(const oneflow::JobSet& job_set) { return launch(job_set); }
