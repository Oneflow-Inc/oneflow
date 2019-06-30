#include "oneflow/core/job/oneflow.h"

int run_serialized_job_set(const oneflow::JobSet& job_set) { return Main(job_set); }
