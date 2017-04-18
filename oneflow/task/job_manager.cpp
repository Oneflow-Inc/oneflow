#include "task/job_manager.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <limits>
#include "common/common.h"
#include "context/one.h"
#include "context/id_map.h"
#include "context/config_parser.h"
#include "context/machine_descriptor.h"
#include "context/strategy_descriptor.h"
#include "device/device_descriptor.h"

namespace oneflow {

template <typename Dtype>
JobManager<Dtype>::JobManager(){

}
template <typename Dtype>
JobManager<Dtype>::~JobManager() {}

template <typename Dtype>
void JobManager<Dtype>::Init() {
}
//INSTANTIATE_CLASS(JobManager);
}  // namespace oneflow
