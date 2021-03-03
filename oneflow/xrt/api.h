/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_XRT_API_H_
#define ONEFLOW_XRT_API_H_

#include "oneflow/core/common/shape.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/passes/pass.h"

namespace oneflow {
namespace xrt {

std::string ExtractOpTypeAsString(const OperatorConf& conf);

XrtDevice DeviceTypeToXrtDevice(const DeviceType& device_type);

XrtDevice DeviceTagToXrtDevice(const std::string& device_tag);

DeviceType XrtDeviceToDeviceType(const XrtDevice& device);

XrtEngine StringToXrtEngine(const std::string& engine);

std::string BlobIdToName(const LogicalBlobId& lbi);

LogicalBlobId BlobNameToId(const std::string& blob_name);

template<typename T>
inline Shape AsShape(const std::vector<T>& dim_vec) {
  return Shape(DimVector(dim_vec.begin(), dim_vec.end()));
}

// Build an xrt graph from launch conf.
std::shared_ptr<XrtGraph> BuildXrtGraph(const XrtLaunchOpConf::Function& function,
                                        const DeviceType& device_type, const JobDesc& job_desc);

// Build an xrt graph from op graph.
std::shared_ptr<XrtGraph> BuildXrtGraph(const OpGraph* op_graph);

void InitXrtConfigurations(const XrtConfig& config);

bool XrtCompilationEnabled();

// Create a default options for xrt pass.
// If environment variables FLAGS_clustering_minimum_nodes,
// FLAGS_clustering_maximum_nodes, and FLAGS_strict_clustering have been set,
// then it will be filled by these values.
XrtPassOptions CreateDefaultXrtPassOptions(bool train_phase = false);

// Run an xrt pass with fixed parameters.
// args:
// pass    "Pass type, sunch as \"BuildSubGraph\"."
// graph   "An XRT graph which be applied by pass."
// options "Specify options to affect pass results."
inline void RunXrtPass(const std::string& pass, XrtGraph* graph, const XrtPassOptions& options) {
  return RunPassImpl(pass, graph, options);
}

// Run an xrt pass with unfixed parameters.
template<typename... Args>
inline void RunXrtPass(const std::string& pass, XrtGraph* graph, const XrtPassOptions& options,
                       Args&&... args) {
  return RunPassImpl(pass, graph, options, std::forward<Args>(args)...);
}

void RunCompilationTimeXrtPasses(const OpGraph& op_graph, Job* job, bool train_phase);

#ifdef WITH_TENSORRT
namespace tensorrt {
void CacheInt8Calibration();
void WriteInt8Calibration(const std::string& path);
}  // namespace tensorrt
#endif  // WITH_TENSORRT

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_API_H_
