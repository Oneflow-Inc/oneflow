#ifndef ONEFLOW_XRT_XRT_H_
#define ONEFLOW_XRT_XRT_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/xrt/graph/graph.h"
#include "oneflow/xrt/parameter.h"
#include "oneflow/xrt/passes/pass.h"
#include "oneflow/xrt/utility/env.h"
#include "oneflow/xrt/xrt.pb.h"

namespace oneflow {
namespace xrt {

std::string ExtractOpTypeAsString(const OperatorConf &conf);

XrtDevice DeviceTypeToBackend(const DeviceType &device_type);

DeviceType BackendToDeviceType(const XrtDevice &backend);

std::string BlobIdToName(const LogicalBlobId &lbi);

LogicalBlobId BlobNameToId(const std::string &blob_name);

// Build an xrt graph from launch conf.
std::shared_ptr<XrtGraph> BuildXrtGraph(const XrtLaunchOpConf &launch_conf,
                                        const DeviceType &device_type,
                                        const JobDesc &job_desc);

// Build an xrt graph from op graph.
std::shared_ptr<XrtGraph> BuildXrtGraph(const OpGraph *op_graph);

// Create a default options for xrt pass.
// If environment variables FLAGS_clustering_minimum_nodes,
// FLAGS_clustering_maximum_nodes, and FLAGS_strict_clustering have been set,
// then it will be filled by these values.
XrtPassOptions CreateDefaultXrtPassOptions();

// Run an xrt pass with fixed parameters.
// args:
// pass    "Pass type, sunch as \"BuildSubGraph\"."
// graph   "An XRT graph which be applied by pass."
// options "Specify options to affect pass results."
inline void RunXrtPass(const std::string &pass, XrtGraph *graph,
                       const XrtPassOptions &options) {
  return RunPassImpl(pass, graph, options);
}

// Run an xrt pass with unfixed parameters.
template <typename... Args>
inline void RunXrtPass(const std::string &pass, XrtGraph *graph,
                       const XrtPassOptions &options, Args &&... args) {
  return RunPassImpl(pass, graph, options, std::forward<Args>(args)...);
}

bool LookupMutability(const XrtLaunchOpConf &launch_conf,
                      const std::string &argument);

Parameter BuildParameter(const Blob &blob, const std::string &name = "");

}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XRT_H_
