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
#include "oneflow/core/operator/operator.h"  // GenLogicalBlobName, GenLogicalBlobId
#include "oneflow/core/framework/to_string.h"
#include "oneflow/xrt/api.h"
#include "oneflow/xrt/build_graph.h"
#include "oneflow/xrt/utility/env.h"

#include "absl/strings/str_cat.h"
#include "glog/logging.h"

#include <fstream>
#include <mutex>

#ifdef WITH_TENSORRT
#include "oneflow/xrt/tensorrt/trt_int8_calibrator.h"
#endif  // WITH_TENSORRT

DEFINE_int32(clustering_minimum_nodes, EnvToInt(FLAGS_clustering_minimum_nodes, 1),
             "Minium nodes of a cluster after clustering.");
DEFINE_int32(clustering_maximum_nodes, EnvToInt(FLAGS_clustering_maximum_nodes, 1000),
             "Maxium nodes of a cluster after clustering.");
DEFINE_bool(strict_clustering, EnvToBool(FLAGS_strict_clustering, true),
            "Option to clustering with strict dependencies analysis.");

// DEFINE_string(engine, EnvToString(FLAGS_engine, "XLA"),
//               "Which third party engine to be used. XLA and TENSORRT are "
//               "valid, Default means using no engine.");
DEFINE_bool(use_xla_jit, EnvToBool(FLAGS_use_xla_jit, false), "It's optional to use xla jit.");
DEFINE_bool(use_tensorrt, EnvToBool(FLAGS_use_tensorrt, false), "It's optional to use tensorrt.");

DEFINE_bool(tensorrt_fp16, EnvToBool(FLAGS_tensorrt_fp16, false),
            "Enable fp16 precision for TENSORRT engine.");
DEFINE_bool(tensorrt_int8, EnvToBool(FLAGS_tensorrt_int8, false),
            "Enable int8 precision for TENSORRT engine.");

DEFINE_string(int8_calibration, EnvToString(FLAGS_int8_calibration, ""),
              "TensorRT int8 calibration table directory. "
              "Default is empty, and this means the calibration table will be "
              "implictly generated if tensorrt_int8 flag is true.");

namespace oneflow {
namespace xrt {

#define OP_TYPE_CASE(op) OperatorConf::k##op##Conf

static std::unordered_map<int32_t, std::string> op_type2string_map = {
    {OP_TYPE_CASE(Identity), "Identity"},
    // {OP_TYPE_CASE(FullyConnected), "FullyConnected"},
    // TODO(hjchen2)
};

static std::unordered_map<std::string, std::string> user_op_type_name2string_map = {
    {"tanh", "Tanh"},
    {"tanh_grad", "TanhGrad"},
    {"gelu", "Gelu"},
    {"gelu_grad", "GeluGrad"},
    {"sigmoid", "Sigmoid"},
    {"relu", "Relu"},
    {"normalization", "Normalization"},
    {"bias_add", "BiasAdd"},
    {"broadcast_add", "BcastAdd"},
    {"broadcast_mul", "BcastMul"},
    {"broadcast_div", "BcastDiv"},
    {"broadcast_min", "BcastMin"},
    {"cast", "Cast"},
    {"concat", "Concat"},
    {"conv2d", "Conv2D"},
    {"multiply", "Multiply"},
    {"add_n", "Add"},
    {"matmul", "MatMul"},
    {"max_pool_2d", "MaxPooling2D"},
    {"avg_pool_2d", "AveragePooling2D"},
    {"reduce_sum", "ReduceSum"},
    {"reduce_mean", "ReduceMean"},
    {"reshape", "Reshape"},
    {"reshape_like", "ReshapeLike"},
    {"softmax", "Softmax"},
    {"softmax_grad", "SoftmaxGrad"},
    {"top_k", "TopK"},
    {"transpose", "Transpose"},
    {"gather", "Gather"},
    {"batch_gather", "BatchGather"},
    {"layer_norm", "LayerNorm"},
    {"layer_norm_param_grad", "LayerNormParamGrad"},
    {"layer_norm_grad", "LayerNormGrad"},
    {"scalar_add", "ScalarAdd"},
    {"scalar_mul", "ScalarMul"},
    {"leaky_relu", "LeakyRelu"},
    {"adam_update", "AdamOptimizer"},
    {"rsqrt", "Rsqrt"},
    {"square_sum", "SquareSum"},
};

std::string ExtractOpTypeAsString(const OperatorConf &conf) {
  if (conf.has_user_conf()) {
    const auto it = user_op_type_name2string_map.find(conf.user_conf().op_type_name());
    if (it != user_op_type_name2string_map.end()) {
      return it->second;
    } else {
      // Return empty if the operator is not in the translation map
      return std::string("");
    }
  } else {
    const auto it = op_type2string_map.find(conf.op_type_case());
    if (it != op_type2string_map.end()) {
      return it->second;
    } else {
      // Return empty if the operator is not in the translation map
      return std::string("");
    }
  }
}

XrtDevice DeviceTagToXrtDevice(const std::string &device_tag) {
  DeviceType device_type = CHECK_JUST(DeviceType4DeviceTag(device_tag));
  return DeviceTypeToXrtDevice(device_type);
}

XrtDevice DeviceTypeToXrtDevice(const DeviceType &device_type) {
  switch (device_type) {
    case DeviceType::kGPU: return XrtDevice::GPU_CUDA;
    case DeviceType::kCPU: return XrtDevice::CPU_X86;
    default:
      DLOG(WARNING) << "Meet invalid device type (" << device_type
                    << "). Use the default xrt device instead.";
      return XrtDevice::CPU_X86;
  }
}

DeviceType XrtDeviceToDeviceType(const XrtDevice &device) {
  if (device == XrtDevice::GPU_CUDA) {
    return DeviceType::kGPU;
  } else if (device == XrtDevice::CPU_X86) {
    return DeviceType::kCPU;
  } else {
    LOG(FATAL) << "Can not convert xrt device (" << device << ") to device type.";
    return DeviceType::kCPU;
  }
}

XrtEngine StringToXrtEngine(const std::string &engine) {
  if (engine == "XLA") {
    return xrt::XrtEngine::XLA;
  } else if (engine == "TENSORRT") {
    return xrt::XrtEngine::TENSORRT;
  } else {
    LOG(FATAL) << "Unknown engine: " << engine;
  }
}

std::string BlobIdToName(const LogicalBlobId &lbi) {
  CHECK_EQ(lbi.has_op_name(), true);
  CHECK_EQ(lbi.has_blob_name(), true);
  if (lbi.op_name() == "") { return lbi.blob_name(); }
  return GenLogicalBlobName(lbi);
}

LogicalBlobId BlobNameToId(const std::string &blob_name) {
  size_t pos = blob_name.find('/');
  if (pos == std::string::npos) {
    return GenLogicalBlobId("/" + blob_name);
  } else {
    return GenLogicalBlobId(blob_name);
  }
}

std::shared_ptr<XrtGraph> BuildXrtGraph(const OpGraph *op_graph) {
  return graph_builder::BuildGraph(op_graph);
}

std::shared_ptr<XrtGraph> BuildXrtGraph(const XrtLaunchOpConf::Function &function,
                                        const DeviceType &device_type, const JobDesc &job_desc) {
  return graph_builder::BuildGraph(function, device_type, job_desc);
}

void InitXrtConfigurations(const XrtConfig &config) {
  if (config.has_use_xla_jit()) { FLAGS_use_xla_jit = config.use_xla_jit(); }
  if (config.has_use_tensorrt()) { FLAGS_use_tensorrt = config.use_tensorrt(); }
  // Set xla configurations.
  if (config.has_tensorrt_config()) {
    const XrtConfig::TensorRTConfig &trt_config = config.tensorrt_config();
    if (trt_config.has_use_fp16()) { FLAGS_tensorrt_fp16 = trt_config.use_fp16(); }
    if (trt_config.has_use_int8()) {
      FLAGS_tensorrt_int8 = trt_config.use_int8();
      if (trt_config.has_int8_calibration()) {
        FLAGS_int8_calibration = trt_config.int8_calibration();
      }
    }
  }
}

bool XrtCompilationEnabled() { return FLAGS_use_xla_jit || FLAGS_use_tensorrt; }

XrtPassOptions CreateDefaultXrtPassOptions(bool train_phase) {
  ClusteringOptions options;
  options.minimum_nodes = FLAGS_clustering_minimum_nodes;
  options.maximum_nodes = FLAGS_clustering_maximum_nodes;
  options.strict_clustering = FLAGS_strict_clustering;

  options.train_phase = train_phase;
  // TODO(hjchen2)
  options.engine = (1U << XrtEngineOptionBit::kUseDefault);
  if (FLAGS_use_xla_jit) { options.engine |= (1U << XrtEngineOptionBit::kUseXlaJit); }
  if (FLAGS_use_tensorrt) { options.engine |= (1U << XrtEngineOptionBit::kUseTensorRT); }

  XrtPassOptions xrt_options;
  xrt_options.clustering_options = options;
  return xrt_options;
}

void RunCompilationTimeXrtPasses(const OpGraph &op_graph, Job *job, bool train_phase) {
  auto graph = BuildXrtGraph(&op_graph);
  // Create options to run xrt passes.
  auto options = CreateDefaultXrtPassOptions(train_phase);

  RunXrtPass("MarkClusterId", graph.get(), options);
  RunXrtPass("BuildSubGraph", graph.get(), options);
  // Rebuild Job
  RunXrtPass("RebuildCompiledJob", graph.get(), options, job);
}

#ifdef WITH_TENSORRT
namespace tensorrt {
void CacheInt8Calibration() {
  const auto &calib_resources = TRTInt8CalibratorResource::All();
  for (const auto &res : calib_resources) {
    std::lock_guard<std::mutex> lock(res.second->mutex_);
    if (!res.second->calibrator_->isDone()) {
      res.second->calibrator_->waitAndSetDone();
      res.second->thread_->join();
    }
    res.second->calibrator_->ReleaseDevBuffers();
  }
}

void WriteInt8Calibration(const std::string &path) {
  const auto &calib_resources = TRTInt8CalibratorResource::All();
  for (const auto &res : calib_resources) {
    CHECK(res.second->calibrator_->isDone())  // NOLINT
        << "Calibration table maybe has not been generated "
        << "since the calibrator has not been done.";

    const std::string &calibration_table_data =
        res.second->calibrator_->getCalibrationTableAsString();
    CHECK(calibration_table_data.size()) << "Calibration table data is empty.";

    std::string calib_store_path =  // NOLINT
        absl::StrCat(path, "/", res.first /*calibrator name*/);
    std::ofstream ofile(calib_store_path, std::ios::out);
    CHECK(ofile.good()) << "Could not open calibration file: " << calib_store_path;
    ofile << calibration_table_data;
    ofile.close();
  }
}
}  // namespace tensorrt
#endif  // WITH_TENSORRT

}  // namespace xrt
}  // namespace oneflow
