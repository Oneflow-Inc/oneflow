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

#include "oneflow/api/common/ofblob.h"
#include "oneflow/api/cpp/env_impl.h"
#include "oneflow/api/cpp/framework/device.h"
#include "oneflow/api/cpp/framework/dtype.h"
#include "oneflow/api/cpp/framework/graph.h"
#include "oneflow/api/cpp/framework/ivalue.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/api/common/job_build_and_infer_ctx.h"
#include "oneflow/api/python/job_build/job_build_and_infer.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/tensor_util.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_build_and_infer_ctx.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/job/job_conf.cfg.h"
#include "oneflow/core/job/job_conf.pb.h"
#include "oneflow/core/job/job_ir.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/job/scope.h"
#include "oneflow/core/job/session.h"
#include "oneflow/core/operator/interface_blob_conf.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/vm/vm_util.h"

namespace oneflow_api {

namespace of = oneflow;

enum class XrtKind : int { kNone = 0, kTensorRT = 1, kOpenVino = 2 };

namespace {

class CompileScope {
 public:
  CompileScope(const of::JobConfigProto& job_config, const of::Device& device, XrtKind kind) {
    of::JobConfigProto mut_job_config = job_config;
    const std::shared_ptr<of::Scope> scope = CHECK_JUST(MakeScope(mut_job_config, device));
    CHECK_JUST(of::ThreadLocalScopeStackPush(scope));

    ConfigXrt(mut_job_config, kind);
    CHECK_JUST(of::JobBuildAndInferCtx_Open(mut_job_config.job_name()));
    CHECK_JUST(CHECK_JUST(of::GetCurInferCtx())->SetJobConf(mut_job_config));
  }

  ~CompileScope() {
    CHECK_JUST(of::JobBuildAndInferCtx_Close());
    CHECK_JUST(of::ThreadLocalScopeStackPop());
  }

 private:
  of::LazyMode::Guard lazy_mode_enabled_guard{true};

  void ConfigXrt(of::JobConfigProto& job_config, XrtKind kind) {
    if (kind == XrtKind::kTensorRT) {
#ifdef WITH_TENSORRT
      job_config.mutable_xrt_config()->set_use_tensorrt(true);
#else
      LOG(WARNING) << "XRT TensorRT is unavailable while tensorrt is enabled";
#endif
    } else if (kind == XrtKind::kOpenVino) {
#ifdef WITH_OPENVINO
      job_config.mutable_xrt_config()->set_use_openvino(true);
#else
      LOG(WARNING) << "XRT OpenVINO is unavailable while openvino is enabled";
#endif
    }
  }
};

std::shared_ptr<of::one::TensorTuple> ConvertToTensorTuple(
    const std::vector<std::shared_ptr<of::one::Tensor>>& tensors) {
  auto tensor_tuple = std::make_shared<of::one::TensorTuple>();
  for (const auto& tensor : tensors) { tensor_tuple->emplace_back(tensor); }
  return tensor_tuple;
}

std::string GetDeviceTag(const Device& device) { return device.type(); }

template<class T1, class T2>
const std::pair<std::vector<T1>, std::vector<T2>> Unzip(const of::HashMap<T1, T2>& hash_map) {
  std::vector<T1> vec1;
  std::vector<T2> vec2;
  for (const auto& entry : hash_map) {
    vec1.emplace_back(entry.first);
    vec2.emplace_back(entry.second);
  }
  return std::make_pair(vec1, vec2);
}

Shape OfShapeToOfApiShape(const of::Shape& of_shape) {
  std::vector<int64_t> dims(of_shape.dim_vec().begin(), of_shape.dim_vec().end());
  return Shape(dims);
}

}  // namespace

class Graph::GraphImpl final {
 public:
  explicit GraphImpl(const std::string& model_path, const Device& device = Device("cpu"));

  GraphImpl(const GraphImpl& graph) = delete;
  GraphImpl(GraphImpl&& graph) = default;

  ~GraphImpl();

  GraphImpl& operator=(const GraphImpl& graph) = delete;
  GraphImpl& operator=(GraphImpl&& graph) = default;

  InputOutputInfos GetInputInfos();
  InputOutputInfos GetOutputInfos();
  std::vector<Tensor> Forward(const std::vector<Tensor>& inputs);
  void set_batch_size(int batch_size) { batch_size_ = batch_size; }
  void enable_tensorrt() { xrt_kind_ = XrtKind::kTensorRT; }
  void enable_openvino() { xrt_kind_ = XrtKind::kOpenVino; }

 private:
  of::Maybe<void> CollectInputOutputInfos();
  of::Maybe<void> Compile(const std::vector<Tensor>& inputs);
  of::Maybe<std::vector<Tensor>> Run(const std::vector<Tensor>& inputs) const;
  of::Maybe<void> AddOp(of::OperatorConf op_conf);
  of::Maybe<void> BuildGraph();
  of::Maybe<void> LoadCheckpoint();
  of::Maybe<void> RegisterTensors(const std::vector<Tensor>& inputs);

  std::shared_ptr<of::NNGraph> graph_ = nullptr;
  std::string model_path_;
  bool is_compiled_ = false;
  int batch_size_ = 0;
  XrtKind xrt_kind_ = XrtKind::kNone;
  Device device_;
  of::Job job_;

  InputOutputInfos input_infos_;
  InputOutputInfos output_infos_;
  of::HashMap<std::string, std::shared_ptr<of::one::Tensor>> output_name_to_tensor_;
  of::HashMap<std::string, std::shared_ptr<of::one::Tensor>> variable_op_name_to_tensor_;
  std::shared_ptr<of::one::TensorTuple> output_tensor_tuple_;
  std::shared_ptr<of::one::TensorTuple> parameter_tensor_tuple_;
};

Graph::Graph(const std::string& model_path, const Device& device)
    : graph_(std::make_unique<GraphImpl>(model_path, device)) {}

Graph::~Graph() = default;

Graph::Graph(Graph&& graph) noexcept : graph_(std::move(graph.graph_)) {}

Graph& Graph::operator=(Graph&& graph) noexcept {
  if (&graph == this) { return *this; }
  graph_ = std::move(graph.graph_);
  return *this;
}

InputOutputInfos Graph::GetInputInfos() { return graph_->GetInputInfos(); }

InputOutputInfos Graph::GetOutputInfos() { return graph_->GetOutputInfos(); }

IValue Graph::Forward(const IValue& inputs) {
  std::vector<Tensor> input_tensors;
  if (inputs.IsNone()) {
    // do nothing
  } else if (inputs.IsTensor()) {
    input_tensors.emplace_back(inputs.ToTensor());
  } else if (inputs.IsTensorVector()) {
    input_tensors = inputs.ToTensorVector();
  } else {
    LOG(WARNING) << "Graph currently only support types: Tensor/vector(Tensor)/None";
  }

  std::vector<Tensor> output_tensors = graph_->Forward(input_tensors);
  if (output_tensors.empty()) {
    return IValue{};
  } else if (output_tensors.size() == 1) {
    return IValue(output_tensors.at(0));
  } else {
    return IValue(output_tensors);
  }
}

void Graph::set_batch_size(int batch_size) { graph_->set_batch_size(batch_size); }

void Graph::enable_tensorrt() { graph_->enable_tensorrt(); }

void Graph::enable_openvino() { graph_->enable_openvino(); }

Graph Graph::Load(const std::string& model_path, const Device& device) {
  Graph graph(model_path, device);
  return graph;
}

Graph::GraphImpl::GraphImpl(const std::string& model_path, const Device& device)
    : model_path_(model_path), device_(device) {
  CHECK_JUST(of::LoadJobFromIR(&job_, model_path + "/model.mlir"));
  CollectInputOutputInfos();
  if (of::ParseBooleanFromEnv("ONEFLOW_SERVING_DEBUG", false)) { LOG(ERROR) << job_.DebugString(); }
  job_.mutable_job_conf()->mutable_predict_conf();
  job_.mutable_job_conf()->set_job_name(job_.mutable_job_conf()->job_name() + of::NewUniqueId());
  CHECK(of::Global<OneFlowEnv>::Get() != nullptr);
  graph_ = std::make_shared<of::NNGraph>(job_.job_conf().job_name(),
                                         of::Global<OneFlowEnv>::Get()->GetSessionCtx());
}

InputOutputInfos Graph::GraphImpl::GetInputInfos() { return input_infos_; }

InputOutputInfos Graph::GraphImpl::GetOutputInfos() { return output_infos_; }

of::Maybe<void> Graph::GraphImpl::CollectInputOutputInfos() {
  const of::OpGraph op_graph(job_);
  size_t input_order = 0;
  size_t output_order = 0;
  op_graph.TopoForEachNode([&](const of::OpNode* node) -> of::Maybe<void> {
    const of::OperatorConf& op_conf = node->op().op_conf();
    if (op_conf.has_input_conf()) {
      of::InterfaceBlobConf blob_conf = op_conf.input_conf().blob_conf();
      input_infos_[op_conf.name()] =
          InputOutputAttribute(static_cast<DType>(blob_conf.data_type()),
                               OfShapeToOfApiShape(of::Shape(blob_conf.shape())), input_order);
      input_order += 1;
    } else if (op_conf.has_output_conf()) {
      of::InterfaceBlobConf blob_conf = op_conf.output_conf().blob_conf();
      output_infos_[op_conf.name()] =
          InputOutputAttribute(static_cast<DType>(blob_conf.data_type()),
                               OfShapeToOfApiShape(of::Shape(blob_conf.shape())), output_order);
      output_order += 1;
    }
    return of::Maybe<void>::Ok();
  });
  return of::Maybe<void>::Ok();
}

std::vector<Tensor> Graph::GraphImpl::Forward(const std::vector<Tensor>& inputs) {
  if (!is_compiled_) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    Compile(inputs).GetOrThrow();
    is_compiled_ = true;
  }
  return Run(inputs).GetOrThrow();
}

of::Maybe<void> Graph::GraphImpl::Compile(const std::vector<Tensor>& inputs) {
  JUST(BuildGraph());
  JUST(LoadCheckpoint());
  JUST(RegisterTensors(inputs));
  JUST(graph_->CompileAndInitRuntime());
  return of::Maybe<void>::Ok();
}

of::Maybe<std::vector<Tensor>> Graph::GraphImpl::Run(const std::vector<Tensor>& inputs) const {
  const auto input_tensor_tuple = std::make_shared<of::one::TensorTuple>();
  for (const auto& tensor : inputs) { input_tensor_tuple->emplace_back(tensor.tensor_); }

  JUST(of::RunLazyNNGraph(*input_tensor_tuple, *output_tensor_tuple_, *parameter_tensor_tuple_,
                          graph_));
  JUST(of::SoftSyncNNGraphBuffers(*output_tensor_tuple_, graph_));

  std::vector<Tensor> outputs;
  for (const auto& tensor : *output_tensor_tuple_) { outputs.emplace_back(Tensor(tensor)); }
  return outputs;
}

of::Maybe<void> Graph::GraphImpl::AddOp(of::OperatorConf op_conf) {
  {
    const std::shared_ptr<of::Scope> scope = JUST(of::GetCurrentScope());
    op_conf.set_scope_symbol_id(scope->symbol_id().value_or(0));
  }
  op_conf.set_device_tag(GetDeviceTag(device_));
  if (batch_size_ > 0 && op_conf.has_input_conf()) {
    op_conf.mutable_input_conf()->mutable_blob_conf()->mutable_shape()->mutable_dim()->Set(
        0, batch_size_);
  }
  auto* ctx = JUST(of::GetCurInferCtx());
  JUST(ctx->AddAndInferConsistentOp(op_conf));
  return of::Maybe<void>::Ok();
}

of::Maybe<void> Graph::GraphImpl::BuildGraph() {
  CompileScope build_graph_scope(job_.job_conf(), *device_.device_->shared_from_symbol(),
                                 xrt_kind_);
  {
    const of::OpGraph op_graph(job_);
    op_graph.TopoForEachNode([&](const of::OpNode* node) -> of::Maybe<void> {
      const of::OperatorConf& op_conf = node->op().op_conf();
      JUST(AddOp(op_conf));
      if (op_conf.has_variable_conf()) {
        const of::LazyMode::Guard lazy_mode_disabled_guard{false};
        const of::VariableOpConf& variable_conf = op_conf.variable_conf();
        variable_op_name_to_tensor_[op_conf.name()] = JUST(of::one::functional::Empty(
            of::Shape(variable_conf.shape()),
            JUST(of::DType::Get(static_cast<of::DataType>(variable_conf.data_type()))),
            *device_.device_, /*pin_memory=*/false));
      }
      return of::Maybe<void>::Ok();
    });
  }
  JUST(of::CurJobBuildAndInferCtx_Complete());
  {
    const std::shared_ptr<of::Job> complete_job = JUST(of::GetCurrentJob());
    const of::OpGraph complete_graph(*complete_job);
    complete_graph.TopoForEachNode([&](const of::OpNode* node) -> of::Maybe<void> {
      const of::LazyMode::Guard lazy_mode_disabled_guard{false};
      const of::OperatorConf& op_conf = node->op().op_conf();
      if (op_conf.has_output_conf()) {
        of::InterfaceBlobConf blob_conf = op_conf.output_conf().blob_conf();
        if (batch_size_ > 0) {
          const std::string input_lbi_str = op_conf.output_conf().in();
          const of::LogicalBlobId input_lbi = of::GenLogicalBlobId(input_lbi_str);
          int64_t batch_size = node->LogicalBlobDesc4Lbi(input_lbi).shape().At(0);
          blob_conf.mutable_shape()->set_dim(0, batch_size);
        }
        output_name_to_tensor_[op_conf.name()] = JUST(of::one::functional::Empty(
            of::Shape(blob_conf.shape()),
            JUST(of::DType::Get(static_cast<of::DataType>(blob_conf.data_type()))),
            *device_.device_, /*pin_memory=*/false));
      }
      return of::Maybe<void>::Ok();
    });
  }
  return of::Maybe<void>::Ok();
}

of::Maybe<void> Graph::GraphImpl::LoadCheckpoint() {
  for (const auto& variable_op_name_and_tensor : variable_op_name_to_tensor_) {
    const auto& variable_op_name = variable_op_name_and_tensor.first;
    const auto& variable_tensor = variable_op_name_and_tensor.second;
    const std::string variable_filename = model_path_ + "/" + variable_op_name + "/out";
    const std::string buffer = [&]() {
      std::ifstream variable_file(variable_filename, std::ios::binary);
      CHECK(variable_file.is_open());
      std::stringstream ss;
      ss << variable_file.rdbuf();
      return ss.str();
    }();
    const auto& callback = [&](uint64_t of_blob_ptr) {
      CHECK_JUST(of::BlobBufferCopyUtil<void>::From(
          of_blob_ptr, buffer.data(),
          variable_tensor->shape()->elem_cnt()
              * of::GetSizeOfDataType(variable_tensor->dtype()->data_type())));
    };
    JUST(of::one::SyncAccessTensorWithTimeOut(variable_tensor, callback, "mut"));
  }

  return of::Maybe<void>::Ok();
}

of::Maybe<void> Graph::GraphImpl::RegisterTensors(const std::vector<Tensor>& inputs) {
  {
    std::vector<std::string> input_op_names(inputs.size());
    std::vector<std::shared_ptr<of::one::Tensor>> input_tensors(inputs.size());
    for (const auto& input_info : input_infos_) {
      size_t index = input_info.second.input_output_index_;
      input_op_names[index] = input_info.first;
      input_tensors[index] = inputs.at(index).tensor_;
    }
    JUST(graph_->RegisterInputOpNamesAndTensors(input_op_names, input_tensors));
  }
  {
    const auto& pair = Unzip(output_name_to_tensor_);
    const std::vector<std::string>& output_op_names = pair.first;
    const std::vector<std::shared_ptr<of::one::Tensor>>& output_tensors = pair.second;
    JUST(graph_->RegisterOutputOpNamesAndTensors(output_op_names, output_tensors));
    output_tensor_tuple_ = ConvertToTensorTuple(output_tensors);
  }
  {
    const auto& pair = Unzip(variable_op_name_to_tensor_);
    const std::vector<std::string>& variable_op_names = pair.first;
    const std::vector<std::shared_ptr<of::one::Tensor>>& variable_tensors = pair.second;
    JUST(graph_->RegisterVariableOpNamesAndTensors(variable_op_names, variable_tensors));
    parameter_tensor_tuple_ = ConvertToTensorTuple(variable_tensors);
  }
  return of::Maybe<void>::Ok();
}

Graph::GraphImpl::~GraphImpl() { of::vm::ClusterSync().GetOrThrow(); }

}  // namespace oneflow_api
