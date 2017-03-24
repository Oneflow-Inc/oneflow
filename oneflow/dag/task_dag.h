#ifndef _DAG_TASK_DAG_H_
#define _DAG_TASK_DAG_H_
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "common/shape.h"
#include "common/str_util.h"
#include "common/blob_name_converter.h"
#include "common/task_type.h"
#include "common/string_pair.h"
#include "context/one.h"
#include "context/strategy_descriptor.h"
#include "context/config_parser.h"
#include "context/id_map.h"
#include "dag/dag_node.h"
#include "dag/dag.h"
#include "dag/actor_dag.h"
#include "dag/node_meta.h"
#include "dag/segment_dag.h"
#include "dag/dag_builder.h"
#include "dag/dag_iterator.h"
#include "dag/blob_info_manager.h"
#include "dag/register_info_manager.h"
#include "layers/base_layer.h"
#include "memory/blob.h"
#include <glog/logging.h>

#include "dag/register_info.h"
#include "task/job_manager.h"

// A DAG consists of layers inside a particular task.
namespace oneflow {
class BlobMeta;

template <typename Dtype>
class LayerMeta;

using SegmentSegmentPair = StringPair;

template <typename Dtype>
class DagBuilder;

template <typename Dtype>
class TaskDag : public Dag<BlobMeta, LayerMeta<Dtype>> {
  friend class DagIterator<TaskDag<Dtype>>;
  friend class DagIterator<TaskDag<Dtype>, true>;
  friend class DagReverseIterator<TaskDag<Dtype>>;
  friend class DagReverseIterator<TaskDag<Dtype>, true>;
  using Dag<BlobMeta, LayerMeta<Dtype>>::name_;
  using Dag<BlobMeta, LayerMeta<Dtype>>::path_type_;
  using DNode = DataNode<BlobMeta>;
  using ONode = OpNode<LayerMeta<Dtype>>;
  using Dag<BlobMeta, LayerMeta<Dtype>>::Dag;
  using Dag<BlobMeta, LayerMeta<Dtype>>::op_name_to_node_;
  using Dag<BlobMeta, LayerMeta<Dtype>>::data_name_to_node_;
  using Dag<BlobMeta, LayerMeta<Dtype>>::AddStartAndEndNodes;
  using Dag<BlobMeta, LayerMeta<Dtype>>::PostProcessing;
  using Dag<BlobMeta, LayerMeta<Dtype>>::NewDataNode;
  using Dag<BlobMeta, LayerMeta<Dtype>>::GetDataNode;
 public:
  TaskDag(const DagBuilder<Dtype>& dag_builder,
    TaskType type,
    int32_t task_id,
    PathType path_type,
    const std::string& actor_name,
    bool is_forward);
  ~TaskDag();

  TaskType task_type() const { return type_; }
  int32_t task_id() const { return task_id_; }
  PathType path_type() const { return path_type_; }
  const std::string& actor_name() const { return this->name_; }
  bool is_forward() const { return is_forward_;  }
  bool is_h2d() const { CHECK(type_ == TaskType::kCopyTask); return is_h2d_; }
  bool is_net_receiver() const {
    CHECK(type_ == TaskType::kNetTask); return is_net_receiver_;
  }

  std::string task_blob_from_layer_blob(const std::string& layer_blob) const;
  std::string logical_blob_from_task_blob(const std::string& task_blob) const;
  std::string register_blob_from_layer_blob(const std::string& layer_blob) const;

  // 1, Establish the DAG consisting of OpNodes and DataNodes.
  // 2, Register all the required blobs to |blob_info_manager_|.
  void Build();

  // Collect all the RegisterInfos produced by this TaskDag. The RegisterInfo
  // can be consumed by another TaskDag in the same path (InPath) or in another
  // path (CrossPath).
  virtual void AddProducedRegisterInfos() = 0;

  // Collect all the RegisterInfos who are produced in the same path and
  // meanwhile are consumed by this TaskDag, which may or may not be an
  // 'immediate' consumer.
  virtual void AddConsumedRegisterInfosInPath() = 0;

  //// For cross-path sharing

  // (1) The consumer adds its dependency (by calling |AddConsumedGroupId|) to
  // the produced RegisterInfo with |produced_group_id|, which is ready even
  // before establishing the dependency.
  // (2) The consumer establishes the mapping between some of its blobs and
  // the RegisterInfo with |produced_group_id|.
  // (3) The consumer needs to return an incomplete RegisterInfo with type of
  // |consumed_register_type|, which embodies its requirements to the producer.
  // The incomplete RegisterInfo will be merged with another incomplete
  // RegisterInfo held by the producer.
  // ComputeTaskDag and CopyTaskDag need to override the implementation.
  virtual RegisterInfo CompleteConsumedRegisterInfoCrossPath(
    RegisterType consumer_register_type, int64_t produced_group_id) {
    LOG(FATAL) << "Only kComputeTask and kCopyTask could act as a consumer in"
      << " a cross-path dependency";
    return RegisterInfo();
  };

  // Completes its produced RegisterInfo with |my_register_type|shared in
  // cross-path, by merging the incomplete RegisterInfo |other_register_info|
  RegisterInfo CompleteProducedRegisterInfoCrossPath(
    RegisterType my_register_type,
    const RegisterInfo& other_register_info);

  // The current TaskDag already produces a RegisterInfo with |my_register_type|.
  // This function disables it and replaces it with another RegisterInfo with
  // |other_group_id|.
  virtual RegisterInfo ReplaceProducedRegisterInfoCrossPath(
    RegisterType my_register_type, int64_t other_group_id) {
    LOG(FATAL) << "Only kCopyTask could act as a producer but not an owner in"
      << "a cross-path dependency";
    return RegisterInfo();
  }
  //// End of cross-path sharing

  void Setup();

  // Get all the immediate consumers in the same path. The number of returned
  // consumers should be exactly one for TaskDag other than kBoxingTask.
  // With "immediate", we mean there is a direct connection between producer
  // and consumer in the ActorDag. A counter-example for non-immediate
  // connection is the producer-consumer relationship between a forward compute
  // TaskDag and its backward correspondence. Though the backward compute
  // TaskDag consumes the RegisterType::kDataType register, this dependency is
  // not characterized in the ActorDag.
  std::vector<std::string> GetImmediateConsumerNamesInPath() const;
  // Get all the immediate producers in the same path. The number of returned
  // producers should be exactly one for TaskDag other than kBoxingTask.
  std::vector<std::string> GetImmediateProducerNamesInPath() const;

  // Get the produced group id which is immediately consumed by |consumer_name|
  // from the same path. kBoxingTask needs to override this function.
  // See the comment to |GetImmediateConsumerNamesInPath| for the meaning of
  // "immediate".
  virtual int64_t GetImmediateProducedGroupIdInPath(
    const std::string& consumer_name) const;

  // Get all the group ids consumed by this TaskDag
  std::vector<int64_t> GetConsumedGroupIds() const;
  // Get all the group ids produced by this TaskDag
  std::vector<int64_t> GetProducedGroupIds() const;

  // The consumer queries to the producer for the group_id of a particular type
  // of RegisterInfo. Useful for non-immediate producer-consumer dependency:
  // (1) non-immediate dependency in the same path, e.g., the
  // RegisterType::kDataType RegisterInfo between the forward-backward TaskDags.
  // (2) the cross-path dependency.
  // For the cases other than the above two, please use
  // |TaskDag.GetImmediateProducedGroupIdInPath(const std::string& consumer_name)|.
  int64_t GetProducedGroupIdByRegisterType(RegisterType type) const;

  std::vector<int64_t> GetGroupIdsConsumedByOthers() const;

  // Get all the consumers of a RegisterInfo indicated with |group_id|.
  // (1) It is possible that the consumer comes from another path.
  // (2) It is possible that the |group_id| is not produced by current TaskDag.
  std::vector<int32_t> GetConsumersOfGroupId(int64_t group_id) const;

  // Get how many registers will be allocated in the group |group_id|
  int32_t GetProducedGroupSize(int64_t group_id) const;

  // Get the RegisterInfo corresponding to |group_id|.
  const RegisterInfo& GetProducedRegisterInfo(int64_t group_id) const;

  // Get the topologically sorted layers in this TaskDag
  std::vector<std::shared_ptr<BaseLayer<Dtype>>> GetOrderedLayers() const;

  //// Get an ordered array of layer_blobs in this TaskDag. The order is
  //// determined by two factors: (1) the topologically sorted layers which
  //// contain the layer_blobs; (2) the blobs contained by a particular layer.
  std::vector<std::string> GetLayerBlobsInExecution() const;

  // Given the |task_blob| name, get the blob shape.
  Shape GetBlobShapeFromTaskBlobName(const std::string& task_blob) const;

  virtual void RegisterProducer() {}

  // Tell this TaskDag that another task whose task_id is |consumer_id| will
  // consume its produced RegisterInfo indicated by |group_id|.
  void RegisterConsumer(int32_t consumer_id, int64_t group_id);

  // This TaskDag needs to consume the RegisterInfo indicated by |group_id|.
  void AddConsumedGroupId(int64_t group_id);

  // Get the |task_blob| name of a blob who has the |logical_blob| name in
  // the RegisterInfo of |group_id|.
  std::string GetTaskBlobFromRegisterInfo(int64_t group_id,
    const std::string& logical_blob);

  // Whether this TaskDag is a placeholder in a cross-path dependency. A TaskDag
  // could be in the following cases:
  // (1) no cross-path dependency -> is_placeholder_ == false;
  // (2) has cross-path dependency, either be a producer or a consumer, or be
  //     both. is_placeholder_ is false, or true (for producer or for consumer,
  //     but not both).
  bool IsPlaceholder() const { return is_placeholder_; }
  void SetIsPlaceholder(bool value) { is_placeholder_ = value; }

 protected:
  const DagBuilder<Dtype>& dag_builder_;
  TaskType type_;
  int32_t task_id_;
  bool is_forward_;  // valid only if path_type_ == kDataPath
  bool setup_{ false };
  bool is_h2d_;      // valid only if type_ == TaskType::kCopyTask
  bool is_net_receiver_; // valid only if type_ == TaskType::kNetTask

  // Some task does not perform any actual work, it is just used to as a
  // placeholder for DagBuilder to generate ActorDag from a LogicalDag. Similar
  // to the notion of stub/service in RPC (Remote Procedure Call), we call such
  // task as a stub task, in other words, it is a proxy of another task who does
  // actual stuff, namely service task.
  // Stub TaskDag does not own any produced Register. However, we still generate
  // some produced RegisterInfos if it has. Its position in the ActorDag helps
  // to build the message topology of the service task.
  // We need to identify the service task for a stub task.
  // If |is_placeholder_| is true, the task will not generate Register according
  // to the produced RegisterInfo.
  // If |is_placeholder_| is true, the task will re-direct all the producer
  // and consumer dependencies to the corresponding service task.
  bool is_placeholder_;

  BlobInfoManager blob_info_manager_;
  RegisterInfoManager register_info_manager_;

  // BuildForward should establish a forward task's DAG consisting of OpNodes
  // and DataNodes. At the same time, it registers all the kInput/kOutput blobs
  // required by this task to |blob_info_manager_|.
  // Each sub-class of TaskDag has its own implementation of BuildForward.
  virtual void BuildForward() = 0;

  // BuildBackward should establish a backward task's DAG consisting of OpNodes
  // and DataNodes. At the same time, it registers all the kInput/kOutput/
  // kInDiff/kOutDiff blobs required by this task to |blob_info_manager_|.
  // kCopyTask, kBoxingTask kNetTaskDag share the same way for building backward
  // TaskDag. However, kComputeTaskDag needs to override this function.
  virtual void BuildBackward();

  // For a particular task, besides the kInput/kOutput blobs registered in the
  // BuildForward or BuildBackward phase, it may also need other types of blobs.
  // We need to register other required layer blobs not added yet. Override this
  // function if necessary, such as kModel blobs in kComputeTask, kOther blobs
  // in kBoxingTask.
  virtual void RegisterNonInputOutputBlobs() {};

  // kDataTask and kNetTask with 'in_net' property can not be setup with the
  // regular routine, we need to override |ForwardSetup|.
  virtual void ForwardSetup();

  void BackwardSetup();

  // To infer the shapes of blobs inside a TaskDag, we need to firstly fill
  // the shapes of some seeding blobs, such as the input blobs of the TaskDag.
  // |ForwardSetupPrepare| is designed for this purpose.
  void ForwardSetupPrepare();

  // Secondly, with a topological-sorted traversal, we could infer the shape of
  // other blobs.
  void ForwardSetupInternal();

  void ForwardSetupDataNode(DNode* dnode);
  void ForwardSetupOpNode(ONode* onode);

  OpNode<LayerMeta<Dtype>>* AddOpNode(
    const std::string& op_name,
    const std::string& op_type,
    const std::string& op_param_str);
  DataNode<BlobMeta>* AddDataNode(const std::string& data_name);

  OpNode<LayerMeta<Dtype>>* AddBackwardOpNode(
    const std::string& op_name,
    std::shared_ptr<LayerMeta<Dtype>> forward_layer_meta);

  using BlobFilter = std::function<bool(const std::string&)>;
  const BlobFilter null_filter_ = [](const std::string& blob_var) {
    return true;
  };
  void AddBlobsToProducedRegisterInfo(
    const std::string& op_name,
    const std::vector<std::string>& layer_vars,
    RegisterInfo* register_info,
    EnvelopeFlag envelope_flag,
    BlobFilter filter);

  void AddBlobsToConsumedRegisterInfo(
    const std::string& op_name,
    const std::vector<std::string>& layer_vars,
    std::shared_ptr<TaskDag<Dtype>> producer_task_dag,
    int64_t group_id,
    BlobFilter filter);

  void AddBlobsToConsumedRegisterInfoCrossPath(
    const std::string& op_name,
    const std::vector<std::string>& layer_vars,
    RegisterInfo* register_info,
    int64_t group_id,
    EnvelopeFlag envelope_flag);
};

template <typename Dtype>
TaskDag<Dtype>::TaskDag(const DagBuilder<Dtype>& dag_builder, TaskType type,
  int32_t task_id, PathType path_type, const std::string& actor_name,
  bool is_forward)
  : dag_builder_(dag_builder), type_(type), task_id_(task_id),
    is_forward_(is_forward), is_placeholder_(false),
    Dag<PathType, const std::string&>::Dag(path_type, actor_name), is_h2d_(false), is_net_receiver_(false) { }

template <typename Dtype>
TaskDag<Dtype>::~TaskDag() {}

template <typename Dtype>
std::string TaskDag<Dtype>::task_blob_from_layer_blob(
  const std::string& layer_blob) const {
  return blob_info_manager_.task_blob_from_layer_blob(layer_blob);
}

template <typename Dtype>
std::string TaskDag<Dtype>::logical_blob_from_task_blob(
  const std::string& task_blob) const {
  return blob_info_manager_.logical_blob_from_task_blob(task_blob);
}

template <typename Dtype>
std::string TaskDag<Dtype>::register_blob_from_layer_blob(
  const std::string& layer_blob) const {
  return blob_info_manager_.register_blob_from_layer_blob(layer_blob);
}

template <typename Dtype>
void TaskDag<Dtype>::Build() {
  // 1, Build the DAG consisting of OpNodes and DataNodes; Register the kInput
  // and kOutput blobs to |blob_info_manager_|.
  if (is_forward_) {
    this->BuildForward();
  } else {
    this->BuildBackward();
  }
  AddStartAndEndNodes();
  PostProcessing();

  // 2, Register the non-kInput/kOutput blobs to |blob_info_manager_|.
  RegisterNonInputOutputBlobs();
}

template <typename Dtype>
void TaskDag<Dtype>::BuildBackward() {
  // Workable for kCopyTaskDag, kBoxingTaskDag, kNetTaskDag.
  // (1) The backward TaskDag share the same operators (i.e., layers) with the
  // forward TaskDag.
  // (2) The backward TaskDag has its own blobs distinct from the forward one.
  auto&& actor_dag = dag_builder_.actor_dag();
  auto forward_task_name = actor_dag->GetForwardTaskName(name_);
  auto forward_task_id = actor_dag->GetTaskID(forward_task_name);
  auto backward_task_id = actor_dag->GetTaskID(name_);
  auto forward_task_dag = dag_builder_.GetTaskDag(forward_task_id);

  // 'Clone' the forward_task_dag to this backward TaskDag by reversely
  // traversing the forward_task_dag. The forward-backward TaskDags pair share
  // the operators (i.e., layers).
  DagReverseIterator<TaskDag<Dtype>, true> dag_iterator(*forward_task_dag);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kOpNode) continue;
    auto layer_node
      = dynamic_cast<const OpNode<LayerMeta<Dtype>>*>(current_node);
    auto layer_name = current_node->node_name();
    auto layer_meta = layer_node->op();
    // Directly re-use the layer created in |forward_task_dag|
    auto op_node = AddBackwardOpNode(layer_name, layer_meta);
    auto op_meta = op_node->op();
    auto layer = op_meta->layer();

    std::vector<DNode*> input_nodes;
    auto outputs = layer->GetOutputVars();
    for (auto output : outputs) {
      // Get its corresponding layer_blob name in forward-direction, the name
      // is also used in this backward-direction task dag.
      auto layer_blob = strings::full_blob_name_in_layer(layer_name, output);
      auto task_blob = forward_task_dag->task_blob_from_layer_blob(layer_blob);
      auto logical_blob
        = forward_task_dag->logical_blob_from_task_blob(task_blob);
      // Construct its name in dag according to its correspondence in
      // forward_task_dag.
      auto task_diff = strings::get_diff_blob_name(task_blob);
      auto data_node_it = data_name_to_node_.find(task_diff);
      DNode* data_node;
      if (data_node_it == data_name_to_node_.end()) {
        data_node = AddDataNode(task_diff);
      } else {
        data_node = data_node_it->second;
      }
      input_nodes.push_back(data_node);
      blob_info_manager_.RegisterBlob(layer_blob, task_diff, logical_blob, true);
    }

    std::vector<DNode*> output_nodes;
    auto inputs = layer->GetInputVars();
    for (auto input : inputs) {
      // Get its corresponding layer_blob name in forward-direction, the name is
      // also used in this backward-direction task dag.
      auto layer_blob = strings::full_blob_name_in_layer(layer_name, input);
      auto task_blob = forward_task_dag->task_blob_from_layer_blob(layer_blob);
      auto logical_blob
        = forward_task_dag->logical_blob_from_task_blob(task_blob);
      auto task_diff = strings::get_diff_blob_name(task_blob);
      auto data_node = AddDataNode(task_diff);
      output_nodes.push_back(data_node);
      blob_info_manager_.RegisterBlob(layer_blob, task_diff, logical_blob, false);

      // TODO(jiyuan): if this TaskDag is the last one in BP direction and this
      // blob is the last one. We don't need compute the |dag_blob_diff_name| at
      // all, neither need to allocate memory for it.
      // TODO(jiyuan): if the SplitLayer is for label-path, do not include them
      // in the backward TaskDag.
    }
    this->AddEdges(op_node, input_nodes, output_nodes);
  }
}

template <typename Dtype>
OpNode<LayerMeta<Dtype>>* TaskDag<Dtype>::AddOpNode(
  const std::string& op_name,
  const std::string& op_type,
  const std::string& op_param_str) {
  auto op_node = this->NewOpNode(op_name);
  auto&& layer_meta = op_node->mutable_op();
  layer_meta = std::make_shared<LayerMeta<Dtype>>(op_type);
  layer_meta->mutable_param_str() = op_param_str;

  auto&& layer = layer_meta->mutable_layer();
  layer = LayerRegistry<Dtype>::CreateLayer(op_type, op_name, op_param_str);
  layer->InitParam();

  auto it = op_name_to_node_.find(op_name);
  CHECK(it == op_name_to_node_.end()) << "Duplicate op_name: " << op_name;
  op_name_to_node_.insert({ op_name, op_node });
  return op_node;
}

template <typename Dtype>
OpNode<LayerMeta<Dtype>>* TaskDag<Dtype>::AddBackwardOpNode(
  const std::string& op_name,
  std::shared_ptr<LayerMeta<Dtype>> forward_layer_meta) {
  auto op_node = this->NewOpNode(op_name);
  auto&& layer_meta = op_node->mutable_op();
  layer_meta = forward_layer_meta;
  auto it = op_name_to_node_.find(op_name);
  CHECK(it == op_name_to_node_.end()) << "Duplicate op_name: " << op_name;
  op_name_to_node_.insert({ op_name, op_node });
  return op_node;
}

template <typename Dtype>
DataNode<BlobMeta>* TaskDag<Dtype>::AddDataNode(const std::string& data_name) {
  auto data_node = NewDataNode(data_name);
  auto node_id = data_node->node_id();
  auto&& blob_meta = data_node->mutable_data();
  blob_meta = std::make_shared<BlobMeta>(data_name);
  auto it = data_name_to_node_.find(data_name);
  CHECK(it == data_name_to_node_.end())
    << "Duplicate data_name: " << data_name;
  data_name_to_node_.insert({ data_name, data_node });
  return data_node;
}

template <typename Dtype>
void TaskDag<Dtype>::Setup() {
  if (is_forward_) {
    ForwardSetup();
  } else {
    BackwardSetup();
  }
  setup_ = true;
}

template <typename Dtype>
void TaskDag<Dtype>::ForwardSetup() {
  LOG(INFO) << "Setup: " << name_;
  ForwardSetupPrepare();
  ForwardSetupInternal();
}

template <typename Dtype>
void TaskDag<Dtype>::BackwardSetup() {
  DLOG(INFO) << "Setup: " << name_;
  // Actually do nothing in the backward TaskDag, the shape of data node is
  // inferred from its correspondence in forward TaskDag.
  // In a later stage, if the backward TaskDag needs to know the shape of a blob
  // other than kOutDiff/kInDiff, it needs to turn the corresponding forward
  // TaskDag again.
  auto actor_dag = dag_builder_.actor_dag();
  auto forward_task_name = actor_dag->GetForwardTaskName(name_);
  auto forward_task_id = actor_dag->GetTaskID(forward_task_name);
  auto backward_task_id = actor_dag->GetTaskID(name_);
  auto forward_task_dag = dag_builder_.GetTaskDag(forward_task_id);

  DagIterator<TaskDag<Dtype>> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() != NodeType::kDataNode) continue;
    auto diff_name = current_node->node_name();
    auto diff_node = dynamic_cast<DNode*>(current_node);
    CHECK_NOTNULL(diff_node);
    auto&& diff_meta = diff_node->mutable_data();
    auto data_name = strings::get_data_blob_name(diff_name);
    auto data_node = forward_task_dag->GetDataNode(data_name);
    diff_meta->mutable_shape() = data_node->data()->shape();
    DLOG(INFO) << "Node name: " << diff_name;
    DLOG(INFO) << "Node shape: " << diff_meta->shape().shape_string();
  }
}

template <typename Dtype>
void TaskDag<Dtype>::ForwardSetupPrepare() {
  auto id_map = oneflow::TheOne<Dtype>::id_map();
  auto actor_dag = dag_builder_.actor_dag();
  auto input_task_blobs = blob_info_manager_.input_task_blobs();

  for (auto &task_blob : input_task_blobs) {
    int64_t group_id = blob_info_manager_.group_id_of_task_blob(task_blob);
    // auto producer = blob_info_manager_.producer_name_from_task_blob(task_blob);
    auto register_blob
      = blob_info_manager_.register_blob_from_task_blob(task_blob);

    auto producer_task_id = id_map->task_id_from_group_id(group_id);
    auto producer_task_dag = dag_builder_.GetTaskDag(producer_task_id);
    Shape shape = producer_task_dag->GetBlobShapeFromTaskBlobName(register_blob);
    auto blob_node = GetDataNode(task_blob);
    blob_node->data()->mutable_shape() = shape;
  }
}

template <typename Dtype>
void TaskDag<Dtype>::ForwardSetupInternal() {
  // In topological order, set up the OpNode to infer various blobs' shapes
  DagIterator<TaskDag<Dtype>> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() == NodeType::kDataNode) {
      auto current_dnode = dynamic_cast<DNode*>(current_node);
      CHECK_NOTNULL(current_dnode);
      ForwardSetupDataNode(current_dnode);
    } else if (current_node->Type() == NodeType::kOpNode) {
      auto current_onode = dynamic_cast<ONode*>(current_node);
      CHECK_NOTNULL(current_onode);
      ForwardSetupOpNode(current_onode);
    } else {
      // Do nothing
    }
  }
}

template <typename Dtype>
void TaskDag<Dtype>::ForwardSetupDataNode(DNode* dnode) {
  auto&& blob_meta = dnode->data();
  const Shape& shape = blob_meta->shape();
  DLOG(INFO) << "Node name: " << blob_meta->name();
  DLOG(INFO) << "Node shape: " << shape.shape_string();
}

template <typename Dtype>
void TaskDag<Dtype>::ForwardSetupOpNode(ONode* onode) {
  auto &layer_meta = onode->op();
  auto&& layer = layer_meta->mutable_layer();
  auto layer_name = onode->node_name();

  std::shared_ptr<DataParam<Dtype>> data_param(layer->CreateDataParam());
  data_param->AllocateEmptyBlobs();
  // Set the input shape of |data_param|
  auto input_vars = layer->GetInputVars();
  for (auto& input_var : input_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, input_var);
    auto task_blob = blob_info_manager_.task_blob_from_layer_blob(layer_blob);
    auto dnode = GetDataNode(task_blob);
    auto&& blob_meta = dnode->data();
    data_param->SetShape(layer_blob, blob_meta->shape());
  }

  // Infer the other blobs' shape from input shape
  layer->InitFromInputShape(data_param.get());

  // According the inference, update the output's shape
  auto output_vars = layer->GetOutputVars();
  for (auto& output_var : output_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(layer_name, output_var);
    auto task_blob = blob_info_manager_.task_blob_from_layer_blob(layer_blob);
    auto dnode = GetDataNode(task_blob);
    auto&& blob_meta = dnode->mutable_data();
    blob_meta->mutable_shape() = data_param->GetShape(layer_blob);
  }

  // Update shapes in |blob_info_manager_|
  auto layer_data_param = layer->GetDataParam();
  auto data_param_blobs = layer_data_param->blob_names();
  for (auto& data_blob : data_param_blobs) {
    auto task_blob = blob_info_manager_.task_blob_from_layer_blob(data_blob);
    auto shape = layer_data_param->GetShape(data_blob);
    blob_info_manager_.SetBlobShape(task_blob, shape);
  }
  auto layer_model_param = layer->GetModelParam();
  auto model_param_blobs = layer_model_param->blob_names();
  for (auto& model_blob : model_param_blobs) {
    auto task_blob = blob_info_manager_.task_blob_from_layer_blob(model_blob);
    auto shape = layer_data_param->GetShape(model_blob);
    blob_info_manager_.SetBlobShape(task_blob, shape);
  }
}

template <typename Dtype>
Shape TaskDag<Dtype>::GetBlobShapeFromTaskBlobName(
  const std::string& task_blob) const {
  return blob_info_manager_.GetBlobShape(task_blob);
}

template <typename Dtype>
std::vector<int64_t> TaskDag<Dtype>::GetConsumedGroupIds() const {
  return register_info_manager_.GetConsumedGroupIds();
}

template <typename Dtype>
std::vector<std::string> TaskDag<Dtype>::GetImmediateConsumerNamesInPath() const {
  auto actor_dag = dag_builder_.actor_dag();
  auto consumer_names = actor_dag->GetSucceedingOpNodeNames(name_);
  return consumer_names;
}

template <typename Dtype>
std::vector<std::string> TaskDag<Dtype>::GetImmediateProducerNamesInPath() const {
  auto actor_dag = dag_builder_.actor_dag();
  auto preceding_op_names = actor_dag->GetPrecedingOpNodeNames(name_);
  return preceding_op_names;
}

template <typename Dtype>
int64_t TaskDag<Dtype>::GetImmediateProducedGroupIdInPath(
  const std::string& consumer_name) const {
  CHECK(type_ != TaskType::kBoxingTask);
  // The |consumer_name| is only used to check whether it is really an immediate
  // consumer of this TaskDag.
  auto consumer_names = GetImmediateConsumerNamesInPath();
  CHECK_EQ(consumer_names.size(), 1);
  CHECK(consumer_names[0] == consumer_name);

  // The following routine is based on two facts:
  // (1) For producer and consumer in the same path, the RegisterInfo between
  // them must be either RegisterType::kDataType or RegisterType::kDataDiffType.
  // (2) A TaskDag other than kBoxingTask has exactly one RegisterInfo for a
  // RegisterType. Instead, kBoxingTask may have multiple RegisterInfos with the
  // same RegisterType::kDataType. Therefore, kBoxingTask will override this
  // function with an alternative implementation.

  RegisterType type;
  if (is_forward_) {
    type = RegisterType::kDataType;
  } else {
    type = RegisterType::kDataDiffType;
  }
  return register_info_manager_.GetProducedGroupIdForNonBoxingTask(type);
}

template <typename Dtype>
std::vector<int64_t> TaskDag<Dtype>::GetProducedGroupIds() const {
  return register_info_manager_.GetProducedGroupIds();
}

template <typename Dtype>
std::vector<int64_t> TaskDag<Dtype>::GetGroupIdsConsumedByOthers() const {
  return register_info_manager_.GetGroupIdsConsumedByOthers();
}

template <typename Dtype>
std::vector<int32_t> TaskDag<Dtype>::GetConsumersOfGroupId(
  int64_t group_id) const {
  return register_info_manager_.GetConsumersOfGroupId(group_id);
}

template <typename Dtype>
int32_t TaskDag<Dtype>::GetProducedGroupSize(int64_t group_id) const {
  return register_info_manager_.GetProducedGroupSize(group_id);
}

template <typename Dtype>
const RegisterInfo& TaskDag<Dtype>::GetProducedRegisterInfo(
  int64_t group_id) const {
  return register_info_manager_.GetProducedRegisterInfo(group_id);
}

template <typename Dtype>
int64_t TaskDag<Dtype>::GetProducedGroupIdByRegisterType(
  RegisterType type) const {
  // kBoxingTask does not support this query
  CHECK(type_ != TaskType::kBoxingTask);
  return register_info_manager_.GetProducedGroupIdForNonBoxingTask(type);
}

template <typename Dtype>
void TaskDag<Dtype>::AddBlobsToProducedRegisterInfo(
  const std::string& op_name,
  const std::vector<std::string>& layer_vars,
  RegisterInfo* register_info,
  EnvelopeFlag envelope_flag,
  BlobFilter filter) {
  int64_t group_id = register_info->group_id();
  for (auto& layer_var : layer_vars) {
    if (filter(layer_var)) {
      auto layer_blob = strings::full_blob_name_in_layer(op_name, layer_var);
      auto task_blob = task_blob_from_layer_blob(layer_blob);
      register_info->AddEmptyBlob(task_blob, envelope_flag);
      blob_info_manager_.AddProducedBlobToRegister(layer_blob, group_id);
    }
  }
}

template <typename Dtype>
void TaskDag<Dtype>::AddBlobsToConsumedRegisterInfo(
  const std::string& op_name,
  const std::vector<std::string>& layer_vars,
  std::shared_ptr<TaskDag<Dtype>> producer_task_dag,
  int64_t group_id,
  BlobFilter filter) {
  for (auto& layer_var : layer_vars) {
    if (filter(layer_var)) {
      auto layer_blob = strings::full_blob_name_in_layer(op_name, layer_var);
      // The register is produced by other TaskDag, we need to obtain
      // |register_blob| with the help of |logical_blob|.
      auto task_blob = task_blob_from_layer_blob(layer_blob);
      auto logical_blob
        = blob_info_manager_.logical_blob_from_task_blob(task_blob);
      auto register_blob
        = producer_task_dag->GetTaskBlobFromRegisterInfo(group_id, logical_blob);
      blob_info_manager_.AddConsumedBlobToRegister(
        layer_blob, register_blob, group_id);
    }
  }
}

template <typename Dtype>
void TaskDag<Dtype>::AddBlobsToConsumedRegisterInfoCrossPath(
  const std::string& op_name,
  const std::vector<std::string>& layer_vars,
  RegisterInfo* register_info,
  int64_t group_id,
  EnvelopeFlag envelope_flag) {
  for (auto layer_var : layer_vars) {
    auto layer_blob = strings::full_blob_name_in_layer(op_name, layer_var);
    auto task_blob = task_blob_from_layer_blob(layer_blob);
    auto register_blob = task_blob;

    blob_info_manager_.AddConsumedBlobToRegister(
      layer_blob, register_blob, group_id);

    register_info->AddEmptyBlob(task_blob, envelope_flag);
  }
}

template <typename Dtype>
void TaskDag<Dtype>::RegisterConsumer(int32_t consumer_id,
  int64_t group_id) {
  register_info_manager_.AddConsumerOfGroupId(consumer_id, group_id);
}

template <typename Dtype>
std::string TaskDag<Dtype>::GetTaskBlobFromRegisterInfo(
  int64_t group_id, const std::string& logical_blob) {
  return blob_info_manager_.task_blob_from_logical_blob(group_id, logical_blob);
}

template <typename Dtype>
std::vector<std::shared_ptr<BaseLayer<Dtype>>>
TaskDag<Dtype>::GetOrderedLayers() const {
  std::vector<std::shared_ptr<BaseLayer<Dtype>>> ordered_layers;
  DagIterator<TaskDag<Dtype>, true> dag_iterator(*this);
  for (dag_iterator.First(); !dag_iterator.IsDone(); dag_iterator.Next()) {
    auto current_node = dag_iterator.CurrentNode();
    if (current_node->Type() == NodeType::kOpNode) {
      auto layer_node
        = dynamic_cast<const OpNode<LayerMeta<Dtype>>*>(current_node);
      ordered_layers.push_back(layer_node->op()->mutable_layer());
    }
  }
  return ordered_layers;
}

template <typename Dtype>
std::vector<std::string> TaskDag<Dtype>::GetLayerBlobsInExecution() const {
  return blob_info_manager_.layer_blobs_in_execution();
}

template <typename Dtype>
void TaskDag<Dtype>::AddConsumedGroupId(int64_t group_id) {
  register_info_manager_.AddConsumedGroupId(group_id);
}

template <typename Dtype>
RegisterInfo TaskDag<Dtype>::CompleteProducedRegisterInfoCrossPath(
  RegisterType my_register_type,
  const RegisterInfo& other_register_info) {
  return register_info_manager_.CompleteProducedRegisterInfoCrossPath(
    my_register_type, other_register_info);
}

}  // namespace oneflow
#endif  // _DAG_TASK_DAG_H_
