#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include "oneflow/core/graph/graph.h"

namespace oneflow {

namespace {

std::shared_ptr<Operator> ConstructOp(const OperatorConf& op_conf, DeviceType device_type) {
  OperatorConf dev_op_conf = op_conf;
  dev_op_conf.set_device_type(device_type);
  return ConstructOp(dev_op_conf);
}

std::function<const ParallelConf*(const std::string&)> MakeGetterParallelConf4OpName(
    const Placement& placement) {
  auto op_name2parallel_conf = std::make_shared<HashMap<std::string, const ParallelConf*>>();
  for (const auto& placement_group : placement.placement_group()) {
    for (const std::string& op_name : placement_group.op_set().op_name()) {
      const ParallelConf* parallel_conf = &placement_group.parallel_conf();
      CHECK(op_name2parallel_conf->emplace(op_name, parallel_conf).second);
    }
  }
  return [op_name2parallel_conf](const std::string& op_name) {
    return op_name2parallel_conf->at(op_name);
  };
}

std::function<ParallelConf*(const std::string&)> MakeGetterMutParallelConf4OpName(
    Placement* placement) {
  auto op_name2parallel_conf = std::make_shared<HashMap<std::string, ParallelConf*>>();
  FOR_RANGE(int, idx, 0, placement->placement_group_size()) {
    auto* placement_group = placement->mutable_placement_group(idx);
    for (const std::string& op_name : placement_group->op_set().op_name()) {
      ParallelConf* parallel_conf = placement_group->mutable_parallel_conf();
      CHECK(op_name2parallel_conf->emplace(op_name, parallel_conf).second);
    }
  }
  return [op_name2parallel_conf](const std::string& op_name) {
    return op_name2parallel_conf->at(op_name);
  };
}

class OpEdge;

class OpNode final : public Node<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpNode);
  explicit OpNode(const OperatorConf& op_conf, DeviceType device_type)
      : op_(ConstructOp(op_conf, device_type)), has_in_diff_(false) {}
  ~OpNode() = default;

  const Operator& op() const { return *op_; }
  bool HasBackward() const { return has_in_diff() || has_model_diff(); }
  bool has_in_diff() const { return has_in_diff_; }
  bool has_model_diff() const { return op().model_diff_bns().size() > 0; }
  void set_has_in_diff(bool has_in_diff) { has_in_diff_ = has_in_diff; }

 private:
  std::shared_ptr<Operator> op_;
  bool has_in_diff_;
};

class OpEdge final : public Edge<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpEdge);
  explicit OpEdge(const std::vector<LogicalBlobId>& lbis,
                  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns)
      : lbis_(lbis), lbi2ibns_(lbi2ibns) {}
  ~OpEdge() = default;

  const LogicalBlobId& SoleLbi() const {
    CHECK_EQ(lbis_.size(), 1);
    return lbis_.front();
  }

  const std::vector<LogicalBlobId>& lbis() const { return lbis_; }
  const HashMap<LogicalBlobId, std::vector<std::string>>& lbi2ibns() const { return lbi2ibns_; }

 private:
  std::vector<LogicalBlobId> lbis_;
  HashMap<LogicalBlobId, std::vector<std::string>> lbi2ibns_;
};

class OpGraph final : public Graph<OpNode, OpEdge> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OpGraph);
  explicit OpGraph(const JobDesc& job_desc) : job_desc_(job_desc) { Init(); }
  ~OpGraph() = default;

 private:
  void Init() {
    InitNodes();
    ForEachNode(
        [&](OpNode* node) { CHECK(op_name2op_node_.emplace(node->op().op_name(), node).second); });
    InitEdges();
    UpdateOpNodeHasInDiff();
  }
  void InitNodes() {
    auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job_desc_.placement());
    for (const auto& op_conf : job_desc_.dlnet_conf().op()) {
      ParallelDesc pr(*ParallelConf4OpName(op_conf.name()));
      OpNode* node = new OpNode(op_conf, pr.device_type());
      AddAllocatedNode(node);
    }
  }
  void InitEdges() {
    HashMap<LogicalBlobId, OpNode*> lbi2producer;
    ForEachNode([&](OpNode* op_node) {
      for (const auto& obn : op_node->op().output_bns()) {
        CHECK(lbi2producer.emplace(op_node->op().BnInOp2Lbi(obn), op_node).second);
      }
    });
    ForEachNode([&](OpNode* op_node) {
      HashMap<std::string, std::vector<LogicalBlobId>> producer_name2lbis;
      HashMap<std::string, HashMap<LogicalBlobId, std::vector<std::string>>>
          consumer_op_name2lbi2ibns;
      for (const auto& ibn : op_node->op().input_bns()) {
        const LogicalBlobId& lbi = op_node->op().BnInOp2Lbi(ibn);
        producer_name2lbis[lbi.op_name()].push_back(lbi);
        consumer_op_name2lbi2ibns[op_node->op().op_name()][lbi].push_back(ibn);
      }
      for (const auto& pair : producer_name2lbis) {
        const auto& lbis = pair.second;
        const auto& lbi2ibns = consumer_op_name2lbi2ibns.at(op_node->op().op_name());
        OpNode* producer = lbi2producer.at(lbis.at(0));
        Connect(producer, new OpEdge(lbis, lbi2ibns), op_node);
      }
    });
  }

  void UpdateOpNodeHasInDiff() {
    auto HasIndiff = [&](const OpNode* op_node) -> bool {
      for (OpEdge* edge : op_node->in_edges()) {
        if (edge->src_node()->has_in_diff()) { return true; }
        if (edge->src_node()->has_model_diff()) { return true; }
      }
      return false;
    };
    TopoForEachNode([&](OpNode* op_node) { op_node->set_has_in_diff(HasIndiff(op_node)); });
  }

  const JobDesc& job_desc_;
  HashMap<std::string, OpNode*> op_name2op_node_;
};

void GroupOpConfByProducerOpNameAndConsumerParallelDesc(
    HashMap<std::pair<std::string, ParallelDesc>, HashSet<OperatorConf*>>* grouped,
    DLNetConf* dlnet_conf,
    const std::function<const ParallelConf*(const std::string&)>& ParallelConf2OpName) {
  CHECK(grouped->empty());
  FOR_RANGE(int, idx, 0, dlnet_conf->op_size()) {
    OperatorConf* op_conf = dlnet_conf->mutable_op(idx);
    ParallelDesc pr(*ParallelConf2OpName(op_conf->name()));
    std::shared_ptr<Operator> op = ConstructOp(*op_conf, pr.device_type());
    for (const auto& ibn : op->input_bns()) {
      (*grouped)[std::make_pair(op->BnInOp2Lbi(ibn).op_name(), pr)].insert(op_conf);
    }
  }
}

void CollectInputLbiBlobNamesByProducerOpName(HashSet<std::string>* ret_blob_name,
                                              const HashSet<OperatorConf*>& op_confs,
                                              DeviceType device_type,
                                              const std::string& producer_op_name) {
  CHECK(ret_blob_name->empty());
  for (const auto* op_conf : op_confs) {
    std::shared_ptr<Operator> op = ConstructOp(*op_conf, device_type);
    for (const auto& ibn : op->input_bns()) {
      LogicalBlobId lbi = op->BnInOp2Lbi(ibn);
      if (lbi.op_name() == producer_op_name) { ret_blob_name->insert(lbi.blob_name()); }
    }
  }
}

}  // namespace

float JobDesc::lazy_reduce_ratio() const {
  float ratio = job_conf_.other().lazy_reduce_ratio();
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  return ratio;
}

float JobDesc::reduce_model_update_overlapping_ratio() const {
  float ratio = job_conf_.other().reduce_model_update_overlapping_ratio();
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  return ratio;
}

int64_t JobDesc::piece_num_of_experiment_phase() const {
  return job_conf_.other().exp_run_conf().piece_num_of_experiment_phase();
}

bool JobDesc::enable_experiment_run() const {
  return job_conf_.other().exp_run_conf().enable_experiment_run();
}

size_t JobDesc::persistence_buf_byte() const {
  return job_conf_.other().persistence_buf_mbyte() * 1024 * 1024;
}

size_t JobDesc::reserved_host_mem_byte() const {
  return job_conf_.other().reserved_host_mem_mbyte() * 1024 * 1024;
}

size_t JobDesc::reserved_device_mem_byte() const {
  return job_conf_.other().reserved_device_mem_mbyte() * 1024 * 1024;
}

bool JobDesc::save_downloaded_file_to_local_fs() const {
  return job_conf_.other().save_downloaded_file_to_local_fs();
}

size_t JobDesc::rdma_mem_block_byte() const {
  return job_conf_.other().rdma_mem_block_mbyte() * 1024 * 1024;
}

size_t JobDesc::rdma_recv_msg_buf_byte() const {
  return job_conf_.other().rdma_recv_msg_buf_mbyte() * 1024 * 1024;
}

const std::string& JobDesc::MdSaveSnapshotsPath() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().model_save_snapshots_path();
}
int32_t JobDesc::NumOfBatchesInSnapshot() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().num_of_batches_in_snapshot();
}
int64_t JobDesc::TotalBatchNum() const { return job_conf_.other().total_batch_num(); }
const InitializerConf* JobDesc::DefaultInitializerConf() const {
  CHECK(IsTrain());
  return GetMsgPtrFromPbMessage<InitializerConf>(job_conf_.other().train_conf(),
                                                 "default_initializer_conf");
}
int32_t JobDesc::PieceNumOfPrintLoss() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().piece_num_of_print_loss();
}
int32_t JobDesc::PieceNumOfPrintAccuracy() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().piece_num_of_print_accuracy();
}
int64_t JobDesc::BatchSize() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().batch_size();
}
int64_t JobDesc::NumOfPiecesInBatch() const {
  if (IsPredict()) { return 1; }
  CHECK_EQ(BatchSize() % PieceSize(), 0);
  return BatchSize() / PieceSize();
}
float JobDesc::primary_lr() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().primary_lr();
}
float JobDesc::secondary_lr() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().secondary_lr();
}
float JobDesc::weight_l1() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().weight_l1();
}
float JobDesc::bias_l1() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().bias_l1();
}
float JobDesc::weight_l2() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().weight_l2();
}
float JobDesc::bias_l2() const {
  CHECK(IsTrain());
  return job_conf_.other().train_conf().bias_l2();
}

int32_t JobDesc::DataPartNum() const { return job_conf_.other().data_part_num(); }

const FileSystemConf& JobDesc::data_fs_conf() const { return job_conf_.other().data_fs_conf(); }
const FileSystemConf& JobDesc::snapshot_fs_conf() const {
  return job_conf_.other().snapshot_fs_conf();
}

JobDesc::JobDesc(const std::string& job_conf_filepath) {
  if (TryParseProtoFromTextFile(job_conf_filepath, &job_conf_) == false) {
    JobConf2 job_conf;
    ParseProtoFromTextFile(job_conf_filepath, &job_conf);
    ParseProtoFromTextFile(job_conf.net(), job_conf_.mutable_net());
    ParseProtoFromTextFile(job_conf.resource(), job_conf_.mutable_resource());
    ParseProtoFromTextFile(job_conf.placement(), job_conf_.mutable_placement());
    ParseProtoFromTextFile(job_conf.other(), job_conf_.mutable_other());
  }
  SanityCheck();
  Init();
}

JobDesc::JobDesc(const JobConf1& job_conf) : job_conf_(job_conf) { Init(); }

void JobDesc::Init() {
  SplitDecodeOps();
  AddRecordLoadOps();
#ifndef WITH_RDMA
  CHECK_EQ(job_conf_.other().use_rdma(), false) << "Please compile ONEFLOW with RDMA";
#endif
#ifndef WITH_CUDA
  CHECK_EQ(job_conf_.other().enable_nccl(), false) << "Please compile ONEFLOW with NCCL";
#endif  // WITH_CUDA
  int64_t piece_exp = job_conf_.other().exp_run_conf().piece_num_of_experiment_phase();
  if (job_conf_.other().has_train_conf()) {
    TrainConf* train_conf = job_conf_.mutable_other()->mutable_train_conf();
    if (train_conf->piece_num_of_print_loss() == -1) {
      train_conf->set_piece_num_of_print_loss(NumOfPiecesInBatch());
    }
    if (train_conf->piece_num_of_print_accuracy() == -1) {
      train_conf->set_piece_num_of_print_accuracy(NumOfPiecesInBatch());
    }
    if (piece_exp == -1) { piece_exp = 19 * NumOfPiecesInBatch(); }
    piece_exp = std::max(piece_exp, NumOfPiecesInBatch());
    piece_exp = std::max(piece_exp, train_conf->piece_num_of_print_loss());
    piece_exp = std::min(piece_exp, job_conf_.other().total_batch_num() * NumOfPiecesInBatch());
  } else {
    if (piece_exp == -1) { piece_exp = 19; }
  }
  LOG(INFO) << "Set piece_num_of_experiment_phase " << piece_exp;
  job_conf_.mutable_other()->mutable_exp_run_conf()->set_piece_num_of_experiment_phase(piece_exp);
#ifndef WITH_CUDA
  CHECK_EQ(job_conf_.resource().gpu_device_num(), 0);
#endif
}

void JobDesc::SanityCheck() {
  int64_t machine_num = job_conf_.resource().machine_size();
  FOR_RANGE(int64_t, i, 0, machine_num) { CHECK_EQ(job_conf_.resource().machine(i).id(), i); }
}

int64_t JobDesc::GetMachineId(const std::string& addr) const {
  int64_t machine_id = -1;
  auto resource_conf = job_conf_.resource();
  int64_t machine_num = resource_conf.machine_size();
  FOR_RANGE(int64_t, i, 0, machine_num) {
    if (addr == resource_conf.machine(i).addr()) {
      machine_id = i;
      break;
    }
  }
  CHECK_GE(machine_id, 0);
  CHECK_LT(machine_id, machine_num);
  return machine_id;
}

void JobDesc::SplitDecodeOps() {
  std::vector<OperatorConf> gen_op_confs;
  for (OperatorConf& op_conf : *(job_conf_.mutable_net()->mutable_op())) {
    if (op_conf.has_decode_ofrecord_conf() == false) { continue; }
    if (op_conf.decode_ofrecord_conf().blob_size() == 1) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf.decode_ofrecord_conf();
    PbRpf<BlobConf>* blobs = op_conf.mutable_decode_ofrecord_conf()->mutable_blob();
    Erase<PbRpf<BlobConf>>(
        *blobs,
        [&](const BlobConf& blob_conf) -> bool { return blob_conf.max_sequence_size() > 1; },
        [&](const BlobConf& blob_conf) {
          gen_op_confs.emplace_back(op_conf);
          DecodeOFRecordOpConf* gen_decode_conf =
              gen_op_confs.back().mutable_decode_ofrecord_conf();
          *gen_decode_conf = decode_conf;
          gen_decode_conf->clear_blob();
          *gen_decode_conf->add_blob() = blob_conf;
        });
  }
  // TODO: erase decode op which has no blob any more
  for (OperatorConf& gen_op_conf : gen_op_confs) {
    *(job_conf_.mutable_net()->add_op()) = gen_op_conf;
  }
}

void JobDesc::AddRecordLoadOps() {
  HashMap<std::pair<std::string, std::string>, std::vector<OperatorConf*>> data_info2decode_ops;
  HashMap<std::pair<std::string, std::string>, int32_t> data_info2suffix_length;
  size_t op_num = job_conf_.net().op_size();
  FOR_RANGE(size_t, idx, 0, op_num) {
    OperatorConf* op_conf = job_conf_.mutable_net()->mutable_op()->Mutable(idx);
    if (op_conf->has_decode_ofrecord_conf() == false) { continue; }
    const DecodeOFRecordOpConf& decode_conf = op_conf->decode_ofrecord_conf();
    if (decode_conf.blob_size() == 0) { continue; }
    std::pair<std::string, std::string> data_info = {decode_conf.data_dir(),
                                                     decode_conf.part_name_prefix()};
    data_info2decode_ops[data_info].emplace_back(op_conf);
    int32_t part_name_suffix_length = decode_conf.part_name_suffix_length();
    if (data_info2suffix_length.find(data_info) != data_info2suffix_length.end()) {
      CHECK_EQ(data_info2suffix_length[data_info], part_name_suffix_length);
    } else {
      data_info2suffix_length[data_info] = part_name_suffix_length;
    }
  }

  HashMap<std::string, const ParallelConf*> name2parallel_conf;
  for (const PlacementGroup& p_group : job_conf_.placement().placement_group()) {
    for (const std::string& op_name : p_group.op_set().op_name()) {
      CHECK(name2parallel_conf.emplace(op_name, &p_group.parallel_conf()).second);
    }
  }

  for (const auto& pair : data_info2decode_ops) {
    std::vector<const ParallelConf*> parallel_confs;
    for (const OperatorConf* op_conf : pair.second) {
      auto op_parallel_conf_it = name2parallel_conf.find(op_conf->name());
      CHECK(op_parallel_conf_it != name2parallel_conf.end());
      auto iter = std::find_if(
          parallel_confs.begin(), parallel_confs.end(), [&](const ParallelConf* parallel_conf) {
            PbMd message_diff;
            return message_diff.Equivalent(*parallel_conf, *(op_parallel_conf_it->second));
          });
      if (iter == parallel_confs.end()) {
        parallel_confs.emplace_back(op_parallel_conf_it->second);
      }
    }
    LOG_IF(WARNING, parallel_confs.size() > 1)
        << "Operators sharing the same data information belong to different placement groups";
    for (const ParallelConf* parallel_conf : parallel_confs) {
      std::string record_load_op_name = "loader" + NewUniqueId();
      std::string record_load_out_name = "out";
      std::string record_load_lbi_name = record_load_op_name + "/" + record_load_out_name;
      OperatorConf* op = job_conf_.mutable_net()->add_op();
      RecordLoadOpConf* record_load_op = op->mutable_record_load_conf();
      op->set_name(record_load_op_name);
      record_load_op->set_out(record_load_out_name);
      record_load_op->set_data_dir(pair.first.first);
      record_load_op->set_part_name_prefix(pair.first.second);
      record_load_op->set_part_name_suffix_length(data_info2suffix_length.at(pair.first));
      PlacementGroup* p_group = job_conf_.mutable_placement()->add_placement_group();
      *(p_group->mutable_op_set()->add_op_name()) = record_load_op_name;
      *(p_group->mutable_parallel_conf()) = *parallel_conf;
      for (OperatorConf* op : pair.second) {
        std::string op_name = op->name();
        auto op_parallel_conf_it = name2parallel_conf.find(op_name);
        CHECK(op_parallel_conf_it != name2parallel_conf.end());
        PbMd message_diff;
        if (!message_diff.Equivalent(*parallel_conf, *(op_parallel_conf_it->second))) { continue; }
        op->mutable_decode_ofrecord_conf()->set_in(record_load_lbi_name);
      }
    }
  }
}

void JobDesc::FixAndOptimizeDLNet() {
  FixTickOpIfExists();
  AddIdentityOpForChainMergeOptimization();
  if (IsTrain()) { AddIdentityOpForAllReduceOverlapingUntrainble(); }
}

void JobDesc::AddIdentityOpForChainMergeOptimization() {
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job_conf_.placement());
  HashMap<std::pair<std::string, ParallelDesc>, HashSet<OperatorConf*>> grouped;
  GroupOpConfByProducerOpNameAndConsumerParallelDesc(&grouped, job_conf_.mutable_net(),
                                                     ParallelConf4OpName);
  for (auto& pair : grouped) {
    if (pair.second.size() == 1) { continue; }
    const auto& producer_op_name = pair.first.first;
    ParallelDesc producer_pr(*ParallelConf4OpName(producer_op_name));
    if (producer_pr.parallel_num() == pair.first.second.parallel_num()) { continue; }
    HashSet<std::string> input_lbi_blob_names;
    CollectInputLbiBlobNamesByProducerOpName(&input_lbi_blob_names, pair.second,
                                             pair.first.second.device_type(), producer_op_name);
    ParallelDesc consumer_op_pr(*ParallelConf4OpName((*pair.second.cbegin())->name()));
    std::string id_op_name = AddIdentityOp(producer_op_name, input_lbi_blob_names,
                                           *ParallelConf4OpName((*pair.second.cbegin())->name()));
    // reconnect to identity op
    for (auto* op_conf : pair.second) {
      PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
      std::shared_ptr<Operator> op = ConstructOp(*op_conf, consumer_op_pr.device_type());
      for (const auto& ibn : op->input_bns()) {
        const LogicalBlobId& lbi = op->BnInOp2Lbi(ibn);
        if (lbi.op_name() != producer_op_name) { continue; }
        std::string identity_out_lbn = id_op_name + "/" + lbi.blob_name();
        SetValInPbMessage<std::string>(op_type_conf, ibn, identity_out_lbn);
      }
    }
  }
}

void JobDesc::AddIdentityOpForAllReduceOverlapingUntrainble() {
  HashMap<std::string, OperatorConf*> op_name2op_conf;
  FOR_RANGE(int, idx, 0, job_conf_.net().op_size()) {
    OperatorConf* op_conf = job_conf_.mutable_net()->mutable_op(idx);
    CHECK(op_name2op_conf.emplace(op_conf->name(), op_conf).second);
  }
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job_conf_.placement());
  OpGraph(*this).TopoForEachNode([&](OpNode* op_node) {
    if (op_node->HasBackward()) { return; }
    HashMap<bool, std::vector<OpEdge*>> has_bw2out_op_edges;
    for (OpEdge* edge : op_node->out_edges()) {
      has_bw2out_op_edges[edge->dst_node()->HasBackward()].push_back(edge);
    }
    if (has_bw2out_op_edges.size() <= 1) { return; }
    // only handle op_nodes that:
    // a) have no backward node;
    // b) have trainable and untrainble consumers;

    // group out_edge by trainable consumers' ParallelDesc
    HashMap<ParallelDesc, std::vector<OpEdge*>> consumer_op_pr2edges;
    for (OpEdge* edge : has_bw2out_op_edges.at(true)) {
      ParallelDesc pr(*ParallelConf4OpName(edge->dst_node()->op().op_name()));
      consumer_op_pr2edges[pr].push_back(edge);
    }
    for (const auto& pair : consumer_op_pr2edges) {
      // add identity op
      HashSet<std::string> blob_names;
      for (OpEdge* edge : pair.second) {
        for (const LogicalBlobId& lbi : edge->lbis()) { blob_names.insert(lbi.blob_name()); }
      }
      std::string identity_op_name =
          AddIdentityOp(op_node->op().op_name(), blob_names,
                        *ParallelConf4OpName(pair.second.at(0)->dst_node()->op().op_name()));
      // reconnect to identity op
      for (OpEdge* edge : pair.second) {
        OperatorConf* op_conf = op_name2op_conf.at(edge->dst_node()->op().op_name());
        PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
        for (const LogicalBlobId& lbi : edge->lbis()) {
          std::string lbn_check = lbi.op_name() + "/" + lbi.blob_name();
          std::string identity_out_lbn = identity_op_name + "/" + lbi.blob_name();
          for (const std::string& ibn : edge->lbi2ibns().at(lbi)) {
            CHECK_EQ(GetValFromPbMessage<std::string>(*op_type_conf, ibn), lbn_check);
            SetValInPbMessage<std::string>(op_type_conf, ibn, identity_out_lbn);
          }
        }
      }
    }
  });
}

std::string JobDesc::AddIdentityOp(const std::string& input_op_name,
                                   const HashSet<std::string>& input_lbi_blob_names,
                                   const ParallelConf& parallel_conf) {
  // add tuple identity op
  OperatorConf* tuple_identity_op = job_conf_.mutable_net()->add_op();
  tuple_identity_op->set_name("clone_identity_" + NewUniqueId());
  IdentityOpConf* tuple_identity_op_conf = tuple_identity_op->mutable_identity_conf();
  for (const std::string& blob_name : input_lbi_blob_names) {
    tuple_identity_op_conf->add_in(input_op_name + "/" + blob_name);
    tuple_identity_op_conf->add_out(blob_name);
  }
  // add placement of tuple identity op
  PlacementGroup* p_group = job_conf_.mutable_placement()->add_placement_group();
  *(p_group->mutable_op_set()->add_op_name()) = tuple_identity_op->name();
  *(p_group->mutable_parallel_conf()) = parallel_conf;
  return tuple_identity_op->name();
}

void JobDesc::FixTickOpIfExists() {
  auto MutParallelConf4OpName = MakeGetterMutParallelConf4OpName(job_conf_.mutable_placement());
  OperatorConf* tick_op_conf = nullptr;
  FOR_RANGE(int, idx, 0, job_conf_.mutable_net()->op_size()) {
    OperatorConf* op_conf = job_conf_.mutable_net()->mutable_op(idx);
    if (op_conf->has_tick_conf()) {
      CHECK(tick_op_conf == nullptr);
      tick_op_conf = op_conf;
    }
  }
  if (tick_op_conf == nullptr) { return; }
  std::map<OperatorConf::OpTypeCase, std::vector<OperatorConf*>> op_type_case2source_op_confs;
  FOR_RANGE(int, idx, 0, job_conf_.mutable_net()->op_size()) {
    OperatorConf* op_conf = job_conf_.mutable_net()->mutable_op(idx);
    if (op_conf == tick_op_conf) { continue; }
    DeviceType device_type = ParallelDesc(*MutParallelConf4OpName(op_conf->name())).device_type();
    if (ConstructOp(*op_conf, device_type)->input_bns().size() == 0) {
      op_type_case2source_op_confs[op_conf->op_type_case()].push_back(op_conf);
    }
  }
  if (op_type_case2source_op_confs.find(OperatorConf::kRecordLoadConf)
      != op_type_case2source_op_confs.end()) {
    CHECK_EQ(op_type_case2source_op_confs.size(), 1);
  }
  // set input of tick op
  OperatorConf* source_op_conf = op_type_case2source_op_confs.cbegin()->second.at(0);
  ParallelConf* source_parallel_conf = MutParallelConf4OpName(source_op_conf->name());
  DeviceType device_type = ParallelDesc(*source_parallel_conf).device_type();
  std::shared_ptr<Operator> source_op = ConstructOp(*source_op_conf, device_type);
  CHECK_GE(source_op->output_bns().size(), 1);
  LogicalBlobId src_first_output_lbi = source_op->BnInOp2Lbi(source_op->output_bns().Get(0));
  std::string source_op_output_lbn = GenLogicalBlobName(src_first_output_lbi);
  CHECK_EQ(tick_op_conf->tick_conf().has_in(), false);
  tick_op_conf->mutable_tick_conf()->set_in(source_op_output_lbn);
  // fix tick op placement
  *MutParallelConf4OpName(tick_op_conf->name()) = *source_parallel_conf;
  // add log_counter op connecting to tick op, making tick op always consumed
  OperatorConf* tick_log_counter = job_conf_.mutable_net()->add_op();
  tick_log_counter->set_name("tick_log_counter_" + NewUniqueId());
  LogCounterOpConf* tick_log_counter_conf = tick_log_counter->mutable_log_counter_conf();
  tick_log_counter_conf->set_in(tick_op_conf->name() + "/" + tick_op_conf->tick_conf().out());
  tick_log_counter_conf->set_interval(MaxVal<int32_t>::value);
  // add placement of tick_log_counter op
  PlacementGroup* p_group = job_conf_.mutable_placement()->add_placement_group();
  *(p_group->mutable_op_set()->add_op_name()) = tick_log_counter->name();
  *(p_group->mutable_parallel_conf()) = *source_parallel_conf;
}

}  // namespace oneflow
