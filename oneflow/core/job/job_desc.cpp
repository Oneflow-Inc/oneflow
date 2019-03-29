#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/persistence/hadoop/hadoop_file_system.h"
#include "oneflow/core/graph/graph.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

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

namespace {

std::function<OperatorConf*(const std::string&)> MakeMutableOperatorConf4OpName(
    JobConf1* job_conf) {
  auto op_name2op_conf = std::make_shared<HashMap<std::string, OperatorConf*>>();
  FOR_RANGE(int, idx, 0, job_conf->net().op_size()) {
    OperatorConf* op_conf = job_conf->mutable_net()->mutable_op(idx);
    CHECK(op_name2op_conf->emplace(op_conf->name(), op_conf).second);
  }
  return [op_name2op_conf](const std::string& op_name) { return op_name2op_conf->at(op_name); };
}

void AddIdentityOp(const std::string& prefix, JobConf1* job_conf,
                   const HashSet<LogicalBlobId>& input_lbis,
                   HashMap<LogicalBlobId, LogicalBlobId>* old_lbi2new_lbi,
                   const ParallelConf& parallel_conf) {
  // add tuple identity op
  OperatorConf* tuple_identity_op = job_conf->mutable_net()->add_op();
  tuple_identity_op->set_name(prefix + NewUniqueId());
  TupleIdentityOpConf* tuple_identity_op_conf = tuple_identity_op->mutable_tuple_identity_conf();
  int32_t idx = 0;
  for (const LogicalBlobId& lbi : input_lbis) {
    std::string blob_name = std::string("out_") + std::to_string(idx++);
    {
      LogicalBlobId output_lbi;
      output_lbi.set_op_name(tuple_identity_op->name());
      output_lbi.set_blob_name(blob_name);
      CHECK(old_lbi2new_lbi->emplace(lbi, output_lbi).second);
    }
    tuple_identity_op_conf->add_in(lbi.op_name() + "/" + lbi.blob_name());
    tuple_identity_op_conf->add_out(blob_name);
  }
  // add placement of tuple identity op
  PlacementGroup* p_group = job_conf->mutable_placement()->add_placement_group();
  *(p_group->mutable_op_set()->add_op_name()) = tuple_identity_op->name();
  *(p_group->mutable_parallel_conf()) = parallel_conf;
}

void SetPbMessageField(PbMessage* pb_msg, const std::string& field, const std::string& old_val,
                       const std::string& new_val) {
  const PbFd* fd = pb_msg->GetDescriptor()->FindFieldByName(field);
  if (fd) {
    CHECK_EQ(GetValFromPbMessage<std::string>(*pb_msg, field), old_val);
    SetValInPbMessage<std::string>(pb_msg, field, new_val);
  } else {
    const std::pair<std::string, int32_t> prefix_idx = GenUnRepeatedBn(field);
    CHECK_EQ(GetPbRpfFromPbMessage<std::string>(*pb_msg, prefix_idx.first).Get(prefix_idx.second),
             old_val);
    PbRpf<std::string>* rpf = MutPbRpfFromPbMessage<std::string>(pb_msg, prefix_idx.first);
    *rpf->Mutable(prefix_idx.second) = new_val;
  }
}

void AddIdentityOpAndReconnect(
    const std::string& identity_op_name_prefix, JobConf1* job_conf,
    const std::vector<OpEdge*>& op_edges,
    const std::function<OperatorConf*(const std::string&)>& MutOperatorConf4OpName,
    const ParallelConf& parallel_conf) {
  // add identity op
  HashSet<LogicalBlobId> lbis;
  for (OpEdge* edge : op_edges) { lbis.insert(edge->lbis().begin(), edge->lbis().end()); }
  HashMap<LogicalBlobId, LogicalBlobId> old_lbi2new_lbi;
  AddIdentityOp(identity_op_name_prefix, job_conf, lbis, &old_lbi2new_lbi, parallel_conf);
  // reconnect to identity op
  for (OpEdge* edge : op_edges) {
    OperatorConf* op_conf = MutOperatorConf4OpName(edge->dst_node()->op().op_name());
    PbMessage* op_type_conf = MutableMessageInPbMessage(op_conf, op_conf->op_type_case());
    for (const LogicalBlobId& lbi : edge->lbis()) {
      std::string lbn_check = GenLogicalBlobName(lbi);
      std::string identity_out_lbn = GenLogicalBlobName(old_lbi2new_lbi.at(lbi));
      for (const std::string& ibn : edge->lbi2ibns().at(lbi)) {
        SetPbMessageField(op_type_conf, ibn, lbn_check, identity_out_lbn);
      }
    }
  }
}

}  // namespace

int64_t JobDesc::all_reduce_group_min_byte() const {
  int64_t ret = job_conf_.other().all_reduce_group_min_mbyte() * 1024 * 1024;
  CHECK_GT(ret, 0);
  return ret;
}

float JobDesc::all_reduce_group_size_warmup() const {
  float ret = job_conf_.other().all_reduce_group_size_warmup();
  CHECK_GT(ret, 1);
  return ret;
}

int64_t JobDesc::all_reduce_group_num() const {
  int64_t ret = job_conf_.other().all_reduce_group_num();
  CHECK_GT(ret, 0);
  return ret;
}

float JobDesc::all_reduce_lazy_ratio() const {
  float ratio = job_conf_.other().all_reduce_lazy_ratio();
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
  CHECK_EQ(BatchSize() % RecordPieceSize(), 0);
  return BatchSize() / RecordPieceSize();
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
  if (this->TotalMachineNum() > 1) {
    CHECK_EQ(this->use_rdma(), false) << "Please compile ONEFLOW with RDMA";
  }
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
  HashMap<std::pair<std::string, std::string>, const RandomShuffleConf*> data_info2shuffle_conf;
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
    const RandomShuffleConf* shuffle_conf =
        decode_conf.has_random_shuffle_conf() ? &decode_conf.random_shuffle_conf() : nullptr;
    if (data_info2shuffle_conf.find(data_info) != data_info2shuffle_conf.end()) {
      if (shuffle_conf == nullptr) {
        CHECK(data_info2shuffle_conf.at(data_info) == nullptr);
      } else {
        CHECK(data_info2shuffle_conf.at(data_info) != nullptr);
        CHECK_EQ(data_info2shuffle_conf.at(data_info)->buffer_size(), shuffle_conf->buffer_size());
      }
    } else {
      CHECK(data_info2shuffle_conf.emplace(data_info, shuffle_conf).second);
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
      if (data_info2shuffle_conf.at(pair.first) != nullptr) {
        *record_load_op->mutable_random_shuffle_conf() = *data_info2shuffle_conf.at(pair.first);
      }
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
  ConvertPseudoChainToChain();
  if (IsTrain()) { AddIdentityOpForAllReduceOverlapingUntrainble(); }
}

void JobDesc::ConvertPseudoChainToChain() {
  auto GetSourceNodesAndEdges = [&](const HashSet<OpNode*>& chain_nodes,
                                    HashSet<OpNode*>* source_nodes,
                                    std::vector<OpEdge*>* source_edges) {
    for (OpNode* node : chain_nodes) {
      for (OpEdge* edge : node->in_edges()) {
        if (chain_nodes.find(edge->src_node()) == chain_nodes.end()) {
          source_edges->push_back(edge);
          source_nodes->insert(node);
        }
      }
    }
  };
  auto MutOperatorConf4OpName = MakeMutableOperatorConf4OpName(&job_conf_);
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job_conf_.placement());
  OpGraph(this).ForEachPseudoChain([&](const HashSet<OpNode*>& chain_nodes) {
    HashSet<OpNode*> source_nodes;
    std::vector<OpEdge*> source_edges;
    GetSourceNodesAndEdges(chain_nodes, &source_nodes, &source_edges);
    if (source_edges.size() <= 1) { return; }
    if (source_nodes.size() <= 1) { return; }
    if (chain_nodes.size() - source_nodes.size() <= 2) { return; }
    const OpNode* first_node = *source_nodes.begin();
    if (first_node->parallel_desc().device_type() == DeviceType::kCPU) { return; }
    HashMap<bool, std::vector<OpEdge*>> has_diff2source_edges;
    for (OpEdge* edge : source_edges) { has_diff2source_edges[edge->has_diff()].push_back(edge); }
    for (const auto& pair : has_diff2source_edges) {
      HashSet<OpNode*> src_nodes;
      HashSet<OpNode*> dst_nodes;
      for (OpEdge* edge : pair.second) {
        src_nodes.emplace(edge->src_node());
        dst_nodes.emplace(edge->dst_node());
      }
      if (src_nodes.size() > 1 && dst_nodes.size() > 1) {
        AddIdentityOpAndReconnect("pseudo_chain_header_", &job_conf_, pair.second,
                                  MutOperatorConf4OpName,
                                  *ParallelConf4OpName(first_node->op().op_name()));
      }
    }
  });
}

void JobDesc::AddIdentityOpForAllReduceOverlapingUntrainble() {
  auto MutOperatorConf4OpName = MakeMutableOperatorConf4OpName(&job_conf_);
  auto ParallelConf4OpName = MakeGetterParallelConf4OpName(job_conf_.placement());
  OpGraph(this).TopoForEachNode([&](OpNode* op_node) {
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
      AddIdentityOpAndReconnect(
          "all_reduce_overlapping_untrainable_", &job_conf_, pair.second, MutOperatorConf4OpName,
          *ParallelConf4OpName(pair.second.at(0)->dst_node()->op().op_name()));
    }
  });
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
