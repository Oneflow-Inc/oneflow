#include "oneflow/core/job_completer/debug_blob_dump.h"
#include <google/protobuf/text_format.h>

namespace oneflow {

namespace {

ParallelConf GenCpuOneToOneParallelConf(const ParallelDesc& desc) {
  ParallelConf conf;
  conf.set_policy(ParallelPolicy::kDataParallel);
  for (const int64_t machine_id : desc.sorted_machine_ids()) {
    for (const int64_t dev_id : desc.sorted_dev_phy_ids(machine_id)) {
      conf.add_device_name(std::to_string(machine_id) + ":cpu:" + std::to_string(dev_id));
    }
  }
  return conf;
}

}  // namespace

void DebugBlobDump(const OpGraph& op_graph, Job* job) {
  JobBuilder job_builder(job);
  BlobDumpMeta meta;
  const std::string& base_dir =
      Global<JobDesc>::Get()->job_conf().other().debug_blob_dump_conf().base_dir();
  SnapshotFS()->RecursivelyCreateDirIfNotExist(base_dir);
  std::set<OperatorConf::OpTypeCase> black_list = {OperatorConf::kReduceSplitConf};
  op_graph.ForEachNode([&](const OpNode* node) {
    if (black_list.count(node->op().op_conf().op_type_case()) != 0) { return; }
    for (const std::string& bn : node->op().output_bns()) {
      const LogicalBlobId& lbi = node->op().BnInOp2Lbi(bn);
      OperatorConf identity_op_conf{};
      identity_op_conf.set_name("System-Debug-Identity-" + NewUniqueId());
      IdentityOpConf* identity_conf = identity_op_conf.mutable_identity_conf();
      identity_conf->set_in(GenLogicalBlobName(lbi));
      identity_conf->set_out("out");
      job_builder.AddOps(node->parallel_desc().parallel_conf(), {identity_op_conf});
      OperatorConf dump_op_conf{};
      dump_op_conf.set_name("System-Debug-BlobDump-" + NewUniqueId());
      BlobDumpOpConf* conf = dump_op_conf.mutable_blob_dump_conf();
      conf->set_in(identity_op_conf.name() + "/" + identity_conf->out());
      const std::string sub_dir_name = GenLogicalBlobName(lbi);
      std::string dir = JoinPath(base_dir, sub_dir_name);
      SnapshotFS()->RecursivelyCreateDirIfNotExist(dir);
      conf->set_dir(dir);
      job_builder.AddOps(GenCpuOneToOneParallelConf(node->parallel_desc()), {dump_op_conf});
      BlobDumpMetaElem* blob_meta = meta.mutable_blob()->Add();
      *blob_meta->mutable_lbi() = lbi;
      op_graph.GetLogicalBlobDesc(lbi).ToProto(blob_meta->mutable_blob_desc());
      *blob_meta->mutable_parallel_conf() = node->parallel_desc().parallel_conf();
      *blob_meta->mutable_sbp_parallel() = node->SbpParallel4Lbi(lbi);
      *blob_meta->mutable_dir() = sub_dir_name;
    }
    PersistentOutStream out_stream(SnapshotFS(), JoinPath(base_dir, "meta"));
    std::string output;
    google::protobuf::TextFormat::PrintToString(meta, &output);
    out_stream << output;
    out_stream.Flush();
  });
}

}  // namespace oneflow
