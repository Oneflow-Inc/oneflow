#include "oneflow/core/job_completer/add_lbi_diff_watcher.h"
#include "oneflow/core/job/lbi_diff_watcher_info.pb.h"
#include "oneflow/core/operator/operator.h"

namespace oneflow {

void AddLbiDiffWatcherOpConfs(Job* job) {
  JobBuilder job_builder(job);
  const auto& map = Global<LbiDiffWatcherInfo>::Get()->job_name2lbi_and_watcher_uuids();
  if (map.find(GlobalJobDesc().job_name()) == map.end()) { return; }
  const auto& tag2lbi_relations = job->helper().tag2lbi_relations();
  const auto& conf_iter = tag2lbi_relations.find(kProducedLbi2ConsumedDiffLbi);
  if (conf_iter == tag2lbi_relations.end()) { return; }
  HashMap<LogicalBlobId, LogicalBlobId> lbi2diff_lbi;
  for (const auto& pair : conf_iter->second.pair()) {
    CHECK(lbi2diff_lbi.emplace(pair.first(), pair.second()).second);
  }
  const auto& pair_list = map.at(GlobalJobDesc().job_name()).lbi_and_uuid_pair();
  for (const LbiAndDiffWatcherUuidPair& pair : pair_list) {
    if (lbi2diff_lbi.find(pair.lbi()) == lbi2diff_lbi.end()) { continue; }
    OperatorConf foreign_watcher_op;
    foreign_watcher_op.set_name("System-LbiDiffWatcher-ForeignWatcher-" + NewUniqueId());
    auto* foreign_watcher_conf = foreign_watcher_op.mutable_foreign_watch_conf();
    foreign_watcher_conf->set_in(GenLogicalBlobName(lbi2diff_lbi.at(pair.lbi())));
    foreign_watcher_conf->set_handler_uuid(pair.watcher_uuid());
    job_builder.AddOps(job_builder.ParallelConf4Lbi(pair.lbi()), {foreign_watcher_op});
  }
}

}  // namespace oneflow
