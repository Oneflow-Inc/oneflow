#include "oneflow/core/schedule/demo_sgraph.h"

namespace oneflow {
namespace schedule {

void DemoSGraph::InitFromDemo() {
  std::string input_str = "ln 1 fw1\n"
                          "ln 2 fw2\n"
                          "ln 3 fw3\n"
                          "ln 4 fw4\n"
                          "ln 5 loss\n"
                          "ln 6 bp2\n"
                          "ln 7 bp4\n"
                          "ln 8 bp3\n"
                          "ln 9 bp1\n"
                          "gl 5 loss\n"
                          "ae 1 2\n"
                          "ae 1 3\n"
                          "ae 3 4\n"
                          "ae 2 5\n"
                          "ae 4 5\n"
                          "ae 5 6\n"
                          "ae 5 7\n"
                          "ae 7 8\n"
                          "ae 6 9\n"
                          "ae 8 9\n"
                          "pr 1 1002\n"
                          "pr 1 1003\n"
                          "pr 3 3004\n"
                          "pr 2 2005\n"
                          "pr 4 4005\n"
                          "pr 5 5006\n"
                          "pr 5 5007\n"
                          "pr 7 7008\n"
                          "pr 6 6009\n"
                          "pr 8 8009\n"
                          "sr 2 1002\n"
                          "sr 3 1003\n"
                          "sr 4 3004\n"
                          "sr 5 2005\n"
                          "sr 5 4005\n"
                          "sr 6 5006\n"
                          "sr 6 2005\n"
                          "sr 7 5007\n"
                          "sr 7 4005\n"
                          "sr 8 7008\n"
                          "sr 8 3004\n"
                          "sr 9 6009\n"
                          "sr 9 8009\n"
                          "sr 9 1002\n"
                          "sr 9 1003\n"
                          "#dn 1 type\n"
                          "#dn 9 type\n"
                          "#dn 2 type\n"
                          "#dn 3 type\n"
                          "#dn 6 type\n"
                          "#dn 7 type\n"
                          "#dn 8 type\n"
                          "#dn 4 type\n"
                          "#dn 5 type\n"
                          "#dn 1 type-0\n"
                          "#dn 9 type-1\n"
                          "#dn 2 type-2\n"
                          "#dn 3 type-0\n"
                          "#dn 6 type-1\n"
                          "#dn 7 type-2\n"
                          "#dn 8 type-0\n"
                          "#dn 4 type-1\n"
                          "#dn 5 type-2\n"
                          "#dn 1 type-0\n"
                          "#dn 9 type-0\n"
                          "#dn 2 type-1\n"
                          "#dn 3 type-1\n"
                          "#dn 6 type-1\n"
                          "#dn 7 type-2\n"
                          "#dn 8 type-4\n"
                          "#dn 4 type-2\n"
                          "#dn 5 type-3\n"
                          "dn 1 type-01\n"
                          "dn 9 type-02\n"
                          "#dn 2 type-13\n"
                          "dn 2 type-47\n"
                          "dn 3 type-14\n"
                          "dn 6 type-15\n"
                          "dn 7 type-16\n"
                          "dn 8 type-47\n"
                          "dn 4 type-28\n"
                          "dn 5 type-39\n"
                          "dt type    1\n"
                          "dt type-01 1\n"
                          "dt type-02 1\n"
                          "dt type-13 1\n"
                          "dt type-14 1\n"
                          "dt type-15 1\n"
                          "dt type-16 1\n"
                          "dt type-47 2\n"
                          "dt type-28 1\n"
                          "dt type-39 1\n"
                          "dt type-0 1\n"
                          "dt type-1 10\n"
                          "dt type-4 1\n"
                          "dt type-2 1\n"
                          "dt type-3 1\n";
  auto get_id = [](uint64_t id) { return id * 1001; };

  std::string arg0, arg1, arg2;
  std::stringstream ss;
  std::stringstream input_stream(input_str);
  while (input_stream >> arg0 >> arg1 >> arg2) {
    if (arg0 == "ln") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      STask* node = mut_node_mgr<STask>()->CreateWithId(get_id(id), name);
      if (node) { mut_children_arc_mgr()->CreateIfNotFound(this, node); }
    } else if (arg0 == "gl") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      STask* node = node_mgr<STask>().Find(get_id(id));
      if (node) { mut_loss_arc_mgr()->CreateIfNotFound(this, node); }
    } else if (arg0 == "ae") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      STask* src_node = node_mgr<STask>().Find(get_id(from_id));
      STask* dst_node = node_mgr<STask>().Find(get_id(to_id));
      if (src_node && dst_node) {
        mut_arc_mgr()->CreateIfNotFound(src_node, dst_node);
      }
    } else if (arg0 == "dt") {
      uint64_t time;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> name >> time;
      SDevice* dev = mut_node_mgr<SDevice>()->CreateIfNotFound(name, name, 1);
      dev->set_time(time);
    } else if (arg0 == "dn") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      SDevice* dev = mut_node_mgr<SDevice>()->CreateIfNotFound(name, name, 1);
      STask* node = mut_node_mgr<STask>()->CreateIfNotFound(get_id(id));
      mut_device_arc_mgr()->CreateIfNotFound(node, dev);
    } else if (arg0 == "pr") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      STask* src_node = node_mgr<STask>().Find(get_id(from_id));
      SRegstDesc* dst_node =
          mut_node_mgr<SRegstDesc>()->CreateIfNotFound(get_id(to_id));
      if (src_node && dst_node) {
        mut_produced_regst_desc_mgr()->CreateIfNotFound(src_node, dst_node);
      }
    } else if (arg0 == "sr") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      STask* src_node = node_mgr<STask>().Find(get_id(from_id));
      SRegstDesc* dst_node =
          mut_node_mgr<SRegstDesc>()->CreateIfNotFound(get_id(to_id));
      if (src_node && dst_node) {
        mut_subscribed_regst_desc_mgr()->CreateIfNotFound(src_node, dst_node);
      }
    }
  }
}

}  // namespace schedule
}  // namespace oneflow
