/**
 * Copyright 2017 Xinqi Li
 */
#include <fstream>
#include <sstream>
#include <string>
#include "oneflow/core/schedule/data_structure/node.h"

namespace oneflow {
namespace schedule {

void TestGraph(const std::string& input_name) {
  auto graph = unique_ptr_new<GraphNode>("root");
  auto root = graph.get();
  auto pool = graph->mut_pool();

  auto get_id = [](uint64_t id) { return id * 1001; };

  std::string arg0, arg1, arg2;
  std::stringstream ss;
  std::ifstream input_file(input_name);
  while (std::cin >> arg0 >> arg1 >> arg2) {
    // while(input_file >> arg0 >> arg1 >> arg2) {
    if (arg0 == "ln") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      Node* node = pool->mut_node_mgr().CreateWithId(get_id(id), name);
      if (node) { pool->mut_children_arc_mgr().CreateIfNotFound(root, node); }
    } else if (arg0 == "gl") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      Node* node = pool->mut_node_mgr().Find(get_id(id));
      if (node) { pool->mut_loss_arc_mgr().CreateIfNotFound(root, node); }
    } else if (arg0 == "ae") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      Node* from = pool->mut_node_mgr().Find(get_id(from_id));
      Node* to = pool->mut_node_mgr().Find(get_id(to_id));
      if (from && to) { pool->mut_arc_mgr().CreateIfNotFound(from, to); }
    } else if (arg0 == "dt") {
      uint64_t time;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> name >> time;
      auto dev = pool->mut_device_mgr().CreateIfNotFound(name, 1);
      dev->mut_time() = time;
    } else if (arg0 == "dn") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      auto dev = pool->mut_device_mgr().CreateIfNotFound(name, 1);
      auto node = pool->mut_node_mgr().CreateIfNotFound(get_id(id));
      pool->mut_device_arc_mgr().CreateIfNotFound(node, dev);
    } else if (arg0 == "pr") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      Node* from = pool->mut_node_mgr().Find(get_id(from_id));
      Node* to = pool->mut_regst_desc_mgr().CreateIfNotFound(get_id(to_id));
      if (from && to) {
        pool->mut_produced_regst_desc_mgr().CreateIfNotFound(from, to);
      }
    } else if (arg0 == "sr") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      Node* from = pool->mut_node_mgr().Find(get_id(from_id));
      Node* to = pool->mut_regst_desc_mgr().CreateIfNotFound(get_id(to_id));
      if (from && to) {
        pool->mut_subscribed_regst_desc_mgr().CreateIfNotFound(from, to);
      }
    }
  }

  //  auto nr_device = root->DeviceCount();

  Session sess(root);

  UnlimitedMode<PositiveStrategy> m0(&sess);
  //  UnlimitedMode<NegativeStrategy> m0(&sess);
  m0.Run();

  sess.RegstDescCount();
  auto logger = sess.GetLoggerThenReset();
  UnlimitedMode<NegativeStrategy> m1(&sess);
  m1.Run();

  sess.RegstDescCount();

  sess.logger()->MergeTimeGapToLossInPlace(&*logger);
  sess.logger()->UpdateDuration(&sess, &m1);

  auto regst_desc2count = sess.RegstDescCount();

  Session session(root);

  std::cout << "------limited------" << std::endl;
  LimitedMode<NegativeStrategy> m3(&session, *regst_desc2count);
  m3.Run();
  auto log = session.GetLoggerThenReset();

  LimitedMode<PositiveStrategy> m4(&session, *regst_desc2count);
  m4.Run();

  session.logger()->MergeTimeGapToLossInPlace(&*log);
  session.logger()->UpdateDuration(&session, &m4);
  session.RegstDescCount();

  exit(0);

  std::cout << "------lesser------" << std::endl;
  int target = 0;
  for (int i = 0; i < 10; i++) {
    std::cout << "---------------" << std::endl;
    Session::PipeCount limited;
    int count = 0;
    bool declined = false;
    for (const auto& p : *regst_desc2count) {
      limited[p.first].count = p.second.count;
      if (count == target && limited[p.first].count > 1) {
        limited[p.first].count -= 1;
        declined = true;
      }
      //      limited[p.first].count = 4;
      std::cout << p.first << "\t" << limited[p.first].count << std::endl;
      count++;
    }
    target++;
    if (declined) {
      LimitedMode<NegativeStrategy> m3(&session, limited);
      m3.Run();
      auto log = session.GetLoggerThenReset();

      LimitedMode<PositiveStrategy> m4(&session, limited);
      m4.Run();

      session.logger()->MergeTimeGapToLossInPlace(&*log);
      session.logger()->UpdateDuration(&session, &m4);
      session.RegstDescCount();
    }
  }
}

}  // namespace schedule
}  // namespace oneflow

int main(int argc, char* argv[]) {
  std::string input_name;
  if (argc > 1) { input_name = argv[1]; }
  oneflow::schedule::TestGraph(input_name);
  return 0;
}
