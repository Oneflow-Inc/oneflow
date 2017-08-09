/**
 * Copyright 2017 Xinqi Li
 */
#include <fstream>
#include <sstream>
#include <string>
#include "oneflow/core/schedule/factory.h"
#include "oneflow/core/schedule/node.h"
#include "oneflow/core/schedule/simulator.h"

namespace oneflow {
namespace schedule {

void SimulatorPolicyDemo() {
  auto ph = PH("naive");
  auto graph = ph->DemoGraph();
  ph->PrintGraph(*graph, "");
  auto session = ph->MakeSession(*graph);
  auto schedule_result = ph->Schedule(*session);
  ph->Retiming(*session, schedule_result.get());
  ph->AllocateFromSchedule(*session, schedule_result.get());
  bool success = ph->ValidateAllocation(*session, *schedule_result);
  std::cout << "allocation is " << (success ? "" : "NOT ") << "optimal"
            << std::endl;
}

/*
void TestGraph(const std::string& input_name) {
  auto graph_ptr = unique_ptr_new<SGraph>("graph");
  auto graph = graph_ptr.get();

  auto get_id = [](uint64_t id) { return id * 1001; };

  std::string arg0, arg1, arg2;
  std::stringstream ss;
  std::ifstream input_file(input_name);
  while (std::cin >> arg0 >> arg1 >> arg2) {
    if (arg0 == "ln") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      Node* node = graph->mut_node_mgr().CreateWithId(get_id(id), name);
      if (node) { graph->mut_children_arc_mgr().CreateIfNotFound(graph, node); }
    } else if (arg0 == "gl") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      Node* node = graph->mut_node_mgr().Find(get_id(id));
      if (node) { graph->mut_loss_arc_mgr().CreateIfNotFound(graph, node); }
    } else if (arg0 == "ae") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      Node* from = graph->mut_node_mgr().Find(get_id(from_id));
      Node* to = graph->mut_node_mgr().Find(get_id(to_id));
      if (from && to) { graph->mut_arc_mgr().CreateIfNotFound(from, to); }
    } else if (arg0 == "dt") {
      uint64_t time;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> name >> time;
      auto dev = graph->mut_device_mgr().CreateIfNotFound(name, 1);
      dev->mut_time() = time;
    } else if (arg0 == "dn") {
      uint64_t id;
      std::string name;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> id >> name;
      auto dev = graph->mut_device_mgr().CreateIfNotFound(name, 1);
      auto node = graph->mut_node_mgr().CreateIfNotFound(get_id(id));
      graph->mut_device_arc_mgr().CreateIfNotFound(node, dev);
    } else if (arg0 == "pr") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      Node* from = graph->mut_node_mgr().Find(get_id(from_id));
      Node* to = graph->mut_regst_desc_mgr().CreateIfNotFound(get_id(to_id));
      if (from && to) {
        graph->mut_produced_regst_desc_mgr().CreateIfNotFound(from, to);
      }
    } else if (arg0 == "sr") {
      uint64_t from_id;
      uint64_t to_id;
      ss.clear();
      ss << arg1 << "\t" << arg2;
      ss >> from_id >> to_id;

      Node* from = graph->mut_node_mgr().Find(get_id(from_id));
      Node* to = graph->mut_regst_desc_mgr().CreateIfNotFound(get_id(to_id));
      if (from && to) {
        graph->mut_subscribed_regst_desc_mgr().CreateIfNotFound(from, to);
      }
    }
  }

  //  auto nr_device = graph->DeviceCount();
  graph->Update();

  SimulatorSession sess(graph);

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

  SimulatorSession session(graph);

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
    SimulatorSession::PipeCount limited;
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
                auto get_regst_num = [&](uint64_t id) {
                        return limited[id];
                };
    target++;
    if (declined) {
      LimitedMode<NegativeStrategy> m3(&session, get_regst_num);
      m3.Run();
      auto log = session.GetLoggerThenReset();

      LimitedMode<PositiveStrategy> m4(&session, get_regst_num);
      m4.Run();

      session.logger()->MergeTimeGapToLossInPlace(&*log);
      session.logger()->UpdateDuration(&session, &m4);
      session.RegstDescCount();
    }
  }
}
*/

}  // namespace schedule
}  // namespace oneflow

int main(int argc, char* argv[]) {
  std::string input_name;
  if (argc > 1) { input_name = argv[1]; }
  //  oneflow::schedule::TestGraph(input_name);
  oneflow::schedule::SimulatorPolicyDemo();
  return 0;
}
