/*
 * simple_graph_execution_state.h
 * Copyright (C) 2017 xiaoshu <2012wxs@gmail.com>
 *
 * Distributed under terms of the MIT license.
 */

#ifndef SIMPLE_GRAPH_EXECUTION_STATE_H
#define SIMPLE_GRAPH_EXECUTION_STATE_H

namespace oneflow {
struct SimpleGraphExecutionStateOptions {};

struct SimpleClientGraph {
  explicit SimpleClientGraph() {}
};

class SimpleGraphExecutionState {
 public:
  SimpleGraphExecutionState();
  virtual ~SimpleGraphExecutionState();
  void Create();
};

}


#endif /* !SIMPLE_GRAPH_EXECUTION_STATE_H */
