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
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/job/id_state.h"

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;

  py::class_<IdState>(m, "IdState")
      .def(py::init<>())
      .def_readwrite("regst_desc_id_state", &IdState::regst_desc_id_state_)
      .def_readwrite("mem_block_id_state", &IdState::mem_block_id_state_)
      .def_readwrite("chunk_id_state", &IdState::chunk_id_state_)
      .def_readwrite("job_id_state", &IdState::job_id_state_)
      .def_readwrite("task_index_state", &IdState::task_index_state_)
      .def_readwrite("stream_index_state", &IdState::stream_index_state_)
      // support pickle
      .def(py::pickle(
          [](const IdState& id_state) {
            return py::make_tuple(id_state.regst_desc_id_state_, id_state.mem_block_id_state_,
                                  id_state.chunk_id_state_, id_state.job_id_state_,
                                  id_state.task_index_state_, id_state.stream_index_state_);
          },
          [](const py::tuple& t) {
            CHECK(t.size() == 6);
            IdState id_state;
            id_state.regst_desc_id_state_ = t[0].cast<int64_t>();
            id_state.mem_block_id_state_ = t[1].cast<int64_t>();
            id_state.chunk_id_state_ = t[2].cast<int64_t>();
            id_state.job_id_state_ = t[3].cast<int64_t>();
            id_state.task_index_state_ = t[4].cast<HashMap<int64_t, uint32_t>>();
            id_state.stream_index_state_ = t[5].cast<HashMap<int64_t, uint32_t>>();
            return id_state;
          }));

  m.def("set_id_state", [](const IdState& id_state) {
    Singleton<MultiClientSessionContext>::Get()->SetIdState(id_state);
  });
  m.def("get_id_state", []() { return Singleton<MultiClientSessionContext>::Get()->GetIdState(); });
}
