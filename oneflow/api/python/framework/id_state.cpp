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
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "oneflow/api/python/of_api_registry.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/job/id_state.h"

namespace py = pybind11;

ONEFLOW_API_PYBIND11_MODULE("", m) {
  using namespace oneflow;

  py::bind_map<HashMap<StreamId, TaskId::task_index_t>>(m, "HashMapStreamIdTaskIndex");
  py::bind_map<HashMap<DeviceId, StreamId::stream_index_t>>(m, "HashMapDeviceIdStreamIndex");

  py::class_<IdState>(m, "IdState")
      .def(py::init())
      .def_readwrite("regst_desc_id_state", &IdState::regst_desc_id_state_)
      .def_readwrite("mem_block_id_state", &IdState::mem_block_id_state_)
      .def_readwrite("chunk_id_state", &IdState::chunk_id_state_)
      .def_readwrite("task_index_state", &IdState::task_index_state_)
      .def_readwrite("stream_index_state", &IdState::stream_index_state_);

  m.def("load_id_state",
        [](const IdState& id_state) { Singleton<IdStateMgr>::Get()->LoadIdState(id_state); });
  m.def("save_id_state", []() -> IdState { return Singleton<IdStateMgr>::Get()->SaveIdState(); });
}
