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
#ifndef ONEFLOW_CORE_COMMON_SWITCH_FUNC_H_
#define ONEFLOW_CORE_COMMON_SWITCH_FUNC_H_

#include "oneflow/core/common/preprocessor.h"
#include <tuple>
#include <utility>

template<typename... Args>
auto SwitchCase(Args&&... args) {
  return std::make_tuple(std::forward<Args>(args)...);
}

#define DEFINE_STATIC_SWITCH_FUNC(return_type, func_name, make_switch_entry, ctrv_seq, ...) \
  DEFINE_STATIC_SWITCH_FUNC_FROM_TUPLE(return_type, func_name, make_switch_entry,           \
                                       OF_PP_CAT((ctrv_seq, ##__VA_ARGS__), ))

#define DEFINE_STATIC_SWITCH_FUNC_FROM_TUPLE(return_type, func_name, make_switch_entry,        \
                                             ctrv_seq_tuple)                                   \
  template<typename... Args>                                                                   \
  static return_type Switch##func_name(                                                        \
      const OF_PP_I_CTRV_SEQ_TUPLE2STD_TUPLE_TYPE(ctrv_seq_tuple) & switch_tuple,              \
      Args && ... args) {                                                                      \
    static const std::map<OF_PP_I_CTRV_SEQ_TUPLE2STD_TUPLE_TYPE(ctrv_seq_tuple),               \
                          std::function<return_type(Args&&...)>>                               \
        case_handlers{OF_PP_I_MAKE_ALL_SWITCH_ENTRIES_FROM_TUPLE(make_switch_entry, func_name, \
                                                                 Args, ctrv_seq_tuple)};       \
    return case_handlers.at(switch_tuple)(std::forward<Args>(args)...);                        \
  }

// CTRV: Compile-time Token and Runtime Value pair,
// CTRV example: (float, DataType::kFloat)
// TYPED_CTRV_SEQ example: (DataType, ((float, DataType::kFloat)))

#define MAKE_DATA_TYPE_CTRV_SEQ(data_type_seq) MAKE_TYPED_CTRV_SEQ(DataType, data_type_seq)
#define MAKE_DEVICE_TYPE_CTRV_SEQ(device_type_seq) \
  MAKE_TYPED_CTRV_SEQ(DeviceType,                  \
                      OF_PP_FOR_EACH_TUPLE(OF_PP_I_MAKE_REPLICATE_TUPLE_SEQ, device_type_seq))
#define MAKE_NDIM_CTRV_SEQ(ndim_seq) \
  MAKE_TYPED_CTRV_SEQ(int32_t, OF_PP_FOR_EACH_TUPLE(OF_PP_I_MAKE_REPLICATE_TUPLE_SEQ, ndim_seq))

#define MAKE_STRINGIZED_DATA_TYPE_CTRV(data_type_pair) \
  (OF_PP_PAIR_FIRST(data_type_pair), OF_PP_STRINGIZE(OF_PP_PAIR_FIRST(data_type_pair)))
#define MAKE_STRINGIZED_DATA_TYPE_CTRV_SEQ(data_type_seq) \
  (std::string, OF_PP_SEQ_MAP(MAKE_STRINGIZED_DATA_TYPE_CTRV, data_type_seq))

#define MAKE_TYPED_CTRV_SEQ(runtime_value_type, ctrv_pair_seq) (runtime_value_type, ctrv_pair_seq)

//  internal preprocessor macros

#define OF_PP_I_MAKE_SWITCH_ENTRY_MAP_PAIR(switch_case, func_args_type, func) \
  {switch_case,                                                               \
   [](func_args_type&&... args) { return func(std::forward<func_args_type>(args)...); }},

#define OF_PP_I_MAKE_REPLICATE_TUPLE_SEQ(x) OF_PP_MAKE_TUPLE_SEQ(x, x)

#define OF_PP_I_MAKE_SWITCH_FUNC_ENTRY_1(make_template_func, func_name, func_args_type, \
                                         switch_case_pair0)                             \
  OF_PP_I_MAKE_SWITCH_ENTRY_MAP_PAIR(                                                   \
      SwitchCase(OF_PP_PAIR_SECOND(switch_case_pair0)), func_args_type,                 \
      make_template_func(func_name, OF_PP_PAIR_FIRST(switch_case_pair0)))

#define OF_PP_I_MAKE_SWITCH_FUNC_ENTRY_2(make_template_func, func_name, func_args_type,       \
                                         switch_case_pair0, switch_case_pair1)                \
  OF_PP_I_MAKE_SWITCH_ENTRY_MAP_PAIR(                                                         \
      SwitchCase(OF_PP_PAIR_SECOND(switch_case_pair0), OF_PP_PAIR_SECOND(switch_case_pair1)), \
      func_args_type,                                                                         \
      make_template_func(func_name, OF_PP_PAIR_FIRST(switch_case_pair0),                      \
                         OF_PP_PAIR_FIRST(switch_case_pair1)))

#define OF_PP_I_MAKE_SWITCH_FUNC_ENTRY_3(make_template_func, func_name, func_args_type,           \
                                         switch_case_pair0, switch_case_pair1, switch_case_pair2) \
  OF_PP_I_MAKE_SWITCH_ENTRY_MAP_PAIR(                                                             \
      SwitchCase(OF_PP_PAIR_SECOND(switch_case_pair0), OF_PP_PAIR_SECOND(switch_case_pair1),      \
                 OF_PP_PAIR_SECOND(switch_case_pair2)),                                           \
      func_args_type,                                                                             \
      make_template_func(func_name, OF_PP_PAIR_FIRST(switch_case_pair0),                          \
                         OF_PP_PAIR_FIRST(switch_case_pair1),                                     \
                         OF_PP_PAIR_FIRST(switch_case_pair2)))

#define OF_PP_I_MAKE_SWITCH_FUNC_ENTRY_4(make_template_func, func_name, func_args_type,            \
                                         switch_case_pair0, switch_case_pair1, switch_case_pair2,  \
                                         switch_case_pair3)                                        \
  OF_PP_I_MAKE_SWITCH_ENTRY_MAP_PAIR(                                                              \
      SwitchCase(OF_PP_PAIR_SECOND(switch_case_pair0), OF_PP_PAIR_SECOND(switch_case_pair1),       \
                 OF_PP_PAIR_SECOND(switch_case_pair2), OF_PP_PAIR_SECOND(switch_case_pair3)),      \
      func_args_type,                                                                              \
      make_template_func(func_name, OF_PP_PAIR_FIRST(switch_case_pair0),                           \
                         OF_PP_PAIR_FIRST(switch_case_pair1), OF_PP_PAIR_FIRST(switch_case_pair2), \
                         OF_PP_PAIR_FIRST(switch_case_pair3)))

#define OF_PP_I_MAKE_ALL_SWITCH_ENTRIES_FROM_TUPLE(make_switch_entry, func_name, args_type, t) \
  OF_PP_FORCE(OF_PP_CAT(OF_PP_I_MAKE_ALL_SWITCH_ENTRIES_FROM_TUPLE_, OF_PP_TUPLE_SIZE(t))(     \
      make_switch_entry, func_name, args_type, t))

#define OF_PP_I_MAKE_ALL_SWITCH_ENTRIES_FROM_TUPLE_1(make_switch_entry, func_name, args_type, \
                                                     ctrv_seq_tuple)                          \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_I_MAKE_SWITCH_FUNC_ENTRY_1, (make_switch_entry),     \
                                   (func_name), (args_type),                                  \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(0, ctrv_seq_tuple)))
#define OF_PP_I_MAKE_ALL_SWITCH_ENTRIES_FROM_TUPLE_2(make_switch_entry, func_name, args_type, \
                                                     ctrv_seq_tuple)                          \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_I_MAKE_SWITCH_FUNC_ENTRY_2, (make_switch_entry),     \
                                   (func_name), (args_type),                                  \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(0, ctrv_seq_tuple)),    \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(1, ctrv_seq_tuple)))
#define OF_PP_I_MAKE_ALL_SWITCH_ENTRIES_FROM_TUPLE_3(make_switch_entry, func_name, args_type, \
                                                     ctrv_seq_tuple)                          \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_I_MAKE_SWITCH_FUNC_ENTRY_3, (make_switch_entry),     \
                                   (func_name), (args_type),                                  \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(0, ctrv_seq_tuple)),    \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(1, ctrv_seq_tuple)),    \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(2, ctrv_seq_tuple)))
#define OF_PP_I_MAKE_ALL_SWITCH_ENTRIES_FROM_TUPLE_4(make_switch_entry, func_name, args_type, \
                                                     ctrv_seq_tuple)                          \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(OF_PP_I_MAKE_SWITCH_FUNC_ENTRY_4, (make_switch_entry),     \
                                   (func_name), (args_type),                                  \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(0, ctrv_seq_tuple)),    \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(1, ctrv_seq_tuple)),    \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(2, ctrv_seq_tuple)),    \
                                   OF_PP_PAIR_SECOND(OF_PP_TUPLE_ELEM(3, ctrv_seq_tuple)))

#define OF_PP_I_CTRV_SEQ_TUPLE2STD_TUPLE_TYPE(t) \
  OF_PP_FORCE(OF_PP_CAT(OF_PP_I_CTRV_SEQ_TUPLE2STD_TUPLE_TYPE_, OF_PP_TUPLE_SIZE(t))(t))

#define OF_PP_I_CTRV_SEQ_TUPLE2STD_TUPLE_TYPE_1(t) \
  std::tuple<OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(0, t))>
#define OF_PP_I_CTRV_SEQ_TUPLE2STD_TUPLE_TYPE_2(t) \
  std::tuple<OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(0, t)), OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(1, t))>
#define OF_PP_I_CTRV_SEQ_TUPLE2STD_TUPLE_TYPE_3(t)                                               \
  std::tuple<OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(0, t)), OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(1, t)), \
             OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(2, t))>
#define OF_PP_I_CTRV_SEQ_TUPLE2STD_TUPLE_TYPE_4(t)                                               \
  std::tuple<OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(0, t)), OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(1, t)), \
             OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(2, t)), OF_PP_PAIR_FIRST(OF_PP_TUPLE_ELEM(3, t))>

#endif  // ONEFLOW_CORE_COMMON_SWITCH_FUNC_H_
