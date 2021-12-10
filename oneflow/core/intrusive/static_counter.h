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
#ifndef ONEFLOW_CORE_INTRUSIVE_STATIC_COUNTER_H_
#define ONEFLOW_CORE_INTRUSIVE_STATIC_COUNTER_H_

namespace oneflow {

#define STATIC_COUNTER(counter_name) _STATIC_COUNTER_NAME(counter_name)<_AUTO_INCREMENT()>::value

#define DEFINE_STATIC_COUNTER(counter_name) _DEFINE_STATIC_COUNTER(_AUTO_INCREMENT(), counter_name)

#define INCREASE_STATIC_COUNTER(counter_name) \
  _INCREASE_STATIC_COUNTER(_AUTO_INCREMENT(), counter_name)

// details

#define _STATIC_COUNTER_NAME(counter_name) StaticCounter_##counter_name
#define _AUTO_INCREMENT() __COUNTER__

#define _DEFINE_STATIC_COUNTER(auto_counter, counter_name)                               \
  template<int tpl_counter, typename Enabled = void>                                     \
  struct _STATIC_COUNTER_NAME(counter_name) {                                            \
    static const int value = _STATIC_COUNTER_NAME(counter_name)<tpl_counter - 1>::value; \
  };                                                                                     \
  template<typename Enabled>                                                             \
  struct _STATIC_COUNTER_NAME(counter_name)<auto_counter, Enabled> {                     \
    static const int value = 0;                                                          \
  };

#define _INCREASE_STATIC_COUNTER(auto_counter, counter_name)                                  \
  template<typename Enabled>                                                                  \
  struct _STATIC_COUNTER_NAME(counter_name)<auto_counter, Enabled> {                          \
    static const int value = _STATIC_COUNTER_NAME(counter_name)<auto_counter - 1>::value + 1; \
  };
}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_STATIC_COUNTER_H_
