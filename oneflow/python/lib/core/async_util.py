"""
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
"""
import threading


def Await(counter, func):
    assert counter > 0
    cond_var = threading.Condition()
    counter_box = [counter]
    result_list = []

    def Yield(result=None):
        result_list.append(result)
        cond_var.acquire()
        assert counter_box[0] > 0
        counter_box[0] -= 1
        cond_var.notify()
        cond_var.release()

    func(Yield)
    cond_var.acquire()
    while counter_box[0] > 0:
        cond_var.wait()
    cond_var.release()
    return result_list
