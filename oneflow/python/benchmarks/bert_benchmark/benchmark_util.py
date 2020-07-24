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
import time

import numpy as np


class StopWatch:
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()
        self.last_split = self.start_time

    def set_start(self, val):
        self.start_time = val
        self.last_split = self.start_time

    def split(self):
        now = time.time()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return self.stop_time - self.start_time


class BERTSpeedometer:
    def __init__(self):
        self.watch = StopWatch()
        self.throughoutput_list = []

    def speedometer_cb(
        self,
        step,
        start_time,
        total_batch_size,
        skip_iter_num,
        iter_num,
        loss_print_every_n_iter,
    ):
        def callback(train_loss):
            assert skip_iter_num >= 0
            if skip_iter_num == 0 and step == 0:
                self.watch.set_start(start_time)
                print("Start trainning without any skipping iteration.")

            if step < skip_iter_num:
                if step == 0:
                    print(
                        "Skipping {} iterations for benchmark purpose.".format(
                            skip_iter_num
                        )
                    )
                if (step + 1) == skip_iter_num:
                    self.watch.start()
                    print("Start trainning.")
            else:
                train_step = step - skip_iter_num

                if (train_step + 1) % loss_print_every_n_iter == 0:
                    total_loss = train_loss[0].mean()
                    mlm_loss = train_loss[1].mean()
                    nsp_loss = train_loss[2].mean()

                    avg_elapse_time_per_iter = (
                        self.watch.split() / loss_print_every_n_iter
                    )
                    sentences_per_sec = total_batch_size / avg_elapse_time_per_iter
                    print(
                        "iter {}, total_loss: {:.3f}, mlm_loss: {:.3f}, nsp_loss: {:.3f}, speed: {:.3f}(sec/batch), {:.3f}(sentences/sec)".format(
                            train_step,
                            total_loss,
                            mlm_loss,
                            nsp_loss,
                            avg_elapse_time_per_iter,
                            sentences_per_sec,
                        )
                    )
                    self.throughoutput_list.append(sentences_per_sec)

                if (train_step + 1) == iter_num:
                    self.watch.stop()
                    totoal_duration = self.watch.duration()
                    avg_sentences_per_sec = (
                        total_batch_size * iter_num / totoal_duration
                    )

                    print("-".ljust(66, "-"))
                    print(
                        "average speed: {:.3f}(sentences/sec), new_cal_method: {:.3f}(sentences/sec)".format(
                            avg_sentences_per_sec, np.mean(self.throughoutput_list)
                        )
                    )
                    print("-".ljust(66, "-"))

        return callback
