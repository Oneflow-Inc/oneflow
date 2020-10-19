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
import os
import oneflow.core.summary.projector_pb2 as projector_pb2
from oneflow.python.oneflow_export import oneflow_export
import time

import oneflow as flow


@oneflow_export("summary.Projector")
class Projector(object):
    r"""The class of Projector

    This class can create an 'embedding_projector' or 'exception_projector'
    """

    def __init__(self, logdir=None):
        r"""Create a Projector objector

        Args:
            logdir: The log dir

        Raises:
            Exception: If 'logdir' is None or illegal
        """
        if logdir is None:
            raise Exception("logdir should not be None!")
        logdir += "/projector"
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.logdir_ = logdir
        self.embedding_filename_ = None
        self.exception_filename_ = None

    def create_embedding_projector(self):
        if (self.embedding_filename_ is not None) and (
            os.path.exists(self.embedding_filename_)
        ):
            raise OSError("You must create only one embedding projector!")
        self.embedding_filename_ = (
            self.logdir_ + "/projector." + str(int(time.time())) + ".log"
        )

    def create_exception_projector(self):
        if (self.exception_filename_ is not None) and (
            os.path.exists(self.exception_filename_)
        ):
            raise OSError("You must create only one embedding projector!")
        self.exception_filename_ = (
            self.logdir_ + "/projector.gradit." + str(int(time.time())) + ".log"
        )

    @property
    def logdir(self):
        return self.logdir_

    @property
    def exception_filename(self):
        return self.exception_filename_

    @property
    def embedding_filename(self):
        return self.embedding_filename_

    def write_projector(self, filename=None, projector=None):
        with open(filename, "wb") as f:
            f.write(projector.SerializeToString())
            f.flush()

    def set_tensor(self, tensor: projector_pb2.Tensor, value):
        for d in value.shape:
            td = tensor.shape.dim.add()
            td.size = d
        tensor.dtype = str(value.dtype)
        tensor.content = value.tobytes()

    def set_projector(self, pro, tag, step, value, label=None):
        pro.tag = str(tag)
        pro.step = step
        pro.WALL_TIME = time.time()
        self.set_tensor(pro.value, value)
        if label is not None:
            self.set_tensor(pro.label, label)

    def set_sample(self, sample, name, x, sample_type):
        if name is not None:
            sample.name = name
        if sample_type == "image" or sample_type == "IMAGE":
            sample.type = projector_pb2.Sample.SampleType.IMAGE
        elif sample_type == "audio" or sample_type == "AUDIO":
            sample.type = projector_pb2.Sample.SampleType.AUDIO
        elif sample_type == "text" or sample_type == "TEXT":
            sample.type = projector_pb2.Sample.SampleType.TEXT
        else:
            raise NotImplementedError
        if x is not None:
            self.set_tensor(sample.X, x)

    def embedding_projector(
        self,
        value=None,
        label=None,
        tag=None,
        step=None,
        sample_name=None,
        sample_type=None,
        x=None,
    ):
        if tag is None:
            tag = "embedding_projector"
        summary_projector = projector_pb2.SummaryProjector()
        summary_projector.metadata.type = projector_pb2.MetaData.ProjectorType.EMBEDDING
        projector = summary_projector.projector.add()
        self.set_projector(pro=projector, tag=tag, step=step, value=value, label=label)
        if (sample_name is not None) and (sample_type is not None):
            self.set_sample(
                sample=summary_projector.sample,
                name=sample_name,
                x=x,
                sample_type=sample_type,
            )
        self.write_projector(self.embedding_filename_, summary_projector)

    def exception_projector(
        self,
        value=None,
        tag=None,
        step=None,
        sample_name=None,
        sample_type=None,
        x=None,
    ):
        if tag is None:
            tag = "exception_projector"
        summary_projector = projector_pb2.SummaryProjector()
        summary_projector.metadata.type = projector_pb2.MetaData.ProjectorType.EXCEPTION
        projector = summary_projector.projector.add()
        self.set_projector(pro=projector, tag=tag, step=step, value=value)
        if (sample_name is not None) and (sample_type is not None):
            self.set_sample(
                sample=summary_projector.sample,
                name=sample_name,
                x=x,
                sample_type=sample_type,
            )
        self.write_projector(self.exception_filename_, summary_projector)
