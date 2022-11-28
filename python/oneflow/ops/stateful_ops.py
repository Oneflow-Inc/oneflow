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
import oneflow
import oneflow._oneflow_internal
import oneflow.framework.id_util as id_util


class StatefulOp(object):
    def __init__(self, op_type_name, op_name=None):
        if op_name is None:
            op_name = id_util.UniqueStr(op_type_name)
        self._builder = oneflow._oneflow_internal.one.OpBuilder(op_type_name, op_name)
        self._op = None
        self._op_type_name = op_type_name

    @property
    def op(self):
        """access the builtin op

        Returns:
            the builtin op
        """
        if self._op is None:
            self._op = self._builder.build()
        return self._op

    def Input(self, input_name, num=1):
        """Set input blob of op

        Args:
            input_name (str): input name of blob
            num (int, optional) : Defaults to 1.

        Returns:
            self
        """
        assert isinstance(num, int) and num >= 1
        self._builder.input(input_name, num)
        return self

    def Output(self, output_name, num=1):
        """Set output blob of op

        Args:
            output_name (str): name of output blob
            num (int, optional):  Defaults to 1.

        Returns:
            self
        """
        assert isinstance(num, int) and num >= 1
        self._builder.output(output_name, num)
        return self

    def Build(self):
        """Explicitly complete the construction of the builtin op

        Returns:
            the completed builtin op
        """
        if self._op is None:
            self._op = self._builder.build()
        return self._op
