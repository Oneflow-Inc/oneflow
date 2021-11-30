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

import oneflow._oneflow_internal
from doctest import DocTestParser, DebugRunner, DocTestRunner


def _test_docstr(docstr, verbose=True, optionflags=0, raise_on_error=True):
    parser = DocTestParser()
    if raise_on_error:
        runner = DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        runner = DocTestRunner(verbose=verbose, optionflags=optionflags)
    test = parser.get_doctest(docstr, {}, __name__, __file__, 0)
    runner.run(test)


def add_docstr(fun, docstr: str):
    return oneflow._oneflow_internal.add_doc(fun, docstr)


def reset_docstr(o, docstr):
    if type(o) == type:
        assert hasattr(o, "__doc__"), str(o) + " does not have a docstring!"
        setattr(o, "__doc__", docstr)
        return o
    else:
        return oneflow._oneflow_internal.reset_doc(o, docstr)
