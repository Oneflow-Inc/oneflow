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
import inspect
import os
import unittest
from collections import OrderedDict

import oneflow as flow
import oneflow.unittest

from oneflow.test_utils.test_util import GenArgList


def _run_functional_doctest(
    test_case,
    globs=None,
    verbose=None,
    optionflags=0,
    raise_on_error=True,
    module=flow,
):
    import doctest

    parser = doctest.DocTestParser()
    if raise_on_error:
        runner = doctest.DebugRunner(verbose=verbose, optionflags=optionflags)
    else:
        runner = doctest.DocTestRunner(verbose=verbose, optionflags=optionflags)
    r = inspect.getmembers(module)
    for (name, fun) in r:
        if fun.__doc__ is not None:
            test = parser.get_doctest(fun.__doc__, {}, __name__, __file__, 0)
            try:
                runner.run(test)
            except doctest.DocTestFailure as e:
                print(f"\nGot error result in the docstring of {name}")
                print(f"got output: {e.got}")
                raise e
            except doctest.UnexpectedException as e:
                print(f"\nGot UnexpectedException in the docstring of {name}")
                raise e.exc_info[1]

    if not raise_on_error:
        test_case.assertEqual(
            runner.failures,
            0,
            f"{runner.summarize()}, please turn on raise_on_error to see more details",
        )


@flow.unittest.skip_unless_1n1d()
@unittest.skipIf(os.getenv("ONEFLOW_TEST_CPU_ONLY"), "only test cpu cases")
class TestFunctionalDocstrModule(flow.unittest.TestCase):
    def test_functional_docstr(test_case):
        arg_dict = OrderedDict()
        arg_dict["module"] = [flow, flow.Tensor, flow.sbp, flow.env, flow.nn.functional]
        for arg in GenArgList(arg_dict):
            _run_functional_doctest(
                test_case, raise_on_error=True, verbose=True, module=arg[0]
            )


if __name__ == "__main__":
    flow.set_printoptions(linewidth=80)
    unittest.main()
