import unittest
import numpy as np
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as oft

def _test_user_op_attr_auto_type(input, attr1, attr2):
    return flow.user_op_builder('test_user_op_attr_auto_type').Op('test_user_op_attr_auto_type').Input('in', [input]).Output('out').Attr('int1', attr1).Attr('int2', attr2).Build().InferAndTryRun().RemoteBlobList()

@flow.unittest.skip_unless_1n1d()
class TestUserOpAttrAutoType(flow.unittest.TestCase):

    def test_user_op_attr_auto_type(test_case):
        flow.clear_default_session()
        function_config = flow.FunctionConfig()
        function_config.default_data_type(flow.float)

        @flow.global_function(function_config=function_config)
        def _test_user_op_attr_auto_type_job(input: oft.Numpy.Placeholder((1,), dtype=flow.float)):
            attr1 = 1
            attr2 = 2
            return _test_user_op_attr_auto_type(input, attr1, attr2)
        input = [1]
        _test_user_op_attr_auto_type_job(np.array(input, dtype=np.float32))
if __name__ == '__main__':
    unittest.main()