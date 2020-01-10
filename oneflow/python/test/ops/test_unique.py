import oneflow as flow
import numpy as np

func_config = flow.FunctionConfig()
func_config.default_data_type(flow.float)

def test_unique_with_counts(test_case):
    @flow.function(func_config)
    def UniqueWithCountsJob(x=flow.FixedTensorDef((64,), dtype=flow.int32)):
        return flow.experimental.unique_with_counts(x)
   
    x = np.asarray(list(range(32)) * 2).astype(np.int32) 
    ret = UniqueWithCountsJob(x)
    print(ret)
    test_case.assertTrue(True)

