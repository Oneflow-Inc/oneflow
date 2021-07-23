import oneflow.core.common.data_type_pb2 as data_type_conf_util
import oneflow.core.job.initializer_conf_pb2 as initializer_conf_util
import oneflow.core.operator.op_conf_pb2 as op_conf_util


def truncated_normal_initializer(
    stddev: float = 1.0,
) -> initializer_conf_util.InitializerConf:
    initializer = initializer_conf_util.InitializerConf()
    setattr(initializer.truncated_normal_conf, "std", float(stddev))
    return initializer
