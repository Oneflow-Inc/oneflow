import traceback
import oneflow._oneflow_internal


def write_int8_calibration(path):
    try:
        oneflow._oneflow_internal.WriteInt8Calibration(path)
    except oneflow._oneflow_internal.exception.CompileOptionWrongException:
        traceback.print_exc()


def cache_int8_calibration():
    try:
        oneflow._oneflow_internal.CacheInt8Calibration()
    except oneflow._oneflow_internal.exception.CompileOptionWrongException:
        traceback.print_exc()
