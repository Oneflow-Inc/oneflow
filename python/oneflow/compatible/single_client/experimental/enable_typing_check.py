from oneflow.compatible.single_client.python.framework import hob as hob
from oneflow.compatible.single_client.python.lib.core import enable_if as enable_if


def api_enable_typing_check(val: bool = True) -> None:
    """ enable typing check for global_function """
    return enable_if.unique([enable_typing_check])(val)


@enable_if.condition(hob.in_normal_mode & ~hob.any_global_function_defined)
def enable_typing_check(val):
    global typing_check_enabled
    typing_check_enabled = val


typing_check_enabled = False
