_DICT = {}


def get_metrics_dict():
    return _DICT


def put_metrics(metrics):
    _DICT.update(metrics)
    return _DICT


# in mirror mode, it is recommended to clear the dict when one rank is built
def clear_metrics_dict():
    _DICT = {}
