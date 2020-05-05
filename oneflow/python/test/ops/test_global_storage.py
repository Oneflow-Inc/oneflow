global_storage = {}

def get(name):
    return global_storage.get(name)


def set(name):
    global global_storage

    def _set(x):
        global_storage[name] = x

    return _set


def getToNdarray(name):
  return get(name).ndarray()
