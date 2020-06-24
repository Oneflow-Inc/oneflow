import threading


def Await(counter, func):
    assert counter > 0
    cond_var = threading.Condition()
    counter_box = [counter]
    result_list = []

    def Yield(result=None):
        result_list.append(result)
        cond_var.acquire()
        assert counter_box[0] > 0
        counter_box[0] -= 1
        cond_var.notify()
        cond_var.release()

    func(Yield)
    cond_var.acquire()
    while counter_box[0] > 0:
        cond_var.wait()
    cond_var.release()
    return result_list
