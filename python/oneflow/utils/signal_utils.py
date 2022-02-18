import signal

def prepend_signal(sig, f):

    old = None
    if callable(signal.getsignal(sig)):
        # NOTE(jianhao): Maybe we should take care of signal.SIG_DFL, 
	# signal.SIG_IGN and signal.default_int_handler like in
	# http://www.pybloggers.com/2016/02/how-to-always-execute-exit-functions-in-python/
        old = signal.getsignal(sig)

    def helper(*args, **kwargs):
        f(*args, **kwargs)
        if old is not None:
            old(*args, **kwargs)

    signal.signal(sig, helper)
