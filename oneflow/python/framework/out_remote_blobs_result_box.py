from __future__ import absolute_import

import threading
import oneflow.python.framework.inter_user_job_util as inter_user_job_util
import oneflow.python.framework.remote_blob as remote_blob_util

class OutRemoteBlobsResultBox(object):
    def __init__(self):
        self.cond_var_ = threading.Condition()
        self.out_remote_blob_pullers_ = []
        self.inited_ = False

    # user api
    def get(self):
        assert self.inited_
        return self._WaitAndGetResultNdarray(self.out_remote_blob_pullers_)

    def AddResult(self, out_remote_blobs):
        assert self.inited_ == False
        pullers = self._MakeRemoteBlobPullers(out_remote_blobs)
        self.out_remote_blob_pullers_.append(pullers)
        for puller in self._FlatRemoteBlobPullers(pullers): puller.AsyncPull()
        return self

    def SetResult(self, out_remote_blobs):
        assert self.inited_ == False
        assert isinstance(self.out_remote_blob_pullers_, list) 
        assert len(self.out_remote_blob_pullers_) == 0 
        pullers = self._MakeRemoteBlobPullers(out_remote_blobs)
        self.out_remote_blob_pullers_ = pullers
        for puller in self._FlatRemoteBlobPullers(pullers): puller.AsyncPull()
        return self
    
    def Inited(self):
        assert self.inited_ == False
        self.inited_ = True
        return self

    def _WaitAndGetResultNdarray(self, pullers):
        assert self.inited_
        if isinstance(pullers, _RemoteBlobPuller):
            return pullers.WaitAndGetResultNdarray()
        if isinstance(pullers, list) or isinstance(pullers, tuple):
            ret = [self._WaitAndGetResultNdarray(x) for x in pullers]
            if isinstance(pullers, tuple): ret = tuple(ret)
            return ret
        if isinstance(pullers, dict):
            return {
                k : self._WaitAndGetResultNdarray(v) for k, v in pullers.items()
            }
        raise NotImplementedError

    def _FlatRemoteBlobPullers(self, pullers):
        if isinstance(pullers, _RemoteBlobPuller):
            yield pullers
        elif isinstance(pullers, list) or isinstance(pullers, tuple):
            for elem in pullers:
                for x in self._FlatRemoteBlobPullers(elem): yield x
        elif isinstance(pullers, dict):
            for _, v in pullers.items():
                for x in self._FlatRemoteBlobPullers(v): yield x
        else:
            raise NotImplementedError

    def _MakeRemoteBlobPullers(self, out_remote_blobs):
        if isinstance(out_remote_blobs, remote_blob_util.RemoteBlob):
            return _RemoteBlobPuller(out_remote_blobs, self.cond_var_)
        if isinstance(out_remote_blobs, list) or isinstance(out_remote_blobs, tuple):
            ret = [self._MakeRemoteBlobPullers(x) for x in out_remote_blobs]
            if isinstance(out_remote_blobs, tuple): ret = tuple(ret)
            return ret
        if isinstance(out_remote_blobs, dict):
            return {
                k : self._MakeRemoteBlobPullers(v) for k, v in out_remote_blobs.items()
            }
        raise NotImplementedError

class _RemoteBlobPuller(object):
    def __init__(self, remote_blob, cond_var):
        self.op_name_ = remote_blob.op_name
        self.result_ndarray_ = None
        self.finished_ = False
        self.cond_var_= cond_var

    def WaitAndGetResultNdarray(self):
        self._Wait()
        return self.result_ndarray_

    def AsyncPull(self):
        def PullCallback(of_blob):
            self.result_ndarray_ = of_blob.CopyToNdarray()
            self.cond_var_.acquire()
            self.finished_ = True
            self.cond_var_.notify()
            self.cond_var_.release()
            
        inter_user_job_util.AsyncPull(self.op_name_, PullCallback)

    def _Wait(self):
        self.cond_var_.acquire()
        while self.finished_ == False:
            self.cond_var_.wait()
        self.cond_var_.release()


