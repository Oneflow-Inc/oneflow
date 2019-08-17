from __future__ import absolute_import

from oneflow.core.operator.op_conf_pb2 import OperatorConf;
from oneflow.core.job.dlnet_conf_pb2 import DLNetConf
from google.protobuf import text_format;
from oneflow.python.deprecated.blob import Blob
import oneflow.python.ops.op_util as op_util
import oneflow.python.framework.placement_context as placement_ctx
import oneflow.python.framework.compile_context as compile_ctx
import oneflow.core.operator.op_conf_pb2 as op_conf_pb2;
import oneflow.core.common.data_type_pb2 as data_type_pb;
import oneflow.core.common.data_type_pb2 as dt
import oneflow.python.lib.core.pb_util as pb_util
import oneflow.python.deprecated.util as util
import oneflow.python.deprecated.record_util as record_util
from contextlib import contextmanager
from oneflow.python.oneflow_export import oneflow_export
import numpy as np
import os

RepeatedScalarContainer = type(getattr(op_conf_pb2.AddOpConf(), 'in'));

# deprecated
@oneflow_export('deprecated.get_cur_job_dlnet_builder')
def get_cur_job_dlnet_builder():
    global _cur_job2dl_net_builder
    if id(compile_ctx.cur_job) not in _cur_job2dl_net_builder:
        _cur_job2dl_net_builder[id(compile_ctx.cur_job)] = DLNet(compile_ctx.cur_job.net)
    return _cur_job2dl_net_builder[id(compile_ctx.cur_job)]

_cur_job2dl_net_builder = {}

class DLNet(object):
    def __init__(self, dl_net_conf):
        self.dl_net_conf_ = dl_net_conf;
        self.variable_scope_ = [];

    def DecodeOFRecord(self, data_dir="", name=None, **kw):
        assert('blob' in kw);
        assert(len(kw['blob']) > 0);
        kw['data_dir'] = data_dir;
        name = self._GetVariableName(name, util.GetCurFuncName());
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw);
        outs = {}
        for blob in kw['blob']:
            outs[blob['name']] = Blob(self, "%s/%s"%(name, blob['name']));
        return outs;

    def DebugConstant(self, ndarray, **kw):
        test_blob = self.DefineTestBlobLike(ndarray, **kw)
        return self.DebugReplace(test_blob, ndarray);

    def DefineTestBlobLike(self, ndarray, **kw):
        return self.DefineTestBlob(ndarray.shape,
                data_type=_NdarrayDType2OFDataType(ndarray.dtype), **kw)

    def DefineTestBlob(self, blob_shape=None, name=None, **kw):
        conf = {'out': 'out', 'shape': {'dim': blob_shape}, 'data_type':data_type_pb.kFloat}
        conf = util.ExtendDict(conf, **kw)
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **conf)
        return Blob(self, "%s/out" % name)

    def DecodeRandom(self, name=None, **kw):
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName());
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, "%s/out" % name)

    def DebugTee(self, in_blob, print_dir):
        return self.Debug(in_blob, in_blob_dump_dir=print_dir);

    def DebugTeeDiff(self, in_blob, print_dir):
        return self.Debug(in_blob, out_diff_blob_dump_dir=print_dir);

    def DebugReplace(self, in_blob, np_ndarray, filepath=None):
        const_feature_file = record_util.Ndarray2OFFeatureFile(np_ndarray, filepath)
        return self.Debug(in_blob, const_out_feature_load_filepath=const_feature_file)

    def DebugReplaceDiff(self, in_blob, np_ndarray, filepath=None):
        const_feature_file = record_util.Ndarray2OFFeatureFile(np_ndarray, filepath)
        return self.Debug(in_blob, const_in_diff_feature_load_filepath=const_feature_file)

    def Proposal(self, class_prob, bbox_pred, name=None, **kw):
        assert('anchor_generator_conf' in kw);
        assert('bbox_reg_weights' in kw);
        assert('min_size' in kw);
        assert('pre_nms_top_n' in kw);
        assert('post_nms_top_n' in kw);
        assert('nms_threshold' in kw);
        #assert(type(class_prob) in [str, Blob]);
        #assert(type(bbox_pred) in [str, Blob]);
        kw['class_prob'] = str(class_prob);
        kw['bbox_pred'] = str(bbox_pred);
        kw['rois'] = 'rois';
        kw['roi_probs'] = 'roi_probs'
        name = self._GetVariableName(name, util.GetCurFuncName());
        self.CreateOperator('Proposal', name, [], [], **kw);
        return Blob(self, '{}/rois'.format(name)), Blob(self, '{}/roi_probs'.format(name));

    def ProposalTarget(self, rois_blob, gt_boxes, gt_labels, name=None, **kw):
        # input blobs
        assert('rois' not in kw)
        assert('gt_boxes' not in kw)
        assert('gt_labels' not in kw)
        # required field
        assert('foreground_threshold' in kw)
        assert('background_threshold_low' in kw)
        assert('background_threshold_high' in kw)
        assert('foreground_fraction' in kw)
        assert('num_classes' in kw)
        assert('num_sampled_rois_per_image' in kw)
        assert('bbox_reg_weights' in kw)

        if isinstance(rois_blob, Blob):
            kw['rois'] = str(rois_blob)
        elif isinstance(rois_blob, str):
            kw['rois'] = rois_blob
        else:
            assert False, 'ProposalTarget rois blob is invalid'

        if isinstance(gt_boxes, Blob):
            kw['gt_boxes'] = str(gt_boxes)
        elif isinstance(gt_boxes, str):
            kw['gt_boxes'] = gt_boxes
        else:
            assert False, 'ProposalTarget gt_boxes blob is invalid'

        if isinstance(gt_labels, Blob):
            kw['gt_labels'] = str(gt_labels)
        elif isinstance(gt_labels, str):
            kw['gt_labels'] = gt_labels
        else:
            assert False, 'ProposalTarget gt_labels blob is invalid'

        # output blob logic_blob_name
        if 'out_rois' not in kw:
            kw['out_rois'] = 'rois'
        if 'labels' not in kw:
            kw['labels'] = 'labels'
        if 'bbox_targets' not in kw:
            kw['bbox_targets'] = 'bbox_targets'
        if 'bbox_inside_weights' not in kw:
            kw['bbox_inside_weights'] = 'bbox_inside_weights'
        if 'bbox_outside_weights' not in kw:
            kw['bbox_outside_weights'] = 'bbox_outside_weights'

        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator('ProposalTarget', name, [], [], **kw)

        ret = (
            Blob(self, '%s/%s' % (name, kw['out_rois'])),
            Blob(self, '%s/%s' % (name, kw['labels'])),
            Blob(self, '%s/%s' % (name, kw['bbox_targets'])),
            Blob(self, '%s/%s' % (name, kw['bbox_inside_weights'])),
            Blob(self, '%s/%s' % (name, kw['bbox_outside_weights'])),
        )
        return ret


    def AnchorTarget(self, gt_boxes, layers, name=None, **kw):
        # required field
        assert('anchor_generator_conf' in kw)
        assert('positive_overlap_threshold' in kw)
        assert('negative_overlap_threshold' in kw)
        assert('total_subsample_num' in kw)
        assert('foreground_fraction' in kw)
        assert('bbox_reg_weights' in kw)

        # input blob
        if isinstance(gt_boxes, Blob):
            kw['gt_boxes'] = gt_boxes.logical_blob_name
        elif isinstance(gt_boxes, str):
            kw['gt_boxes'] = gt_boxes
        else:
            assert False, 'AnchorTarget gt_boxes param is invalid'

        # output blobs
        num_layers = 0
        if isinstance(layers, int):
            if layers > 1:
                rg = range(layers)
                kw['rpn_labels'] = ['rpn_labels_int32_fpn%d' % (i) for i in rg]
                kw['rpn_bbox_targets'] = ['rpn_bbox_targets_fpn%d' % (i) for i in rg]
                kw['rpn_bbox_inside_weights'] = ['rpn_bbox_inside_weights_fpn%d' % (i) for i in rg]
                kw['rpn_bbox_outside_weights'] = ['rpn_bbox_outside_weights_fpn%d' % (i) for i in rg]
            elif layers == 1:
                kw['rpn_labels'] = ['rpn_labels_int32']
                kw['rpn_bbox_targets'] = ['rpn_bbox_targets']
                kw['rpn_bbox_inside_weights'] = ['rpn_bbox_inside_weights']
                kw['rpn_bbox_outside_weights'] = ['rpn_bbox_outside_weights']
            num_layers = layers
        elif isinstance(layers, list):
            kw['rpn_labels'] = []
            kw['rpn_bbox_targets'] = []
            kw['rpn_bbox_inside_weights'] = []
            kw['rpn_bbox_outside_weights'] = []
            for layer in layers:
                kw['rpn_labels'].append('rpn_labels_int32_fpn%d' % (layer))
                kw['rpn_bbox_targets'].append('rpn_bbox_targets_fpn%d' % (layer))
                kw['rpn_bbox_inside_weights'].append('rpn_bbox_inside_weights_fpn%d' % (layer))
                kw['rpn_bbox_outside_weights'].append('rpn_bbox_outside_weights_fpn%d' % (layer))
            num_layers = len(layers)
        else:
            assert False, 'AnchorTarget layers param is invalid'

        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator('AnchorTarget', name, [], [], **kw)

        if num_layers == 1:
            ret = (
                Blob(self, '%s/%s' % (name, kw['rpn_labels'][0])),
                Blob(self, '%s/%s' % (name, kw['rpn_bbox_targets'][0])),
                Blob(self, '%s/%s' % (name, kw['rpn_bbox_inside_weights'][0])),
                Blob(self, '%s/%s' % (name, kw['rpn_bbox_outside_weights'][0]))
            )
        elif num_layers > 1:
            ret = (
                [Blob(self, '%s/%s' % (name, lbn)) for lbn in kw['rpn_labels']],
                [Blob(self, '%s/%s' % (name, lbn)) for lbn in kw['rpn_bbox_targets']],
                [Blob(self, '%s/%s' % (name, lbn)) for lbn in kw['rpn_bbox_inside_weights']],
                [Blob(self, '%s/%s' % (name, lbn)) for lbn in kw['rpn_bbox_outside_weights']]
            )

        return ret

    def FpnCollect(self, rpn_rois_fpn_blobs, rpn_roi_probs_fpn_blobs, name=None, **kw):
        assert('top_n_per_image' in kw);
        assert('rpn_rois_fpn' not in kw);
        assert('rpn_roi_probs_fpn' not in kw);
        assert(type(rpn_rois_fpn_blobs) == list);
        assert(type(rpn_roi_probs_fpn_blobs) == list);
        name = self._GetVariableName(name, util.GetCurFuncName());
        if 'out' not in kw:
            kw['out'] = 'out';
        kw['rpn_rois_fpn'] = [];
        kw['rpn_roi_probs_fpn'] = [];
        for rois, roi_probs in zip(rpn_rois_fpn_blobs, rpn_roi_probs_fpn_blobs):
            kw['rpn_rois_fpn'].append(str(rois));
            kw['rpn_roi_probs_fpn'].append(str(roi_probs));
        self.CreateOperator('FpnCollect', name, [], [], **kw);
        return Blob(self, "%s/%s"%(name, kw['out']));

    def FpnDistribute(self, in_blob, name=None, prefix='', **kw):
        assert('roi_min_level' in kw);
        assert('roi_max_level' in kw);
        roi_min_level = int(kw['roi_min_level'])
        roi_max_level = int(kw['roi_max_level'])
        assert(roi_min_level < roi_max_level)

        assert('roi_canonical_level' in kw);
        assert('roi_canonical_scale' in kw);

        name = self._GetVariableName(name, util.GetCurFuncName());
        kw['collected_rois'] = in_blob.logical_blob_name; #FpnCollect only

        rois = [];
        roi_blobs = [];
        for level in range(roi_min_level, roi_max_level + 1):
            rois.append(prefix + 'rois_fpn' + str(level));
            roi_blobs.append(Blob(self, "%s/%s"%(name, prefix + 'rois_fpn' + str(level))));
        kw['rois'] = rois
        kw['roi_indices'] = 'roi_indices'

        self.CreateOperator('FpnDistribute', name, [], [], **kw);
        return Blob(self, "%s/%s"%(name, kw['roi_indices'])), roi_blobs;

    def BboxNmsAndLimit(self, bbox_blob, bbox_prob, bbox_pred, name=None, **kw):
        assert('score_threshold' in kw);
        assert('nms_threshold' in kw);
        assert('detections_per_im' in kw);
        assert('threshold' in kw);
        assert('image_height' in kw);
        assert('image_width' in kw);
        assert('bbox_vote_enabled' in kw);
        assert('bbox_reg_weights' in kw);

        name = self._GetVariableName(name, util.GetCurFuncName());
        kw['bbox'] = bbox_blob.logical_blob_name;
        kw['bbox_prob'] = str(bbox_prob);
        kw['bbox_pred'] = str(bbox_pred);
        kw['out_bbox'] = 'out_bbox'
        kw['out_bbox_score'] = 'out_bbox_score'
        kw['out_bbox_label'] = 'out_bbox_label'

        self.CreateOperator('BboxNmsAndLimit', name, [], [], **kw);
        return Blob(self, "%s/%s"%(name, kw['out_bbox'])), \
                     Blob(self, "%s/%s"%(name, kw['out_bbox_score'])), \
                     Blob(self, "%s/%s"%(name, kw['out_bbox_label']))

    def ImageSegmentationMask(self, in_blob, rois, roi_labels, name=None, **kw):
        assert('im_width' in kw);
        assert('im_height' in kw);
        name = self._GetVariableName(name, util.GetCurFuncName());
        kw['roi_labels'] = roi_labels.logical_blob_name;
        kw['rois'] = rois.logical_blob_name;
        kw['masks'] = in_blob.logical_blob_name;
        kw['out'] = 'out';
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw);
        return Blob(self, "%s/out" % name);

    def MaskTarget(self, rois_blob, labels_blob, gt_segm, name=None, **kw):
        # input blobs
        assert('rois' not in kw)
        assert('labels' not in kw)
        assert('gt_segm_polygon_lists' not in kw)
        # required field
        assert('num_classes' in kw)
        assert('mask_height' in kw)
        assert('mask_width' in kw)

        if isinstance(rois_blob, Blob):
            kw['rois'] = str(rois_blob)
        elif isinstance(rois_blob, str):
            kw['rois'] = rois_blob
        else:
            assert False, 'MaskTarget rois blob is invalid'

        if isinstance(labels_blob, Blob):
            kw['labels'] = str(labels_blob)
        elif isinstance(labels_blob, str):
            kw['labels'] = labels_blob
        else:
            assert False, 'MaskTarget labels blob is invalid'

        if isinstance(gt_segm, Blob):
            kw['gt_segm_polygon_lists'] = str(gt_segm)
        elif isinstance(gt_segm, str):
            kw['gt_segm_polygon_lists'] = gt_segm
        else:
            assert False, 'MaskTarget gt_segm_polygon_lists blob is invalid'

        # output blob logic_blob_name
        if 'mask_rois' not in kw:
            kw['mask_rois'] = 'mask_rois'
        if 'masks' not in kw:
            kw['masks'] = 'masks_int32'

        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator('MaskTarget', name, [], [], **kw)

        ret = (
            Blob(self, '%s/%s' % (name, kw['mask_rois'])),
            Blob(self, '%s/%s' % (name, kw['masks']))
        )
        return ret


    def LogCounter(self, in_blob, name=None, interval=10):
        name = self._GetVariableName(name, util.GetCurFuncName());
        conf = {'in': in_blob.logical_blob_name, 'interval': interval};
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **conf);

    def Print(self, name, print_dir, in_blobs, print_names):
        assert(len(in_blobs) > 0);
        in_list = []
        for idx, blob in enumerate(in_blobs):
            one_in = {}
            one_in['lbn'] = blob.logical_blob_name
            one_in['name'] = print_names[idx]
            one_in['encode_case'] = {'raw': {}}
            in_list.append(one_in)

        conf = {'in': in_list, 'print_dir': print_dir};
        name = self._GetVariableName(name, util.GetCurFuncName());
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **conf);

    def BlobDump(self, dump_dir='', name=None, **kw):
        assert('blob' in kw);
        kw['in'] = kw['blob'];
        kw.pop('blob', None);
        kw['dump_dir'] = dump_dir;
        name = self._GetVariableName(name, util.GetCurFuncName());
        self.CreateOperator('BlobDump', name, [], [], **kw);
        return None;

    def Gather(self, in_blob, indices_blob, name=None, **kw):
        kw['indices'] = indices_blob.logical_blob_name
        kw['in'] = in_blob.logical_blob_name
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName());
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def BatchGather(self, in_blob, indices_blob, name=None, **kw):
        kw['indices'] = indices_blob.logical_blob_name
        kw['in'] = in_blob.logical_blob_name
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def ConstantScalar(self, value, **kw):
        return self.ConstantFill(value=value, shape=[1], **kw)

    def VariableLike(self, ndarray, name=None, **kw):
        return self.Variable(
                name=name,
                shape={'dim':ndarray.shape},
                data_type=_NdarrayDType2OFDataType(ndarray.dtype),
                **kw)

    def Variable(self, name=None, **kw):
        assert ('shape' in kw)
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def ConstantFill(self, value, shape=None, **kw):
        if isinstance(value, int):
            kw['initializer'] = {'constant_int_conf': {'value': value}}
        elif isinstance(value, float):
            kw['initializer'] = {'constant_conf': {'value': value}}
        else:
            assert False, "only accept int or float"
        if shape is None:
            kw['shape'] = {'dim':[1]}
        else:
            kw['shape'] = {'dim': shape}
        return self.Constant(**kw)

    def Constant(self, name=None, **kw):
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def ScalarAdd(self, in_blob, scalar, name=None, **kw):
        return self._ScalarBinaryOp(util.GetCurFuncName(), in_blob, scalar, name, **kw);

    def ScalarMul(self, in_blob, scalar, name=None, **kw):
        return self._ScalarBinaryOp(util.GetCurFuncName(), in_blob, scalar, name, **kw);

    def ReduceSum(self, in_blob, name=None, **kw):
        return self._SoleOutputOperator(util.GetCurFuncName(), in_blob, name, **kw);

    def ReduceMean(self, in_blob, name=None, **kw):
        return self._SoleOutputOperator(util.GetCurFuncName(), in_blob, name, **kw);

    def BroadcastAdd(self, a_blob, b_blob, name=None, **kw):
        return self._BroadcastBinaryOp(util.GetCurFuncName(), a_blob, b_blob, name, **kw);

    def BroadcastSub(self, a_blob, b_blob, name=None, **kw):
        return self._BroadcastBinaryOp(util.GetCurFuncName(), a_blob, b_blob, name, **kw);

    def BroadcastMul(self, a_blob, b_blob, name=None, **kw):
        return self._BroadcastBinaryOp(util.GetCurFuncName(), a_blob, b_blob, name, **kw);

    def BroadcastDiv(self, a_blob, b_blob, name=None, **kw):
        return self._BroadcastBinaryOp(util.GetCurFuncName(), a_blob, b_blob, name, **kw);

    def Matmul(self, a_blob, b_blob, name=None, **kw):
        kw['out'] = 'out'
        kw['a'] = a_blob.logical_blob_name
        kw['b'] = b_blob.logical_blob_name
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def BiasAdd(self, a_blob, b_blob, name=None, **kw):
        kw['a'] = a_blob.logical_blob_name
        kw['b'] = b_blob.logical_blob_name
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName());
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def Concat(self, in_blobs, axis, name=None, **kw):
        kw['in'] = [blob.logical_blob_name for blob in in_blobs]
        kw['axis'] = axis
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def OneHot(self, indices, depth, name=None, **kw):
        kw['indices'] = indices.logical_blob_name
        kw['depth'] = depth
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def Add(self, in_blobs, activation=op_conf_pb2.kNone, name=None, **kw):
        kw['in'] = [blob.logical_blob_name for blob in in_blobs]
        if activation != op_conf_pb2.kNone:
            kw['activation'] = activation
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def Conv2DWithBN(self, input_blob, **kw):
        dl_net = input_blob.dl_net()
        input_blob = dl_net.Conv2D(input_blob, name='', use_bias=False, **kw)
        input_blob = dl_net.Normalization(input_blob, name='BatchNorm', axis=1, momentum=0.995,
                                                                            epsilon=0.001, scale=False, activation=op_conf_pb2.kRelu)
        return input_blob

    def SparseSoftmaxCrossEntropyLoss(self, prediction, label, name=None, **kw):
        kw['prediction'] = prediction.logical_blob_name
        kw['label'] = label.logical_blob_name
        kw['loss'] = 'loss'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def IdentityLoss(self, prediction,  name=None, **kw):
        kw['prediction'] = prediction.logical_blob_name
        kw['loss'] = 'loss'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/loss'.format(name))

    def SparseCrossEntropy(self, prediction, label, name=None, **kw):
        kw['prediction'] = prediction.logical_blob_name
        kw['label'] = label.logical_blob_name
        kw['out'] = 'out'
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/out'.format(name))

    def Accuracy(self, prediction, label, weight=None, name=None, **kw):
        kw['prediction'] = prediction.logical_blob_name
        kw['label'] = label.logical_blob_name
        kw['accuracy'] = 'accuracy'
        if weight is not None:
            kw['weight'] = weight.logical_blob_name
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return Blob(self, '{}/accuracy'.format(name)), Blob(self, '{}/accuracy_instance_num'.format(name))

    def TupleIdentity(self, in_blobs, name=None, **kw):
        kw['in'] = list(map(lambda blob: blob.logical_blob_name, in_blobs))
        kw['out'] = list(map(lambda i: 'out_{}'.format(i), range(0, len(in_blobs))))
        name = self._GetVariableName(name, util.GetCurFuncName())
        self.CreateOperator(util.GetCurFuncName(), name, [], [], **kw)
        return  list(map(lambda i: Blob(self, '{}/out_{}'.format(name, i)), range(0, len(in_blobs))))

    @contextmanager
    def VariableScope(self, scope_name):
        assert isinstance(scope_name, str);
        self.variable_scope_.append(scope_name);
        try:
            yield None;
        finally:
            self.variable_scope_ = self.variable_scope_[0:-1];

    def _BroadcastBinaryOp(self, op_type_name, a_blob, b_blob, name=None, **kw):
        kw['a'] = a_blob.logical_blob_name;
        kw['b'] = b_blob.logical_blob_name;
        return self._SoleOutputOperator(op_type_name, [], name=name, **kw);

    def _ScalarBinaryOp(self, op_type_name, in_blob, scalar, name=None, **kw):
        assert type(scalar) is int or type(scalar) is float
        kw["int_operand" if type(scalar) is int else "float_operand"] = scalar
        return self._SoleOutputOperator(op_type_name, in_blob, name=name, **kw);

    def _SoleOutputOperator(self, op_type_name, in_blobs, name=None, out=None, **kw):
        assert('in' not in kw);
        assert('out' not in kw);
        if isinstance(in_blobs, Blob):
            in_lbn = in_blobs.logical_blob_name
        elif isinstance(in_blobs, str):
            in_lbn = in_blobs
        else:
            in_lbn = [blob.logical_blob_name for blob in in_blobs];
        name = self._GetVariableName(name, op_type_name);
        if out is None: out = "out";
        self.CreateOperator(op_type_name, name, in_lbn, out, **kw);
        return Blob(self, "%s/%s"%(name, out));

    def _LossOperator(self, op_type_name, prediction_blob, label_blob, name=None, **kw):
        assert('prediction' not in kw);
        assert('label' not in kw);
        assert('loss' not in kw);
        name = self._GetVariableName(name, op_type_name);
        kw['prediction'] = prediction_blob.logical_blob_name;
        kw['label'] = label_blob.logical_blob_name;
        kw['loss'] = 'loss';
        self.CreateOperator(op_type_name, name, [], [], **kw);
        return None;

    def _VariablePrefix(self):
        if len(self.variable_scope_) == 0: return "";
        return "-".join(self.variable_scope_) + "-";

    def _GetVariableName(self, name, op_type_name):
        seq_no = _NewSeqNo(self._VariablePrefix() + op_type_name);
        if name is None: name = "%s_%d" % (op_type_name, seq_no);
        name.replace('/', '-');
        return self._VariablePrefix() + name;

    def dl_net_conf(self):
        return self.dl_net_conf_;

    def CreateOperator(self, op_type_name, op_name, input_lbn, output_bn, **kw):
        op_conf = self.dl_net_conf_.op.add()
        ret = _CreateOperator(op_conf, op_type_name, op_name, input_lbn, output_bn, kw)
        placement_ctx.CurPlacementGroupAddOpConf(op_conf)
        return op_conf

    def __str__(self):
        return text_format.MessageToString(self.dl_net_conf_);

    def OpNames(self):
        op_names = []
        for op in self.dl_net_conf_.op:
            op_names.append(op.name)
        return op_names

    def CurScopeOpNamesExceptCpuOnly(self):
        prefix = self._VariablePrefix();
        for op in self.dl_net_conf_.op:
            if (op.name.startswith(prefix)):
                if op_util.IsOpConfOnlyCpuSupported(op):
                    yield op.name;

    def UnplacedCpuOnlyOpNames(self, placement):
        ret = []
        for op in self.UnplacedOps(placement):
            if op_util.IsOpConfOnlyCpuSupported(op):
                ret.append(op.name)
        return ret

    def UnplacedOps(self, placement):
        placed = set(placement.AllOpNames())
        ret = []
        for op in self.dl_net_conf_.op:
            if not op.name in placed:
                ret.append(op)
        return ret

def _CreateOperator(op_conf_msg, op_type_name, op_name, input_lbn, output_bn, kw):
    assert(type(input_lbn) in [str, tuple, list]);
    if len(input_lbn) > 0: kw['in'] = input_lbn;
    assert(type(output_bn) in [str, tuple, list]);
    if len(output_bn) > 0: kw['out'] = output_bn;
    py_dict = {
        'name': op_name,
        OP_CONF_TYPE_2_FIELD_NAME[op_type_name + 'OpConf']: kw
    };
    if kw.pop('trainable', True) == False: py_dict['trainable'] = False;
    return pb_util.PythonDict2PbMessage(py_dict, op_conf_msg);

def _FilterOperatorConfOpTypeFieldNames():
    fields = OperatorConf().DESCRIPTOR.fields_by_name.keys();
    def is_one_of_op_type_field(field):
        op_conf = OperatorConf();
        try:
            getattr(op_conf, field).CopyFrom(type(getattr(op_conf, field))());
        except:
            pass
        return op_conf.WhichOneof('op_type') == field;
    return filter(is_one_of_op_type_field, fields);

def _GetOpConfTypeName2FieldName():
    fields = _FilterOperatorConfOpTypeFieldNames();
    op_conf = OperatorConf();
    ret_dict = {}
    for field in fields:
        ret_dict[getattr(op_conf, field).DESCRIPTOR.name] = field;
    return ret_dict;

OP_CONF_TYPE_2_FIELD_NAME = _GetOpConfTypeName2FieldName();

def _GetSoleOutputOperatorNamePrefixs():
    global OP_CONF_TYPE_2_FIELD_NAME;
    op_prefixes = [];
    for op_conf_type_name, field_name in OP_CONF_TYPE_2_FIELD_NAME.items():
        op_conf = getattr(OperatorConf(), field_name);
        if op_conf_type_name.endswith('OpConf') and hasattr(op_conf, 'in') \
                and hasattr(op_conf, 'out') and  not isinstance(op_conf.out, RepeatedScalarContainer):
            op_prefixes.append(op_conf_type_name[:-len('OpConf')]);
    return op_prefixes;

def _RegisterSoleOutputBlobOpMethods():
    op_prefixes = _GetSoleOutputOperatorNamePrefixs();
    def make_multi_inputs_lambda(op_prefix):
        return lambda self, in_blobs, **kw: self._SoleOutputOperator(op_prefix, in_blobs, **kw);
    for op_prefix in op_prefixes:
        op_conf = getattr(op_conf_pb2, op_prefix + "OpConf")();
        if not hasattr(DLNet, op_prefix):
            setattr(DLNet, op_prefix, make_multi_inputs_lambda(op_prefix));

_RegisterSoleOutputBlobOpMethods();

def _GetLossOperatorNamePrefixs():
    global OP_CONF_TYPE_2_FIELD_NAME;
    op_prefixes = [];
    for op_conf_type_name, field_name in OP_CONF_TYPE_2_FIELD_NAME.items():
        op_conf = getattr(OperatorConf(), field_name);
        if op_conf_type_name.endswith('OpConf') and hasattr(op_conf, 'prediction') \
                and hasattr(op_conf, 'label') and hasattr(op_conf, 'loss'):
            op_prefixes.append(op_conf_type_name[:-len('OpConf')]);
    return op_prefixes;

def _RegisterLossOperatorMethods():
    op_prefixes = _GetLossOperatorNamePrefixs();
    def make_loss_lambda(op_prefix):
        return lambda self, prediction_blob, label_blob, **kw: \
            self._LossOperator(op_prefix, prediction_blob, label_blob, **kw);
    for op_prefix in op_prefixes:
        if not hasattr(DLNet, op_prefix):
            setattr(DLNet, op_prefix, make_loss_lambda(op_prefix));

_RegisterLossOperatorMethods();

_counters = {};
def _NewSeqNo(key):
    if not key in _counters: _counters[key] = -1
    _counters[key] += 1;
    return _counters[key];

def _NdarrayDType2OFDataType(ndarray_dtype):
    if ndarray_dtype == np.float32: return data_type_pb.kFloat
    if ndarray_dtype == np.float64: return data_type_pb.kDouble
    if ndarray_dtype == np.int64: return data_type_pb.kInt32
    if ndarray_dtype == np.int32: return data_type_pb.kInt32
    assert False, "UNIMPLEMENTED %s" % str(ndarray_dtype)
