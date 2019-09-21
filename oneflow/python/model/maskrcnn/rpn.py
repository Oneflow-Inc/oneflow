import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow as flow
from matcher import Matcher


class RPNHead(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self, layers):
        with flow.deprecated.variable_scope("rpn-head"):
            cls_logits = []
            bbox_preds = []
            for layer_idx, feature in enumerate(layers, 1):
                x = flow.conv2d(
                    inputs=feature,
                    filters=256,
                    kernel_size=[3, 3],
                    padding="SAME",
                    data_format="NCHW",
                    strides=[1, 1],
                    dilation_rate=[1, 1],
                    activation=op_conf_util.kRelu,
                    use_bias=True,
                    name="conv{}".format(layer_idx),
                )
                cls_logits.append(
                    flow.transpose(
                        flow.conv2d(
                            x,
                            filters=3,
                            kernel_size=[1, 1],
                            padding="SAME",
                            data_format="NCHW",
                            strides=[1, 1],
                            dilation_rate=[1, 1],
                            use_bias=True,
                            activation=op_conf_util.kSigmoid,
                            name="cls_logit{}".format(layer_idx),
                        ),
                        perm=[0, 2, 3, 1],
                        name="cls_logits_tp{}".format(layer_idx),
                    )
                )
                bbox_preds.append(
                    flow.transpose(
                        flow.conv2d(
                            x,
                            filters=12,
                            kernel_size=[1, 1],
                            padding="SAME",
                            data_format="NCHW",
                            strides=[1, 1],
                            dilation_rate=[1, 1],
                            use_bias=True,
                            activation=op_conf_util.kSigmoid,
                            name="bbox_pred{}".format(layer_idx),
                        ),
                        perm=[0, 2, 3, 1],
                        name="bbox_preds_tp{}".format(layer_idx),
                    )
                )

            # list (wrt. layers) of list (wrt. images) of [H_i * W_i * A]
            bbox_pred_list = []
            for layer_idx, bbox_pred_per_layer in enumerate(bbox_preds):
                bbox_pred_list.append(
                    [
                        flow.dynamic_reshape(x, shape={"dim": [-1, 4]})
                        for img_idx, x in enumerate(
                            flow.piece_slice(
                                bbox_pred_per_layer, self.cfg.TRAINING_CONF.PIECE_SIZE
                            )
                        )
                    ]
                )

            # list (wrt. layer) of list (wrt. img) of [H_i * W_i * A]
            cls_logit_list = []
            for layer_idx, cls_logit_per_layer in enumerate(cls_logits):
                cls_logit_list.append(
                    [
                        flow.dynamic_reshape(x, shape={"dim": [-1]})
                        for img_idx, x in enumerate(
                            flow.piece_slice(
                                cls_logit_per_layer, self.cfg.TRAINING_CONF.PIECE_SIZE
                            )
                        )
                    ]
                )

        return cls_logit_list, bbox_pred_list


class RPNLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg

    # anchors: list of [num_anchors_i, 4] wrt. fpn layers
    # resized_image_size_list: list of [2,] wrt. images
    # resized_gt_box: list of [num_gt_boxes, 4] wrt. images
    # cls_logit_list: list (wrt. layer) of list (wrt. img) of [H_i * W_i * A]
    # bbox_pred_list: list (wrt. layers) of list (wrt. images) of [H_i * W_i * A]
    def build(
        self,
        anchors,
        resized_image_size_list,
        resized_gt_box_list,
        cls_logit_list,
        bbox_pred_list,
    ):
        with flow.deprecated.variable_scope("rpn-loss"):
            sampled_bbox_pred_list = []
            sampled_bbox_target_list = []
            sampled_cls_logit_list = []
            sampled_cls_label_list = []
            sampled_pos_neg_inds_list = []

            # list (wrt. img) of [M, 4]
            concated_bbox_pred_list = []
            for img_idx, tup in enumerate(zip(*bbox_pred_list)):
                concated_bbox_pred_list += [flow.concat(list(tup), axis=0)
                ]

            # list (wrt. img) of [M,]
            concated_cls_logit = []
            for img_idx, tup in enumerate(zip(*cls_logit_list)):
                concated_cls_logit += [flow.concat(list(tup), axis=0)]

            # 3. construct bbox targets and cls targets
            # concated_anchors: [M, 4], shared with different imgs in the same piece
            concated_anchors = flow.concat(anchors, axis=0)
            for img_idx, gt_boxes in enumerate(resized_gt_box_list):
                self.img_idx = img_idx
                with flow.deprecated.variable_scope("rpn-loss"):
                    matcher = Matcher(
                        self.cfg,
                        self.cfg.RPN.POSITIVE_OVERLAP_THRESHOLD,
                        self.cfg.RPN.NEGATIVE_OVERLAP_THRESHOLD,
                        img_idx,
                    )
                    matched_indices = matcher.build(concated_anchors, gt_boxes, True)

                # exclude outside anchors
                matched_indices = flow.where(
                    flow.detection.identify_outside_anchors(
                        concated_anchors,
                        resized_image_size_list[img_idx],
                        tolerance=0.0,
                    ),
                    flow..constant_like(matched_indices, int(-2)),
                    matched_indices,
                )

                pos_inds = flow.squeeze(
                    flow.local_nonzero(
                        matched_indices >= flow.constant(value=0, dtype=flow.int32, shape=[1]) 
                    ),
                    axis=[1],
                )
                pos_inds = flow.squeeze(
                    flow.local_nonzero(
                        matched_indices == flow.constant(value=-1, dtype=flow.int32, shape=[1]) 
                    ),
                    axis=[1],
                )

                # sampled_pos_inds: [sampled_pos_num,]
                # sampled_neg_inds: [sampled_neg_num,]
                sampled_pos_inds, sampled_neg_inds = flow.detection.pos_neg_sampler(
                    pos_inds,
                    neg_inds,
                    total_subsample_num=self.cfg.RPN.SUBSAMPLE_NUM_PER_IMG,
                    pos_fraction=self.cfg.RPN.FOREGROUND_FRACTION,
                )
                # bbox target and bbox pred
                sampled_bbox_target_list.append(
                    flow.detection.box_encode(
                        flow.local_gather(
                            resized_gt_box_list[img_idx],
                            flow.local_gather(matched_indices, sampled_pos_inds),
                        ),
                        flow.local_gather(concated_anchors, sampled_pos_inds),
                        regression_weights={
                            "weight_x": self.cfg.RPN.WEIGHT_X,
                            "weight_y": self.cfg.RPN.WEIGHT_Y,
                            "weight_h": self.cfg.RPN.WEIGHT_H,
                            "weight_w": self.cfg.RPN.WEIGHT_W,
                        },
                    )
                )
                sampled_bbox_pred_list.append(
                    flow.local_gather(concated_bbox_pred_list[img_idx], sampled_pos_inds)
                )
                # cls label and cls logit
                cls_labels = matched_indices >= flow.constant(value=0 dtype=flow.int32, shape=[1])
                sampled_pos_neg_inds = flow.concat(
                    [sampled_pos_inds, sampled_neg_inds], axis=0
                )
                sampled_pos_neg_inds_list.append(sampled_pos_neg_inds)
                sampled_cls_logit_list.append(
                    flow.local_gather(concated_cls_logit[img_idx], sampled_pos_neg_inds)
                )
                sampled_cls_label_list.append(
                    flow.local_gather(cls_labels, sampled_pos_neg_inds)
                )

            total_sample_cnt = flow.elem_cnt(
                flow.concat(sampled_pos_neg_inds_list, axis=0)
            )
            # rpn bbox_loss
            bbox_loss = flow.div(
                flow.math.reduce_sum(
                    flow.detection.smooth_l1(
                        flow.concat.concat(sampled_bbox_pred_list, axis=0),
                        flow.concat.concat(sampled_bbox_target_list, axis=0),
                        beta=1.0 / 9.0,
                    )
                ),
                flow.cast(total_sample_cnt, data_type=flow.float),
            )

            cls_loss = flow.div(
                flow.math.reduce_sum(
                    flow.detection.binary_cross_entroy(
                        flow.concat(sampled_cls_logit_list, axis=0),
                        flow.concat(sampled_cls_label_list, axis=0),
                    )
                ),
                flow.cast(total_sample_cnt, data_type=flow.float),
            )

        return bbox_loss, cls_loss
