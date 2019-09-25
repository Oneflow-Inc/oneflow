import oneflow.core.operator.op_conf_pb2 as op_conf_util
import oneflow as flow
from matcher import Matcher


class RPNHead(object):
    def __init__(self, cfg):
        self.cfg = cfg

    # features: list of [C_i, H_i, W_i] wrt. fpn layers
    def build(self, features):
        with flow.deprecated.variable_scope("rpn-head"):
            cls_logits = []  # list of [N, H_i, W_i, A] wrt. fpn layers
            bbox_preds = []  # list of [N, H_i, W_i, 4A] wrt. fpn layers
            for layer_idx, feature in enumerate(features, 1):
                x = flow.layers.conv2d(
                    inputs=feature,
                    filters=256,
                    kernel_size=[3, 3],
                    padding="SAME",
                    data_format="NCHW",
                    strides=[1, 1],
                    dilation_rate=[1, 1],
                    activation=flow.keras.activations.relu,
                    use_bias=True,
                    name="conv{}".format(layer_idx),
                )
                cls_logits.append(
                    flow.transpose(
                        flow.layers.conv2d(
                            x,
                            filters=3,
                            kernel_size=[1, 1],
                            padding="SAME",
                            data_format="NCHW",
                            strides=[1, 1],
                            dilation_rate=[1, 1],
                            use_bias=True,
                            activation=flow.keras.activations.sigmoid,
                            name="cls_logit{}".format(layer_idx),
                        ),
                        perm=[0, 2, 3, 1],
                    )
                )
                bbox_preds.append(
                    flow.transpose(
                        flow.layers.conv2d(
                            x,
                            filters=12,
                            kernel_size=[1, 1],
                            padding="SAME",
                            data_format="NCHW",
                            strides=[1, 1],
                            dilation_rate=[1, 1],
                            use_bias=True,
                            activation=flow.keras.activations.sigmoid,
                            name="bbox_pred{}".format(layer_idx),
                        ),
                        perm=[0, 2, 3, 1],
                    )
                )

            # list (wrt. fpn layers) of list (wrt. images) of [H_i * W_i * A, 4]
            bbox_pred_list = []
            for bbox_pred_per_layer in bbox_preds:
                bbox_pred_list.append(
                    [
                        flow.dynamic_reshape(x, shape=[-1, 4])
                        for x in flow.piece_slice(
                            bbox_pred_per_layer,
                            self.cfg.TRAINING_CONF.IMG_PER_GPU,
                        )
                    ]
                )

            # list (wrt. fpn layer) of list (wrt. images) of [H_i * W_i * A]
            cls_logit_list = []
            for cls_logit_per_layer in cls_logits:
                cls_logit_list.append(
                    [
                        flow.dynamic_reshape(x, shape=[-1])
                        for x in flow.piece_slice(
                            cls_logit_per_layer,
                            self.cfg.TRAINING_CONF.IMG_PER_GPU,
                        )
                    ]
                )

        return cls_logit_list, bbox_pred_list


class RPNLoss(object):
    def __init__(self, cfg):
        self.cfg = cfg

    # anchors_list: list of [num_anchors_i, 4] wrt. fpn layers
    # image_size_list: list of [2,] wrt. images
    # gt_boxes_list: list of [num_gt_boxes, 4] wrt. images
    # bbox_pred_list: list (wrt. fpn layers) of list (wrt. images) of [H_i * W_i * A, 4]
    # cls_logit_list: list (wrt. fpn layer) of list (wrt. images) of [H_i * W_i * A]
    def build(
        self,
        anchors_list,
        image_size_list,
        gt_boxes_list,
        bbox_pred_list,
        cls_logit_list,
    ):
        with flow.deprecated.variable_scope("rpn-loss"):
            sampled_bbox_pred_list = []
            sampled_bbox_target_list = []
            sampled_cls_logit_list = []
            sampled_cls_label_list = []
            sampled_pos_neg_inds_list = []

            # concat bbox_pred from all fpn layers for each image
            # list (wrt. images) of [M, 4]
            bbox_pred_wrt_img_list = []
            for _, tup in enumerate(zip(*bbox_pred_list)):
                bbox_pred_wrt_img_list += [flow.concat(list(tup), axis=0)]

            # concat cls_logit from all fpn layers for each image
            # list (wrt. images) of [M,]
            cls_logit_wrt_img_list = []
            for _, tup in enumerate(zip(*cls_logit_list)):
                cls_logit_wrt_img_list += [flow.concat(list(tup), axis=0)]

            anchors = flow.concat(anchors_list, axis=0)  # # anchors: [M, 4]
            for img_idx, gt_boxes in enumerate(gt_boxes_list):
                with flow.deprecated.variable_scope("matcher"):
                    rpn_matcher = Matcher(
                        self.cfg.RPN.POSITIVE_OVERLAP_THRESHOLD,
                        self.cfg.RPN.NEGATIVE_OVERLAP_THRESHOLD,
                    )
                    matched_indices = rpn_matcher.build(anchors, gt_boxes, True)

                # exclude outside anchors
                matched_indices = flow.where(
                    flow.detection.identify_outside_anchors(
                        anchors, image_size_list[img_idx], tolerance=0.0
                    ),
                    flow.constant_like(matched_indices, int(-2)),
                    matched_indices,
                )

                pos_inds = flow.squeeze(
                    flow.local_nonzero(
                        matched_indices
                        >= flow.constant_scalar(value=0, dtype=flow.int32)
                    ),
                    axis=[1],
                )
                neg_inds = flow.squeeze(
                    flow.local_nonzero(
                        matched_indices
                        == flow.constant_scalar(value=-1, dtype=flow.int32)
                    ),
                    axis=[1],
                )

                sampled_pos_inds, sampled_neg_inds = flow.detection.pos_neg_sampler(
                    pos_inds,
                    neg_inds,
                    total_subsample_num=self.cfg.RPN.SUBSAMPLE_NUM_PER_IMG,
                    pos_fraction=self.cfg.RPN.FOREGROUND_FRACTION,
                )

                sampled_bbox_target_list.append(
                    flow.detection.box_encode(
                        flow.local_gather(
                            gt_boxes_list[img_idx],
                            flow.local_gather(
                                matched_indices, sampled_pos_inds
                            ),
                        ),
                        flow.local_gather(anchors, sampled_pos_inds),
                        regression_weights={
                            "weight_x": self.cfg.RPN.WEIGHT_X,
                            "weight_y": self.cfg.RPN.WEIGHT_Y,
                            "weight_h": self.cfg.RPN.WEIGHT_H,
                            "weight_w": self.cfg.RPN.WEIGHT_W,
                        },
                    )
                )
                sampled_bbox_pred_list.append(
                    flow.local_gather(
                        bbox_pred_wrt_img_list[img_idx], sampled_pos_inds
                    )
                )

                cls_labels = matched_indices >= flow.constant_scalar(
                    value=0, dtype=flow.int32
                )
                sampled_pos_neg_inds = flow.concat(
                    [sampled_pos_inds, sampled_neg_inds], axis=0
                )
                sampled_pos_neg_inds_list.append(sampled_pos_neg_inds)
                sampled_cls_logit_list.append(
                    flow.local_gather(
                        cls_logit_wrt_img_list[img_idx], sampled_pos_neg_inds
                    )
                )
                sampled_cls_label_list.append(
                    flow.local_gather(cls_labels, sampled_pos_neg_inds)
                )

            total_sample_cnt = flow.elem_cnt(
                flow.concat(sampled_pos_neg_inds_list, axis=0)
            )

            bbox_loss = flow.math.reduce_sum(
                flow.detection.smooth_l1(
                    flow.concat(sampled_bbox_pred_list, axis=0),
                    flow.concat(sampled_bbox_target_list, axis=0),
                    beta=1.0 / 9.0,
                )
            ) / flow.cast(total_sample_cnt, dtype=flow.float)

            cls_loss = flow.math.reduce_sum(
                flow.nn.sigmoid_cross_entropy_with_logits(
                    flow.concat(sampled_cls_label_list, axis=0),
                    flow.concat(sampled_cls_logit_list, axis=0),
                )
            ) / flow.cast(total_sample_cnt, dtype=flow.float)
        return bbox_loss, cls_loss


def safe_top_k(inputs, k):
    assert len(inputs.shape) == 1
    if inputs.shape[0] < k:
        return flow.math.top_k(inputs, inputs.shape[0])
    else:
        return flow.math.top_k(inputs, k)


class RPNProposal(object):
    def __init__(self, cfg):
        self.cfg = cfg

    # anchors_list: list of [num_anchors_i, 4] wrt. fpn layers
    # image_size_list: list of [2,] wrt. images
    # gt_boxes_list: list of [num_gt_boxes, 4] wrt. images
    # bbox_pred_list: list (wrt. fpn layers) of list (wrt. images) of [H_i * W_i * A, 4]
    # cls_logit_list: list (wrt. fpn layer) of list (wrt. images) of [H_i * W_i * A]
    def build(
        self,
        anchors,
        cls_logit_list,
        bbox_pred_list,
        image_size_list,
        resized_gt_boxes_list,
    ):
        with flow.deprecated.variable_scope("rpn-postprocess"):
            cls_logit_list = list(zip(*cls_logit_list))
            bbox_pred_list = list(zip(*bbox_pred_list))

            proposals = []
            for img_idx in range(len(image_size_list)):
                proposal_list = []
                score_list = []
                for layer_idx in range(len(cls_logit_list[0])):
                    pre_nms_top_k_inds = safe_top_k(
                        cls_logit_list[img_idx][layer_idx],
                        k=self.cfg.RPN.PRE_NMS_TOP_N,
                    )
                    score_per_layer = flow.local_gather(
                        cls_logit_list[img_idx][layer_idx], pre_nms_top_k_inds
                    )
                    proposal_per_layer = flow.detection.box_decode(
                        flow.local_gather(
                            anchors[layer_idx], pre_nms_top_k_inds
                        ),
                        flow.local_gather(
                            bbox_pred_list[img_idx][layer_idx],
                            pre_nms_top_k_inds,
                        ),
                        regression_weights={
                            "weight_x": self.cfg.RPN.WEIGHT_X,
                            "weight_y": self.cfg.RPN.WEIGHT_Y,
                            "weight_h": self.cfg.RPN.WEIGHT_H,
                            "weight_w": self.cfg.RPN.WEIGHT_W,
                        },
                    )

                    # clip to img
                    proposal_per_layer = flow.detection.clip_to_image(
                        proposal_per_layer, image_size_list[img_idx]
                    )

                    # remove small boxes
                    indices = flow.squeeze(
                        flow.local_nonzero(
                            flow.detection.identify_non_small_boxes(
                                proposal_per_layer, min_size=0.0
                            )
                        ),
                        axis=[1],
                    )
                    score_per_layer = flow.local_gather(
                        score_per_layer, indices
                    )
                    proposal_per_layer = flow.local_gather(
                        proposal_per_layer, indices
                    )

                    # NMS
                    indices = flow.squeeze(
                        flow.local_nonzero(
                            flow.detection.nms(
                                proposal_per_layer,
                                nms_iou_threshold=self.cfg.RPN.NMS_THRESH,
                                post_nms_top_n=self.cfg.RPN.POST_NMS_TOP_N,
                            )
                        ),
                        axis=[1],
                    )
                    score_per_layer = flow.local_gather(
                        score_per_layer, indices
                    )
                    proposal_per_layer = flow.local_gather(
                        proposal_per_layer, indices
                    )

                    proposal_list.append(proposal_per_layer)
                    score_list.append(score_per_layer)

                score_in_one_img = flow.concat(score_list, axis=0)
                proposal_in_one_img = flow.concat(proposal_list, axis=0)

                proposal_in_one_img = flow.local_gather(
                    proposal_in_one_img,
                    safe_top_k(score_in_one_img, k=self.cfg.RPN.POST_NMS_TOP_N),
                )
                proposal_in_one_img = flow.concat(
                    [proposal_in_one_img, resized_gt_boxes_list[img_idx]],
                    axis=0,
                )

                proposals.append(proposal_in_one_img)

            return proposals
