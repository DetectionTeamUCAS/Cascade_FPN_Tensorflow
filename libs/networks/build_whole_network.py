# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet, resnet_gluoncv
from libs.networks import mobilenet_v2
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import anchor_utils
from libs.configs import cfgs
from libs.losses import losses
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_opr import postprocess_rpn_proposals
from libs.detection_oprations.anchor_target_layer_without_boxweight import anchor_target_layer
from libs.detection_oprations.proposal_target_layer_cascade import proposal_target_layer


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.num_anchors_per_location = len(cfgs.ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)
        elif self.base_network_name in ['resnet101_v1d', 'resnet50_v1d']:
            return resnet_gluoncv.resnet_base(input_img_batch, scope_name=self.base_network_name,
                                              is_training=self.is_training)
        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def postprocess_cascadercnn(self, rois, bbox_pred, cls_score, stage):
        '''
        :param rois:[-1, 4]
        :param bbox_ppred: bbox_ppred: [-1, 4]
        :param scores: [-1, 1]
        :return:
        '''
        # rois = tf.stop_gradient(rois)
        # bbox_pred = tf.stop_gradient(bbox_pred)
        bbox_pred_ins = tf.reshape(bbox_pred, [-1, cfgs.CLASS_NUM + 1, 4])

        # only keep a box which score is the bigest
        keep_abox = tf.argmax(cls_score, axis=1)
        keep_inds = tf.reshape(tf.transpose(tf.stack([tf.cumsum(tf.ones_like(keep_abox)) - 1, keep_abox])),
                               [-1, 2])
        bbox_pred_fliter = tf.reshape(tf.gather_nd(bbox_pred_ins, keep_inds), [-1, 4])

        # decode boxes
        decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=bbox_pred_fliter,
                                                       reference_boxes=rois,
                                                       scale_factors=cfgs.ROI_SCALE_FACTORS[stage-1])

        return decoded_boxes

    def postprocess_fastrcnn(self, rois, bbox_ppred, scores, img_shape):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn'):
            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM + 1):
                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS[-1])
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                # 3. NMS
                keep = tf.image.non_max_suppression(
                    boxes=tmp_decoded_boxes,
                    scores=tmp_score,
                    max_output_size=cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS,
                    iou_threshold=cfgs.FAST_RCNN_NMS_IOU_THRESHOLD)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])

                final_boxes = tf.gather(final_boxes, kept_indices)
                final_scores = tf.gather(final_scores, kept_indices)
                final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def roi_pooling(self, feature_maps, rois, img_shape, scope):
        '''
        Here use roi warping as roi_pooling

        :param featuremaps_dict: feature map to crop
        :param rois: shape is [-1, 4]. [x1, y1, x2, y2]
        :return:
        '''

        with tf.variable_scope('ROI_Warping_' + scope):
            img_h, img_w = tf.cast(img_shape[1], tf.float32), tf.cast(img_shape[2], tf.float32)
            N = tf.shape(rois)[0]
            x1, y1, x2, y2 = tf.unstack(rois, axis=1)

            normalized_x1 = x1 / img_w
            normalized_x2 = x2 / img_w
            normalized_y1 = y1 / img_h
            normalized_y2 = y2 / img_h

            normalized_rois = tf.transpose(
                tf.stack([normalized_y1, normalized_x1, normalized_y2, normalized_x2]), name='get_normalized_rois')

            normalized_rois = tf.stop_gradient(normalized_rois)

            cropped_roi_features = tf.image.crop_and_resize(feature_maps, normalized_rois,
                                                            box_ind=tf.zeros(shape=[N, ],
                                                                             dtype=tf.int32),
                                                            crop_size=[cfgs.ROI_SIZE, cfgs.ROI_SIZE],
                                                            name='CROP_AND_RESIZE'
                                                            )
            roi_features = slim.max_pool2d(cropped_roi_features,
                                           [cfgs.ROI_POOL_KERNEL_SIZE, cfgs.ROI_POOL_KERNEL_SIZE],
                                           stride=cfgs.ROI_POOL_KERNEL_SIZE)

        return roi_features

    def build_fastrcnn(self, P_list, rois_list, img_shape):

        with tf.variable_scope('Fast-RCNN'):
            # 5. ROI Pooling
            with tf.variable_scope('rois_pooling'):
                pooled_features_list = []
                for level_name, p, rois in zip(cfgs.LEVLES, P_list, rois_list):  # exclude P6_rois
                    # p = tf.Print(p, [tf.shape(p)], summarize=10, message=level_name+'SHPAE***')
                    pooled_features = self.roi_pooling(feature_maps=p, rois=rois, img_shape=img_shape,
                                                       scope=level_name)
                    pooled_features_list.append(pooled_features)

                pooled_features = tf.concat(pooled_features_list, axis=0)  # [minibatch_size, H, W, C]

            # 6. inferecne rois in Fast-RCNN to obtain fc_flatten features
            if self.base_network_name.startswith('resnet'):
                fc_flatten = resnet.restnet_head(inputs=pooled_features,
                                                 is_training=self.is_training,
                                                 scope_name=self.base_network_name)
            elif self.base_network_name.startswith('Mobile'):
                fc_flatten = mobilenet_v2.mobilenetv2_head(inputs=pooled_features,
                                                           is_training=self.is_training)
            else:
                raise NotImplementedError('only support resnet and mobilenet')

            # 7. cls and reg in Fast-RCNN
            with slim.arg_scope([slim.fully_connected], weights_regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

                cls_score = slim.fully_connected(fc_flatten,
                                                 num_outputs=cfgs.CLASS_NUM + 1,
                                                 weights_initializer=cfgs.INITIALIZER,
                                                 activation_fn=None, trainable=self.is_training,
                                                 scope='cls_fc')

                bbox_pred = slim.fully_connected(fc_flatten,
                                                 num_outputs=(cfgs.CLASS_NUM + 1) * 4,
                                                 activation_fn=None, trainable=self.is_training,
                                                 weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.001),
                                                 scope='reg_fc')

                # for convient. It also produce (cls_num +1) bboxes
                cls_score = tf.reshape(cls_score, [-1, cfgs.CLASS_NUM + 1])
                bbox_pred = tf.reshape(bbox_pred, [-1, 4 * (cfgs.CLASS_NUM + 1)])

                return bbox_pred, cls_score

    def assign_levels(self, all_rois, labels=None, bbox_targets=None):
        '''

        :param all_rois:
        :param labels:
        :param bbox_targets:
        :return:
        '''
        with tf.name_scope('assign_levels'):
            # all_rois = tf.Print(all_rois, [tf.shape(all_rois)], summarize=10, message='ALL_ROIS_SHAPE*****')
            xmin, ymin, xmax, ymax = tf.unstack(all_rois, axis=1)

            h = tf.maximum(0., ymax - ymin)
            w = tf.maximum(0., xmax - xmin)

            levels = tf.floor(4. + tf.log(tf.sqrt(w * h + 1e-8) / 224.0) / tf.log(2.))  # 4 + log_2(***)
            # use floor instead of round

            min_level = int(cfgs.LEVLES[0][-1])
            max_level = min(5, int(cfgs.LEVLES[-1][-1]))
            levels = tf.maximum(levels, tf.ones_like(levels) * min_level)  # level minimum is 2
            levels = tf.minimum(levels, tf.ones_like(levels) * max_level)  # level maximum is 5

            levels = tf.stop_gradient(tf.reshape(levels, [-1]))

            def get_rois(levels, level_i, rois, labels, bbox_targets):

                level_i_indices = tf.reshape(tf.where(tf.equal(levels, level_i)), [-1])
                # level_i_indices = tf.Print(level_i_indices, [tf.shape(tf.where(tf.equal(levels, level_i)))[0]], message="SHAPE%d***"%level_i,
                #                            summarize=10)
                tf.summary.scalar('LEVEL/LEVEL_%d_rois_NUM' % level_i, tf.shape(level_i_indices)[0])
                level_i_rois = tf.gather(rois, level_i_indices)

                if self.is_training:
                    if cfgs.CUDA9:
                        # Note: for cuda 9
                        level_i_rois = tf.stop_gradient(level_i_rois)
                        level_i_labels = tf.gather(labels, level_i_indices)

                        level_i_targets = tf.gather(bbox_targets, level_i_indices)
                    else:

                        # Note: for cuda 8
                        level_i_rois = tf.stop_gradient(tf.concat([level_i_rois, [[0, 0, 0., 0.]]], axis=0))
                        # to avoid the num of level i rois is 0.0, which will broken the BP in tf

                        level_i_labels = tf.gather(labels, level_i_indices)
                        level_i_labels = tf.stop_gradient(tf.concat([level_i_labels, [0]], axis=0))

                        level_i_targets = tf.gather(bbox_targets, level_i_indices)
                        level_i_targets = tf.stop_gradient(tf.concat([level_i_targets,
                                                                      tf.zeros(shape=(1, 4 * (cfgs.CLASS_NUM + 1)),
                                                                               dtype=tf.float32)], axis=0))

                    return level_i_rois, level_i_labels, level_i_targets
                else:
                    if not cfgs.CUDA9:
                        # Note: for cuda 8
                        level_i_rois = tf.concat([level_i_rois, [[0, 0, 0., 0.]]], axis=0)
                    return level_i_rois, None, None

            rois_list = []
            labels_list = []
            targets_list = []
            for i in range(min_level, max_level + 1):
                P_i_rois, P_i_labels, P_i_targets = get_rois(levels, level_i=i, rois=all_rois,
                                                             labels=labels,
                                                             bbox_targets=bbox_targets)
                rois_list.append(P_i_rois)
                labels_list.append(P_i_labels)
                targets_list.append(P_i_targets)

            if self.is_training:
                all_labels = tf.concat(labels_list, axis=0)
                all_targets = tf.concat(targets_list, axis=0)
                return rois_list, all_labels, all_targets
            else:
                return rois_list  # [P2_rois, P3_rois, P4_rois, P5_rois] Note: P6 do not assign rois

    def add_anchor_img_smry(self, img, anchors, labels):

        positive_anchor_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])
        negative_anchor_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        positive_anchor = tf.gather(anchors, positive_anchor_indices)
        negative_anchor = tf.gather(anchors, negative_anchor_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=positive_anchor)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=negative_anchor)

        tf.summary.image('positive_anchor', pos_in_img)
        tf.summary.image('negative_anchors', neg_in_img)

    def add_roi_batch_img_smry(self, img, rois, labels):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=pos_roi)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=neg_roi)
        tf.summary.image('pos_rois', pos_in_img)
        tf.summary.image('neg_rois', neg_in_img)

    def build_loss(self, rpn_box_pred, rpn_bbox_targets, rpn_cls_score, rpn_labels,
                   bbox_pred, bbox_targets, cls_score, labels):
        '''

        :param rpn_box_pred: [-1, 4]
        :param rpn_bbox_targets: [-1, 4]
        :param rpn_cls_score: [-1]
        :param rpn_labels: [-1]
        :param bbox_pred: [-1, 4*(cls_num+1)]
        :param bbox_targets: [-1, 4*(cls_num+1)]
        :param cls_score: [-1, cls_num+1]
        :param labels: [-1]
        :return:
        '''
        with tf.variable_scope('build_loss'):

            rpn_cls_loss = 0.0
            rpn_bbox_loss = 0.0
            if rpn_box_pred is not None:
                with tf.variable_scope('rpn_loss'):

                    rpn_bbox_loss = losses.smooth_l1_loss_rpn(bbox_pred=rpn_box_pred,
                                                              bbox_targets=rpn_bbox_targets,
                                                              label=rpn_labels,
                                                              sigma=cfgs.RPN_SIGMA)

                    rpn_select = tf.reshape(tf.where(tf.not_equal(rpn_labels, -1)), [-1])
                    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, rpn_select), [-1, 2])
                    rpn_labels = tf.reshape(tf.gather(rpn_labels, rpn_select), [-1])
                    rpn_cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                                                                 labels=rpn_labels))

                    rpn_cls_loss = rpn_cls_loss * cfgs.RPN_CLASSIFICATION_LOSS_WEIGHT
                    rpn_bbox_loss = rpn_bbox_loss * cfgs.RPN_LOCATION_LOSS_WEIGHT

            with tf.variable_scope('FastRCNN_loss'):
                if not cfgs.FAST_RCNN_MINIBATCH_SIZE == -1:
                    bbox_loss = losses.smooth_l1_loss_rcnn(bbox_pred=bbox_pred,
                                                           bbox_targets=bbox_targets,
                                                           label=labels,
                                                           num_classes=cfgs.CLASS_NUM + 1,
                                                           sigma=cfgs.FASTRCNN_SIGMA)

                    cls_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=cls_score,
                        labels=labels))  # beacause already sample before
                else:
                    ''' 
                    applying OHEM here
                    '''
                    print(20 * "@@")
                    print("@@" + 10 * " " + "TRAIN WITH OHEM ...")
                    print(20 * "@@")
                    cls_loss = bbox_loss = losses.sum_ohem_loss(
                        cls_score=cls_score,
                        label=labels,
                        bbox_targets=bbox_targets,
                        nr_ohem_sampling=128,
                        nr_classes=cfgs.CLASS_NUM + 1)
                cls_loss = cls_loss * cfgs.FAST_RCNN_CLASSIFICATION_LOSS_WEIGHT
                # bbox_loss = bbox_loss * cfgs.FAST_RCNN_LOCATION_LOSS_WEIGHT
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_loc_loss': rpn_bbox_loss,
                'fastrcnn_cls_loss': cls_loss,
                'fastrcnn_loc_loss': bbox_loss
            }
            return loss_dict

    def cascade_rcnn(self, cascade_rois, gtboxes_batch, input_img_batch, P_list,
                     img_shape, fg_threshold, fpn_box_pred, fpn_bbox_targets,
                     fpn_cls_score, fpn_labels, stage=None):

        with tf.variable_scope('cascade_stage{}'.format(stage)):
            if self.is_training:
                # with tf.control_dependencies([fpn_labels]):
                with tf.variable_scope('sample_RCNN_minibatch'):
                    cascade_rois, labels, bbox_targets, gtboxes = \
                        tf.py_func(proposal_target_layer,
                                   [cascade_rois, gtboxes_batch, fg_threshold, stage],
                                   [tf.float32, tf.float32, tf.float32, tf.float32])
                    cascade_rois = tf.reshape(cascade_rois, [-1, 4])
                    labels = tf.to_int32(labels)
                    labels = tf.reshape(labels, [-1])
                    bbox_targets = tf.reshape(bbox_targets, [-1, 4 * (cfgs.CLASS_NUM + 1)])
                    self.add_roi_batch_img_smry(input_img_batch, cascade_rois, labels)
            if self.is_training:
                rois_list, labels, bbox_targets = self.assign_levels(all_rois=cascade_rois,
                                                                     labels=labels,
                                                                     bbox_targets=bbox_targets)
            else:
                rois_list = self.assign_levels(all_rois=cascade_rois)

            # ----------------------------------------------------------------------------------------------------#
            #                                            Fast-RCNN                                                #
            # ----------------------------------------------------------------------------------------------------#

            # 5. build Fast-RCNN
            bbox_pred, cls_score = self.build_fastrcnn(P_list=P_list,
                                                       rois_list=rois_list,
                                                       img_shape=img_shape)

            cls_prob = slim.softmax(cls_score, 'cls_prob')

            # ----------------------------------------------add smry----------------------------------------------
            if self.is_training:
                cls_category = tf.argmax(cls_prob, axis=1)
                fast_acc = tf.reduce_mean(tf.to_float(tf.equal(cls_category, tf.to_int64(labels))))
                tf.summary.scalar('ACC/fast_acc', fast_acc)

            cascade_rois = tf.concat(rois_list, axis=0, name='concat_rois')
            #  6. postprocess_fastrcnn
            if stage != 3:
                cascade_rois = self.postprocess_cascadercnn(rois=cascade_rois,
                                                            bbox_pred=bbox_pred,
                                                            cls_score=cls_score,
                                                            stage=stage)

                if self.is_training:
                    loss_dict = self.build_loss(rpn_box_pred=None,
                                                rpn_bbox_targets=None,
                                                rpn_cls_score=None,
                                                rpn_labels=None,
                                                bbox_pred=bbox_pred,
                                                bbox_targets=bbox_targets,
                                                cls_score=cls_score,
                                                labels=labels)
                    return cascade_rois, bbox_pred, cls_prob, loss_dict
                else:
                    return cascade_rois, bbox_pred, cls_prob

            else:
                if self.is_training:
                    loss_dict = self.build_loss(rpn_box_pred=fpn_box_pred,
                                                rpn_bbox_targets=fpn_bbox_targets,
                                                rpn_cls_score=fpn_cls_score,
                                                rpn_labels=fpn_labels,
                                                bbox_pred=bbox_pred,
                                                bbox_targets=bbox_targets,
                                                cls_score=cls_score,
                                                labels=labels)
                    return cascade_rois, bbox_pred, cls_prob, loss_dict
                else:
                    return cascade_rois, bbox_pred, cls_prob

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch):

        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build base network
        P_list = self.build_base_network(input_img_batch)

        # 2. build rpn
        with tf.variable_scope('build_rpn',
                               regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

            fpn_cls_score = []
            fpn_box_pred = []
            for level_name, p in zip(cfgs.LEVLES, P_list):
                if cfgs.SHARE_HEADS:
                    reuse_flag = None if level_name == cfgs.LEVLES[0] else True
                    scope_list = ['rpn_conv/3x3', 'rpn_cls_score', 'rpn_bbox_pred']
                else:
                    reuse_flag = None
                    scope_list = ['rpn_conv/3x3_%s' % level_name, 'rpn_cls_score_%s' % level_name,
                                  'rpn_bbox_pred_%s' % level_name]
                rpn_conv3x3 = slim.conv2d(
                    p, 512, [3, 3],
                    trainable=self.is_training, weights_initializer=cfgs.INITIALIZER, padding="SAME",
                    activation_fn=tf.nn.relu,
                    scope=scope_list[0],
                    reuse=reuse_flag)
                rpn_cls_score = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location * 2, [1, 1], stride=1,
                                            trainable=self.is_training, weights_initializer=cfgs.INITIALIZER,
                                            activation_fn=None, padding="VALID",
                                            scope=scope_list[1],
                                            reuse=reuse_flag)
                rpn_box_pred = slim.conv2d(rpn_conv3x3, self.num_anchors_per_location * 4, [1, 1], stride=1,
                                           trainable=self.is_training, weights_initializer=cfgs.BBOX_INITIALIZER,
                                           activation_fn=None, padding="VALID",
                                           scope=scope_list[2],
                                           reuse=reuse_flag)
                rpn_box_pred = tf.reshape(rpn_box_pred, [-1, 4])
                rpn_cls_score = tf.reshape(rpn_cls_score, [-1, 2])

                fpn_cls_score.append(rpn_cls_score)
                fpn_box_pred.append(rpn_box_pred)

            fpn_cls_score = tf.concat(fpn_cls_score, axis=0, name='fpn_cls_score')
            fpn_box_pred = tf.concat(fpn_box_pred, axis=0, name='fpn_box_pred')
            fpn_cls_prob = slim.softmax(fpn_cls_score, scope='fpn_cls_prob')

        # 3. generate_anchors
        all_anchors = []
        for i in range(len(cfgs.LEVLES)):
            level_name, p = cfgs.LEVLES[i], P_list[i]

            p_h, p_w = tf.shape(p)[1], tf.shape(p)[2]

            featuremap_height = tf.cast(p_h, tf.float32)
            featuremap_width = tf.cast(p_w, tf.float32)
            anchors = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[i],
                                                anchor_scales=cfgs.ANCHOR_SCALES,
                                                anchor_ratios=cfgs.ANCHOR_RATIOS,
                                                featuremap_height=featuremap_height,
                                                featuremap_width=featuremap_width,
                                                stride=cfgs.ANCHOR_STRIDE_LIST[i],
                                                name="make_anchors_for%s" % level_name)
            all_anchors.append(anchors)
        all_anchors = tf.concat(all_anchors, axis=0, name='all_anchors_of_FPN')

        # 4. postprocess rpn proposals. such as: decode, clip, NMS
        with tf.variable_scope('postprocess_FPN'):
            cascade_rois, roi_scores = postprocess_rpn_proposals(rpn_bbox_pred=fpn_box_pred,
                                                                 rpn_cls_prob=fpn_cls_prob,
                                                                 img_shape=img_shape,
                                                                 anchors=all_anchors,
                                                                 is_training=self.is_training)
            # rois shape [-1, 4]
            # +++++++++++++++++++++++++++++++++++++add img smry+++++++++++++++++++++++++++++++++++++++++++++++++++++++

            if self.is_training:
                score_gre_05 = tf.reshape(tf.where(tf.greater_equal(roi_scores, 0.5)), [-1])
                score_gre_05_rois = tf.gather(cascade_rois, score_gre_05)
                score_gre_05_score = tf.gather(roi_scores, score_gre_05)
                score_gre_05_in_img = show_box_in_tensor.draw_boxes_with_scores(img_batch=input_img_batch,
                                                                                boxes=score_gre_05_rois,
                                                                                scores=score_gre_05_score)
                tf.summary.image('score_greater_05_rois', score_gre_05_in_img)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        if self.is_training:
            with tf.variable_scope('sample_anchors_minibatch'):
                fpn_labels, fpn_bbox_targets = \
                    tf.py_func(
                        anchor_target_layer,
                        [gtboxes_batch, img_shape, all_anchors],
                        [tf.float32, tf.float32])
                fpn_bbox_targets = tf.reshape(fpn_bbox_targets, [-1, 4])
                fpn_labels = tf.to_int32(fpn_labels, name="to_int32")
                fpn_labels = tf.reshape(fpn_labels, [-1])
                self.add_anchor_img_smry(input_img_batch, all_anchors, fpn_labels)

            # --------------------------------------add smry-----------------------------------------------------------

            fpn_cls_category = tf.argmax(fpn_cls_prob, axis=1)
            kept_rpppn = tf.reshape(tf.where(tf.not_equal(fpn_labels, -1)), [-1])
            fpn_cls_category = tf.gather(fpn_cls_category, kept_rpppn)
            acc = tf.reduce_mean(tf.to_float(tf.equal(fpn_cls_category,
                                                      tf.to_int64(tf.gather(fpn_labels, kept_rpppn)))))
            tf.summary.scalar('ACC/fpn_accuracy', acc)

        # cascade rcnn
        total_loss_dict = {}
        fg_thresholds = [0.5, 0.6, 0.7]
        for i in range(len(fg_thresholds)-1):
            if self.is_training:
                cascade_rois, _, _, loss_dict = self.cascade_rcnn(cascade_rois, gtboxes_batch, input_img_batch,
                                                                  P_list,
                                                                  img_shape, fg_thresholds[i],
                                                                  fpn_box_pred, fpn_bbox_targets, fpn_cls_score, fpn_labels,
                                                                  stage=i + 1)

                for k in loss_dict.keys():
                    if k not in total_loss_dict.keys():
                        total_loss_dict[k] = loss_dict[k]
                    else:
                        total_loss_dict[k] += loss_dict[k]
            else:
                cascade_rois, _, _ = self.cascade_rcnn(cascade_rois, gtboxes_batch, input_img_batch,
                                                       P_list,
                                                       img_shape, fg_thresholds[i],
                                                       fpn_box_pred, None,
                                                       fpn_cls_score, None,
                                                       stage=i+1)
        if self.is_training:
            cascade_rois, bbox_pred, cls_prob, loss_dict = self.cascade_rcnn(cascade_rois, gtboxes_batch,
                                                                             input_img_batch,
                                                                             P_list,
                                                                             img_shape, fg_thresholds[-1],
                                                                             fpn_box_pred, fpn_bbox_targets,
                                                                             fpn_cls_score, fpn_labels,
                                                                             stage=len(fg_thresholds))

            final_bbox, final_scores, final_category = self.postprocess_fastrcnn(rois=cascade_rois,
                                                                                 bbox_ppred=bbox_pred,
                                                                                 scores=cls_prob,
                                                                                 img_shape=img_shape)

            for k in loss_dict.keys():
                if k not in total_loss_dict.keys():
                    total_loss_dict[k] = loss_dict[k]
                else:
                    total_loss_dict[k] += loss_dict[k]
            return final_bbox, final_scores, final_category, total_loss_dict
        else:
            cascade_rois, bbox_pred, cls_prob_stage3 = self.cascade_rcnn(cascade_rois, gtboxes_batch,
                                                                         input_img_batch,
                                                                         P_list,
                                                                         img_shape, fg_thresholds[-1],
                                                                         fpn_box_pred, None,
                                                                         fpn_cls_score, None,
                                                                         stage=len(fg_thresholds))
            cascade_rois_ = self.postprocess_cascadercnn(rois=cascade_rois,
                                                         bbox_pred=bbox_pred,
                                                         cls_score=cls_prob_stage3,
                                                         stage=len(fg_thresholds))

            with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane,
                                 slim.conv2d_transpose, slim.separable_conv2d, slim.fully_connected],
                                reuse=True):
                _, _, cls_prob_stage1 = self.cascade_rcnn(cascade_rois_, gtboxes_batch,
                                                          input_img_batch,
                                                          P_list,
                                                          img_shape, fg_thresholds[0],
                                                          fpn_box_pred, None,
                                                          fpn_cls_score, None,
                                                          stage=1)

                _, _, cls_prob_stage2 = self.cascade_rcnn(cascade_rois_, gtboxes_batch,
                                                          input_img_batch,
                                                          P_list,
                                                          img_shape, fg_thresholds[1],
                                                          fpn_box_pred, None,
                                                          fpn_cls_score, None,
                                                          stage=2)

            cls_prob = 0.8 * cls_prob_stage3 + 0.1 * cls_prob_stage2 + 0.1 * cls_prob_stage1

            final_bbox, final_scores, final_category = self.postprocess_fastrcnn(rois=cascade_rois,
                                                                                 bbox_ppred=bbox_pred,
                                                                                 scores=cls_prob,
                                                                                 img_shape=img_shape)

            return final_bbox, final_scores, final_category

    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()

            # for var in model_variables:
            #     print(var.name)
            # print(20*"__++__++__")

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith(self.base_network_name):
                    var_name_in_ckpt = name_in_ckpt_rpn(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20 * "___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients

