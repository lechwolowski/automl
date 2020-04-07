from absl import app, logging, flags
import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
tf.disable_tensor_equality()
from det_model_fn import update_learning_rate_schedule_parameters, learning_rate_schedule, detection_loss, reg_l2_loss
import efficientdet_arch
import utils
import hparams_config
import dataloader
import os
os.environ["TF2_BEHAVIOR"] = "1"

flags.DEFINE_string(
    'training_file_pattern', '../../keras-yolo3/coco/train/*',
    'Glob for training data files (e.g., COCO train - minival set)')
flags.DEFINE_string(
    'validation_file_pattern', None,
    'Glob for evaluation tfrecords (e.g., COCO val2017 set)')
flags.DEFINE_integer('num_examples_per_epoch', 120000,
                     'Number of examples in one epoch')
flags.DEFINE_integer('num_epochs', 15, 'Number of epochs for training')
flags.DEFINE_string('model_dir', './test', 'Location of model_dir')
flags.DEFINE_string('backbone_ckpt', '',
                    'Location of the Efficientdet checkpoint to use for model '
                    'initialization.')
flags.DEFINE_integer('train_batch_size', 2, 'training batch size')
flags.DEFINE_integer('eval_batch_size', 1, 'evaluation batch size')
flags.DEFINE_string('ckpt', None,
                    'Start training from this EfficientDet checkpoint.')
flags.DEFINE_string('model_name', 'efficientdet-d0', 'Number of epochs for training')
flags.DEFINE_integer('input_size', 512, 'Number of classes.')
flags.DEFINE_string('hparams', '',
                    'Comma separated k=v pairs of hyperparameters.')
flags.DEFINE_bool('use_fake_data', False, 'Use fake input.')

FLAGS = flags.FLAGS


def main(_):
    config = hparams_config.get_detection_config(FLAGS.model_name)
    config.override(FLAGS.hparams)
    train_summary_writer = tf2.summary.create_file_writer(
            FLAGS.model_dir)
    params = dict(
        config.as_dict(),
        model_name=FLAGS.model_name,
        num_epochs=FLAGS.num_epochs,
        model_dir=FLAGS.model_dir,
        backbone_ckpt=FLAGS.backbone_ckpt,
        num_examples_per_epoch=FLAGS.num_examples_per_epoch,
        ckpt=FLAGS.ckpt,
        use_tpu=False
    )
    strategy = tf.distribute.MirroredStrategy()

    params.update(batch_size=FLAGS.train_batch_size)
    dataset = dataloader.InputReader(FLAGS.training_file_pattern,
                           is_training=True, use_fake_data=FLAGS.use_fake_data)(params)

    dataset = strategy.experimental_distribute_dataset(dataset)

    with strategy.scope():
        moving_average_decay = params['moving_average_decay']
        if moving_average_decay:
            ema = tf.train.ExponentialMovingAverage(decay=moving_average_decay)
            ema_vars = utils.get_ema_vars()

        # optimizer = tf.keras.optimizers.SGD(params['learning_rate'], momentum=params['momentum'])
        optimizer = tf.keras.optimizers.Adam(params['learning_rate'])

        @tf.function(autograph=False)
        def train_step(features, labels, global_step):
            update_learning_rate_schedule_parameters(params)
            with tf.GradientTape() as tape:
                cls_outputs, box_outputs = efficientdet_arch.efficientdet(features, FLAGS.model_name)
                var_list = tf.trainable_variables()
                tape.watch(var_list)
                learning_rate = learning_rate_schedule(params, global_step)
                optimizer._set_hyper('learning_rate', learning_rate)
                tf.print(optimizer.lr)
                det_loss, cls_loss, box_loss = detection_loss(cls_outputs, box_outputs,
                                                              labels, params)
                l2loss = reg_l2_loss(params['weight_decay'])
                total_loss = det_loss + l2loss

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            if params.get('clip_gradients_norm', 0) > 0:
                logging.info('clip gradients norm by %f', params['clip_gradients_norm'])
                gradients = tape.gradient(total_loss, var_list)
                grads_and_vars = zip(gradients, var_list)
                with tf.name_scope('clip'):
                    grads = [gv[0] for gv in grads_and_vars]
                    tvars = [gv[1] for gv in grads_and_vars]
                    clipped_grads, gnorm = tf.clip_by_global_norm(
                        grads, params['clip_gradients_norm'])
                    grads_and_vars = list(zip(clipped_grads, tvars))

                with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(grads_and_vars)
            else:
                gradients = tape.gradient(total_loss, var_list)
                grads_and_vars = zip(gradients, var_list)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.apply_gradients(grads_and_vars)

            if moving_average_decay:
                with tf.control_dependencies([train_op]):
                    ema.apply(ema_vars)
            return learning_rate, cls_loss, box_loss, det_loss, l2loss, total_loss, gnorm

        dataset_iter = iter(dataset)
        for i in range(params['num_epochs']):
            for j in range(params['num_examples_per_epoch']):
                global_step = i * params['num_examples_per_epoch'] + j
                features, labels = next(dataset_iter)
                learning_rate, cls_loss, box_loss, det_loss, l2loss, total_loss, gnorm = strategy.experimental_run_v2(train_step, args=[features, labels, tf.convert_to_tensor(global_step)])

                cls_loss += strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, cls_loss, axis=None)
                box_loss += strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, box_loss, axis=None)
                det_loss += strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, det_loss, axis=None)
                l2loss += strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, l2loss, axis=None)
                total_loss += strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, total_loss, axis=None)
                with train_summary_writer.as_default():
                    tf2.summary.scalar('lrn_rate', learning_rate, step=global_step)
                    tf2.summary.scalar('trainloss/cls_loss', cls_loss, step=global_step)
                    tf2.summary.scalar('trainloss/box_loss', box_loss, step=global_step)
                    tf2.summary.scalar('trainloss/det_loss', det_loss, step=global_step)
                    tf2.summary.scalar('trainloss/l2_loss', l2loss, step=global_step)
                    tf2.summary.scalar('trainloss/loss', total_loss, step=global_step)
                    tf2.summary.scalar('gnorm', gnorm, step=global_step)


if __name__ == '__main__':
    app.run(main)