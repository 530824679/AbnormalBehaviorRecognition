import os
import tensorflow as tf
from model import *
from scripts.dataset_builder import *

# File path paramters
tf.flags.DEFINE_string('root', "D:\\副业\\DMS\\Project", "The path of project")

# Training paramters
tf.flags.DEFINE_bool('continue_training', True, "Whether continue training using last weight")
tf.flags.DEFINE_float('weight_decay', 0.0005, "")
tf.flags.DEFINE_float('learning_rate', 0.1, "init learning rate")
tf.flags.DEFINE_integer('batch_size', 4, "The number of batch size")
tf.flags.DEFINE_integer('total_epochs', 200, "The total of epochs")
tf.flags.DEFINE_integer('image_width', 320, "number of image width")
tf.flags.DEFINE_integer('image_height', 240, "number of image height")
tf.flags.DEFINE_integer('image_channel', 3, "number of image channel")
tf.flags.DEFINE_integer('num_classes', 3, "number of classes")

# Model paramters
tf.flags.DEFINE_integer('reduction_ratio', 4, "reduction ratio of SE module")
tf.flags.DEFINE_integer('block', 3, "block of residual")
tf.flags.DEFINE_integer('cardinality', 8, "split number")
tf.flags.DEFINE_integer('depth', 64, "depth")

FLAGS = tf.flags.FLAGS

def main(_):
    # Control GPU resource utilization
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # Data File load
    dataset_path = os.path.join(FLAGS.root, 'data')
    train_data_list, train_label_list, test_data_list, test_label_list = prepare_data(dataset_path)
    data_size = len(train_data_list)

    # 320*240
    input = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
    label = tf.placeholder(tf.float32, shape=[None, FLAGS.num_classes])
    is_train = tf.placeholder(tf.bool)

    net = Network(FLAGS.num_classes, is_train, FLAGS.reduction_ratio, FLAGS.block, FLAGS.cardinality, FLAGS.depth)
    logits = net.build_network(input)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(cost + l2_loss * FLAGS.weight_decay)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session(config=config) as sess:
        # init all paramters
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)
        sess.run(tf.global_variables_initializer())

        # Restore weight file
        ckpt_path = os.path.join(FLAGS.root, 'checkpoints/model')
        if FLAGS.continue_training:
            ckpt = tf.train.get_checkpoint_state(ckpt_path)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
        else:
            sess.run(tf.global_variables_initializer())

        # Writer logs
        log_path = os.path.join(FLAGS.root, 'logs')
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)

        # Begin training
        for epoch in range(1, FLAGS.total_epochs + 1):
            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0
            for step in range(1, int(data_size/FLAGS.batch_size) + 1):
                if pre_index + FLAGS.batch_size < data_size:
                    batch_x = train_data_list[pre_index: pre_index + FLAGS.batch_size]
                    batch_y = train_label_list[pre_index: pre_index + FLAGS.batch_size]
                else:
                    batch_x = train_data_list[pre_index:]
                    batch_y = train_label_list[pre_index:]

                batch_x, batch_y = load_min_batch(batch_x, batch_y, FLAGS.image_height, FLAGS.image_width)

                train_feed_dict = {
                    input: batch_x,
                    label: batch_y,
                    is_train: True
                }
                _, batch_loss = sess.run([train_op, cost], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += FLAGS.batch_size

                train_loss /= data_size/FLAGS.batch_size    # average loss
                train_acc /= data_size/FLAGS.batch_size     # average accuracy

                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])

                summary_writer.add_summary(summary=train_summary, global_step=epoch)
                summary_writer.flush()

                print("epoch: %d/%d, train_loss: %.4f, train_acc: %.4f \n" % (epoch, FLAGS.total_epochs, train_loss, train_acc))

                # Create directories if needed
                if not os.path.isdir("checkpoints"):
                    os.makedirs("checkpoints")
                saver.save(sess, "%s/model.ckpt" % ("checkpoints"))
                sess.close()

if __name__ == '__main__':
    tf.app.run()