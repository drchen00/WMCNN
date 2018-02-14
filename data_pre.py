import tensorflow as tf


#  tf1.4以后的data库太僵硬，
#  对于复杂数据的处理太难用且不直观，
#  如果将来版本底层接口被废弃请重写该接口
def get_data(filenames,
             data_len,
             class_num,
             shuffled,
             slice_len=0,
             batch_size=1):
    file_queue = tf.train.string_input_producer(filenames)
    reader = tf.TextLineReader()
    _, value = reader.read(file_queue)
    record_defaults = [[1]] + [[1.0]] * data_len
    data = tf.decode_csv(value, record_defaults)

    #      数据切片
    channel = 1 if slice_len == 0 or slice_len > data_len else data_len - slice_len + 1
    real_len = data_len if slice_len == 0 else min(data_len, slice_len)
    label = data[0] - 1
    series = []
    for i in range(1, 1 + channel):
        series += [tf.stack([t for t in data[i:i + real_len]])]
    series = tf.stack(series)
    series = tf.transpose(series)

    #  对于大数据集可以使用多线程输入具体请查询API
    if shuffled:
        queue = tf.RandomShuffleQueue(16 * batch_size, 4 * batch_size,
                                      [tf.int32, tf.float32],
                                      [[1], [real_len, channel]])
    else:
        queue = tf.FIFOQueue(16 * batch_size, [tf.int32, tf.float32],
                             [[1], [real_len, channel]])

    enqueue = queue.enqueue([[label], series])
    tf.train.add_queue_runner(tf.train.QueueRunner(queue, [enqueue]))

    labels, datas = queue.dequeue_many(batch_size)
    labels = tf.one_hot(tf.reshape(labels, [-1]), class_num)

    assert len(datas.get_shape()) == 3
    assert datas.get_shape()[0] == batch_size
    assert datas.get_shape()[-1] == channel
    assert len(labels.get_shape()) == 2
    assert labels.get_shape()[0] == batch_size
    assert labels.get_shape()[1] == class_num

    return datas, labels
