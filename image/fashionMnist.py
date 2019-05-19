import tensorflow as tf
from    tensorflow import keras
from    tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32) / 255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y

(x,y) ,(x_test,y_test) = datasets.fashion_mnist.load_data()

print(x.shape,y.shape)

batch_size = 128

train_db = tf.data.Dataset.from_tensor_slices((x,y)).map(preprocess).shuffle(100000).batch(batch_size)

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).map(preprocess).shuffle(10000).batch(batch_size)


db_iter = iter(train_db)
sample = next(db_iter)

print(sample[0].shape,sample[1].shape)

model = Sequential([
    layers.Dense(256,activation=tf.nn.relu),
    layers.Dense(128,activation=tf.nn.relu),
    layers.Dense(64,activation=tf.nn.relu),
    layers.Dense(32,activation=tf.nn.relu),
    layers.Dense(10)
])

model.build(input_shape=[None,28*28])
model.summary()

optimizer = optimizers.Adam(lr=1e-3)


def main():
    for epoch in range(1000):
        for step,(x,y) in enumerate(train_db):

            x= tf.reshape(x,[-1,28*28])
            with tf.GradientTape() as tape:
                logits = model(x)

                y_onehot = tf.one_hot(y,depth=10)

                loss_MSE = tf.reduce_mean(tf.losses.MSE(y_onehot,logits))

                loss_CE = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True))

            grads = tape.gradient(loss_CE,model.trainable_variables)
            optimizer.apply_gradients(zip(grads,model.trainable_variables))


            if step % 100 == 0:
                print(epoch,step,'loss',float(loss_CE),float(loss_MSE))

        total_right = 0
        total_num = 0
        for x,y in db_test:
            x = tf.reshape(x, [-1, 28 * 28])
            # [b, 10]
            logits = model(x)
            # logits => prob, [b, 10]
            prob = tf.nn.softmax(logits, axis=1)
            # [b, 10] => [b], int64
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)
            # pred:[b]
            # y: [b]
            # correct: [b], True: equal, False: not equal
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype=tf.int32))

            total_right += int(correct)
            total_num += x.shape[0]

        acc = total_right / total_num
        print(epoch, 'test acc:', acc)



if __name__ == '__main__':
    main()