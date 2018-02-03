import matplotlib.pyplot as plt
import tensorflow as tf
from Helper import Data
import numpy as np

bias_init = tf.zeros_initializer()
kern_init = tf.contrib.layers.xavier_initializer()

# noise should be of shape [None,4,4,16]

z_data = tf.placeholder(tf.float32,shape = [None,4,4,16])
x_data = tf.placeholder(tf.float32,shape = [None,28,28,1])

def generator(noise):
    with tf.variable_scope("generator") as scope:
        ls_layer_units = [64, 32, 16, 8, 4, 1]
        ls_layer_names = ["g_layer" + str(i + 1) for i in range(6)]
        layer = noise
        for i in range(len(ls_layer_units)):
            layer = tf.layers.conv2d_transpose(layer, ls_layer_units[i], [5, 5],
                                                strides=[1, 1], padding='valid',
                                                activation=tf.nn.relu,
                                                use_bias=True,
                                                kernel_initializer=kern_init,
                                                bias_initializer=bias_init,
                                                trainable=True,
                                                name=ls_layer_names[i],reuse=tf.AUTO_REUSE)
        return layer

# x_data should be of shape [None,28,28,1]


def discriminator(img_batch):
    with tf.variable_scope("discriminator") as scope:
        layer1 = tf.layers.conv2d(img_batch, 8, [5, 5], [1, 1], 'same',
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=kern_init,
                              bias_initializer=bias_init,
                              trainable=True, name="d_layer1",reuse=tf.AUTO_REUSE)
        layer2 = tf.layers.conv2d(layer1, 16, [5, 5], [1, 1], 'same',
                              activation=tf.nn.relu,
                              use_bias=True,
                              kernel_initializer=kern_init,
                              bias_initializer=bias_init,
                              trainable=True, name="d_layer2",reuse=tf.AUTO_REUSE)
        fc1 = tf.layers.dense(layer2, 32, activation=tf.nn.relu,
                          use_bias=True,
                          kernel_initializer=kern_init,
                          bias_initializer=bias_init,
                          trainable=True, name="d_fc1",reuse=tf.AUTO_REUSE)
        output = tf.layers.dense(fc1, 1, activation=tf.sigmoid,
                             use_bias=True,
                             kernel_initializer=kern_init,
                             bias_initializer=bias_init,
                             trainable=True, name="d_output",reuse=tf.AUTO_REUSE)


        return output
# 1 means real image
# 0 means fake image

# discriminator should distinguish between fake and real
# so for all images from generator it should give 0 output
#   and for all images from real data it should give 1 output

#generator goal is to produce realistic data
#   so it should force discriminator to produce 0 for all images it generates

epochs = 10

def train(x_data,z_data):
    gen_images = generator(z_data)
    disc_out_gen = discriminator(gen_images)
    disc_out_real = discriminator(x_data)

    loss_generator = tf.reduce_mean(tf.log(1-disc_out_gen))
    loss_discriminator = -(tf.reduce_mean(tf.log(disc_out_real)) + loss_generator)

    var_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    var_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

    train_disc = tf.train.AdamOptimizer(1e-2).minimize(loss_discriminator,var_list=var_disc)
    train_gen = tf.train.AdamOptimizer(1e-2).minimize(loss_generator,var_list=var_gen)


    train_data = Data()
    train_data.get_xdata("data/x_train.csv")
    train_data.get_ydata("data/y_train.csv")
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(epochs):
        train_data.get_rand_batch(32)
        x_batch = train_data.x_batch
        noise = np.random.normal(loc = 0.0,scale = 1.0,size = [32,4,4,16])
        feed_dict = {x_data: x_batch,z_data:noise}
        for _ in range(64):
            sess.run(train_disc,feed_dict)
        sess.run(train_gen,feed_dict)
        print("\rEpoch: ".format(epoch)+str(epoch),end = "")


    test_noise = np.random.normal(loc = 0.0,scale = 1.0,size = [1,4,4,16])
    img_out = sess.run(gen_images, feed_dict={z_data: test_noise})
    img_out = img_out.reshape(28,28)
    plt.imshow(img_out,cmap = "binary")
    plt.show()
    sess.close()

train(x_data,z_data)







