import matplotlib.pyplot as plt
import tensorflow as tf
from Helper import Data
import numpy as np

bias_init = tf.zeros_initializer()
kern_init = tf.truncated_normal_initializer(stddev=0.02)

# noise should be of shape [None,4,4,4]

z_data = tf.placeholder(tf.float32,shape = [None,4,4,4])
x_data = tf.placeholder(tf.float32,shape = [None,28,28,1])

def generator(noise,reuse = False):
    with tf.variable_scope("generator"):
        if(reuse):
            tf.get_variable_scope().reuse_variables()
        ls_layer_units = [64, 32, 16, 8, 4, 1]
        ls_layer_names = ["g_layer" + str(i + 1) for i in range(6)]
        layer = noise
        for i in range(len(ls_layer_units)):
            layer = tf.layers.conv2d_transpose(layer, ls_layer_units[i], [5, 5],
                                                strides=[1, 1], padding='valid',
                                                activation=None,
                                                use_bias=False,
                                                kernel_initializer=kern_init,
                                                trainable=True,
                                                name=ls_layer_names[i])
            layer = tf.contrib.layers.batch_norm(inputs = layer,center = True,scale = True, is_training = True,scope = ls_layer_names[i]+"bn")
            layer = tf.nn.relu(layer)
        return layer

# x_data should be of shape [None,28,28,1]


def discriminator(img_batch,reuse = False):
    with tf.variable_scope("discriminator"):
        if(reuse):
            tf.get_variable_scope().reuse_variables()

        layer1 = tf.layers.conv2d(img_batch, 8, [5, 5], [1, 1], 'same',
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=kern_init,
                              trainable=True, name="d_layer1")
        layer1_pool = tf.layers.max_pooling2d(layer1,[2,2],strides = [1,1],padding = 'same')
        layer2 = tf.layers.conv2d(layer1_pool, 16, [5, 5], [1, 1], 'same',
                              activation=tf.nn.relu,
                              use_bias=False,
                              kernel_initializer=kern_init,
                              trainable=True, name="d_layer2")
        layer2_pool = tf.layers.max_pooling2d(layer2, [2, 2], strides=[1, 1], padding='same')
        flat_tensor = tf.layers.flatten(layer2_pool)
        fc1 = tf.layers.dense(flat_tensor, 32, activation=tf.nn.relu,
                          use_bias=True,
                          kernel_initializer=kern_init,
                          bias_initializer=bias_init,
                          trainable=True, name="d_fc1")
        output = tf.layers.dense(fc1, 1, activation=None,
                             use_bias=True,
                             kernel_initializer=kern_init,
                             bias_initializer=bias_init,
                             trainable=True, name="d_output")

        return output

# 1 means real image
# 0 means fake image

# discriminator should distinguish between fake and real
# so for all images from generator it should give 0 output
#   and for all images from real data it should give 1 output

#generator goal is to produce realistic data
#   so it should force discriminator to produce 0 for all images it generates

epochs = 500

def train(x_data,z_data):
    gen_images = generator(z_data)
    disc_out_gen = discriminator(gen_images)
    disc_out_real = discriminator(x_data,reuse = True)

    loss_generator = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_out_gen,labels = tf.ones_like(disc_out_gen)))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_out_real,labels=tf.ones_like(disc_out_real)))
    d_loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_out_gen,labels = tf.zeros_like(disc_out_gen)))
    loss_discriminator = d_loss_real + d_loss_gen

    var_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator")
    var_disc = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator")

    train_disc = tf.train.AdamOptimizer(1e-3).minimize(loss_discriminator,var_list=var_disc)
    train_gen = tf.train.AdamOptimizer(1e-3).minimize(loss_generator,var_list=var_gen)


    train_data = Data()
    train_data.get_xdata("data/x_train.csv")
    train_data.get_ydata("data/y_train.csv")

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for epoch in range(epochs):
        train_data.get_rand_batch(32)
        x_batch = train_data.x_batch
        noise = np.random.normal(loc = 0.0,scale = 0.02,size = [16,4,4,4])

        feed_dict_gen = {z_data:noise}
        feed_dict_disc = {x_data: x_batch,z_data: noise}
        dloss = sess.run(loss_discriminator, feed_dict_disc)
        gloss = sess.run(loss_generator, feed_dict_gen)
        print("Epoch: ", str(epoch), "\tDisc loss = ", dloss, "\tGen loss: ", gloss)
        while dloss < 1.1*gloss:
            sess.run(train_gen,feed_dict_gen)
            gloss = sess.run(loss_generator, feed_dict_gen)
        sess.run(train_disc,feed_dict_disc)

    sample_noise = np.random.normal(loc = 0.0, size = [1,4,4,4],scale=0.01)
    img = sess.run(gen_images,feed_dict={z_data: sample_noise})
    img = img.reshape(28,28)
    plt.imshow(img,cmap = "binary")
    plt.show()

    sess.close()

train(x_data,z_data)







