
import os
import time
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from IPython import display
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

def conv_block(num_filters,input):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(num_filters, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x

def decoder_block(inputs, skip_connection, filters):
  
  x = tf.keras.layers.Conv2DTranspose(filters,(2,2),
                                      strides=2,
                                      padding="same")(inputs)
  
  x = tf.keras.layers.Concatenate()([skip_connection,x])
  x = conv_block(filters,x)
  return x

def build_vgg_unet(input_shape):
  inputs = tf.keras.layers.Input(shape=(256,256,3))
  VGG = tf.keras.applications.VGG16(include_top=False,input_tensor=inputs)
  s1 = VGG.get_layer("block2_conv2").output #128
  s2 = VGG.get_layer("block3_conv3").output #64

  """Bridge"""
  bridge = VGG.get_layer("block4_conv3").output #32
  
  """Skip connections"""
  d1 = decoder_block(bridge,s2,256)  #d1 shape is  64,64,256
  d2 = decoder_block(d1,s1,128) #d2 shape is 128,128,128


  outputs = tf.keras.layers.Conv2DTranspose(3,(2,2),
                                      strides=2,
                                      padding="same",
                                      activation='tanh')(d2)
  

  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  return model

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer()
  result = tf.keras.models.Sequential()
  result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="same",
                             kernel_initializer=initializer, use_bias=False))
  
  result.add(tf.keras.layers.BatchNormalization())
  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.LeakyReLU())

  return result

OUTPUT_CHANNELS=3
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  l1_loss = tf.reduce_mean(tf.abs(target-gen_output))

  total_gen_loss =   gan_loss + (LAMBDA*l1_loss)
 
  return total_gen_loss,gan_loss,l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0.02)

  inp = tf.keras.layers.Input(shape=[256,256,3], name="input_image")
  tar = tf.keras.layers.Input(shape=[256,256,3],name="target_image")

  x = tf.keras.layers.Concatenate()([inp, tar])

  down1= downsample(64,4,False)(x)
  down2 = downsample(128, 4)(down1)
  down3 = downsample(256, 4)(down2)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)

  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)
  
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)
  
  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output),disc_real_output)
  gen_loss = loss_object(tf.zeros_like(generated_output), generated_output)

  total_loss = real_loss +  gen_loss

  return total_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

EPOCHS = 70



@tf.function
def train_step(generator,discriminator,input_image,
                target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as dsc_tape:
    gen_image = generator(input_image,training=True)
    disc_real_output = discriminator([input_image, target],training=True)

    disc_gen_output = discriminator([input_image, gen_image], training=True)
    gen_total_loss,gan_loss,l1_loss = generator_loss(disc_gen_output, gen_image,target)
    disc_loss = discriminator_loss(disc_real_output, gen_image)

  
  generator_gradients = gen_tape.gradient(gen_total_loss,generator.trainable_variables)
  discriminator_gradients = dsc_tape.gradient(disc_loss,discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
  return gen_total_loss, disc_loss
  
def generate_images(model,test_input, tar):
  gen_img = model(test_input,training=True)
  plt.figure(figsize=(15, 15))
  display_list = [test_input[0], tar[0], gen_img[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

def fit(generator,discriminator,train_ds,epochs,test_ds):
  curr_loss = math.inf
  try:
    os.mkdir("/root/min_loss_gen")
    os.mkdir("/root/test_results")
  except:
    pass

  for epoch in range(epochs):
    
    with tqdm(train_ds,unit=" batch") as tbatch:
      for input_image, target in tbatch:
        tbatch.set_description(f"Epochs {epoch} " )
        gen_total, disc_loss=train_step(generator, discriminator,
                                       input_image, target, epoch)
        tbatch.set_postfix(gen_loss=gen_total.numpy(),
                          disc_loss=disc_loss.numpy())

    gen_loss = []
    disc_loss = []
    count = 0
    for input_image, target in test_ds: 
      gen_image = generator(input_image,training=False)
      disc_real_output = discriminator([input_image, target],training=False)
      if count < 5:
        tf.keras.preprocessing.image.save_img(f'/root/test_results/img{count+1}_{epoch}.png',gen_image[0])
        count += 1


      disc_gen_output = discriminator([input_image, gen_image], training=False)
      gen_total_loss,gan_loss,l1_loss = generator_loss(disc_gen_output, gen_image,target)
      disc_loss_tot = discriminator_loss(disc_real_output, gen_image)
      gen_loss.append(gen_total_loss.numpy())
      disc_loss.append(disc_loss_tot.numpy())

    test_gen_loss = np.mean(np.array(gen_loss))
    print(f"Generator loss: {test_gen_loss}")

    if test_gen_loss < curr_loss:
      generator.save("/root/min_loss_gen")
      curr_loss = test_gen_loss





def load(namex,namey):
  
  input_image = tf.io.read_file(namex)
  real_image = tf.io.read_file(namey)

  input_image = tf.image.decode_jpeg(input_image)
  real_image  = tf.image.decode_jpeg(real_image)

  input_image = tf.cast(input_image,tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image

def resize(input_img, real_img, height, width):
  input_img = tf.image.resize(input_img,
                              [height, width],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  real_img = tf.image.resize(real_img,
                             [height, width],
                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
  
  
  return input_img, real_img

def random_crop(inp_img, real_img):
  stacked_image =  tf.stack([inp_img, real_img],axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(inp_img, real_img):
  inp_img = (inp_img/127.5) - 1
  real_img = (real_img/127.5) - 1

  return inp_img, real_img

@tf.function()
def random_jitter(input_image, real_image):
  input_image, real_image = resize(input_image, real_image, 286, 286)
  input_image, real_image = random_crop(input_image, real_image)
  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(image_filex, image_filey):
  input_image, real_image = load(image_filex, image_filey)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_filex, image_filey):
  input_image, real_image = load(image_filex, image_filey)
  input_image, real_image = resize(input_image, real_image, 
                                   IMG_HEIGHT,IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image


files = os.listdir("train_x")
train, test = train_test_split(files, test_size=0.2,random_state=42)
BUFFER_SIZE = len(train)


with open("trainx.txt","w") as f:
  f.write("\n".join(["/root/train_x/"+fno for fno in train]))

with open("trainy.txt","w") as f:
  f.write("\n".join(["/root/train_y/"+fno for fno in train]))

with open("testx.txt","w") as f:
  f.write("\n".join(["/root/train_x/"+fno for fno in test]))

with open("testy.txt","w") as f:
  f.write("\n".join(["/root/train_y/"+fno for fno in test]))

BATCH_SIZE = 256
IMG_WIDTH = 256
IMG_HEIGHT = 256

trainx = tf.data.TextLineDataset("trainx.txt")
trainy = tf.data.TextLineDataset("trainy.txt")
train_fname = tf.data.Dataset.zip((trainx,trainy))
train_fname = train_fname.shuffle(BUFFER_SIZE)

testx = tf.data.TextLineDataset("testx.txt")
testy = tf.data.TextLineDataset("testy.txt")
test_fname = tf.data.Dataset.zip((testx,testy))

train_dataset = train_fname.map(load_image_train,num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = test_fname.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator=build_vgg_unet((256,256,3))
discriminator=Discriminator()

fit(generator,discriminator,train_dataset, EPOCHS, test_dataset)


generator.save("/root/gen_model")
discriminator.save("/root/disc_model")

