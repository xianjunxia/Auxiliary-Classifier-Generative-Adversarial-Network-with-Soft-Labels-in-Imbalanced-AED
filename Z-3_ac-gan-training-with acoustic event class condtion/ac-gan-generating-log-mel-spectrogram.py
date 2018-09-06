from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
#from PIL import Image

from six.moves import range

import keras.backend as K
import tensorflow as tf
import scipy.io as sio
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.noise import GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.backend.common import _EPSILON
from keras.utils.generic_utils import Progbar
import numpy as np
import os
from keras.backend.tensorflow_backend import set_session
np.random.seed(1331)

K.set_image_dim_ordering('th')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# GPU 1 usage

GPU = "1"
# use specific GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))



seq = 40
Classnum = 11  ### In toal, 11 acoustic event class on the TUT Sound Event 2016 under the home environment
def GetTrainData(GenerateType):
    Data = sio.loadmat('Class0-11.mat')
    feat = Data['feat']
    label= Data['label']
    TestNum = 10
    cnt = np.zeros((Classnum))
    for ind  in range (Classnum):
        if(ind != GenerateType):
            continue
        for i in range(label.shape[0]):
            a = np.sum(label[i,:])    
            if((label[i,ind] == 1) & (a == 1)):
                cnt[ind] = cnt[ind]+ 1
    Dat   = np.zeros((int(np.sum(cnt)),40)) 
    Lab   = np.zeros((int(np.sum(cnt))))
    print(cnt)
    
    count = 0
    
    for ind in range(Classnum):
        if (ind != GenerateType):
            continue
        for i in range(label.shape[0]):         
            a = np.sum(label[i,:])    
            if((label[i,ind] == 1) & (a == 1)):
                Dat[count,:] = feat[i,:]
                Lab[count] = ind
                count = count + 1
    ######################################################## samples with overlapping
    '''
    step = 40
    s = int(int(cnt[GenerateType] - seq)/np.float(step))
    print(s)
    X_Dat = np.zeros((s,seq,40))
    y_Dat = np.zeros((s))
    j = seq/2   
    i = 0
    while (j < s-seq/2):
        X_Dat[i,:,:] = Dat[j-seq/2:j+seq/2,:]
        y_Dat[i] = GenerateType
        i = i + 1
        j = j + step
    X_Dat = np.expand_dims(X_Dat, axis=1)
    return X_Dat,y_Dat
    #########################################################
    '''
    s = 0
    for ind in range(Classnum):
        if (ind != GenerateType):
            continue
        s = s + int(cnt[ind]/np.float(seq))
    print(s)
    X_Dat = np.zeros((s,seq,40))
    y_Dat = np.zeros((s))
    s = 0
    for ind in range(Classnum):
        if(ind != GenerateType):
            continue
        for i in range(int(cnt[ind]/np.float(seq))):
            pos = 0
            for index in range(ind):
                pos = pos + cnt[index]
            X_Dat[s,:,:] = Dat[int(pos+i*seq):int(pos+(i+1)*seq),:]
            y_Dat[s] = ind
            s = s + 1
    X_Dat = np.expand_dims(X_Dat, axis=1)
    return X_Dat,y_Dat
    

def modified_binary_crossentropy(target, output):
    #output = K.clip(output, _EPSILON, 1.0 - _EPSILON)
    return -(target * output + (1.0 - target) * (1.0 - output))
    #return K.mean(target*output)

def build_generator(latent_size):
    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size))
    cnn.add(LeakyReLU())
    cnn.add(Dense(128 * (seq/4) * 10))
    cnn.add(LeakyReLU())
    cnn.add(Reshape((128, seq/4, 10)))

    # upsample to (..., 40, 40)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, (5, 5), padding='same',
                          kernel_initializer='glorot_uniform'))
    cnn.add(LeakyReLU())

    # upsample to (..., 40, 40)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, (5, 5), padding='same',
                          kernel_initializer='glorot_uniform'))
    cnn.add(LeakyReLU())

    # take a channel axis reduction
    cnn.add(Convolution2D(1, (2, 2), padding='same',
                          activation='tanh', kernel_initializer='glorot_uniform'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(Classnum, latent_size,
                              init='glorot_uniform')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)
    #cnn.summary()

    return Model(input=[latent, image_class], output=fake_image)

def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
    #cnn.add(GaussianNoise(0.2, input_shape=(1, seq, 40)))
    cnn.add(Convolution2D(32, (2, 2), padding='same', strides=(2, 2),
                          input_shape=(1, seq, 40)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    cnn.add(Convolution2D(64, (2, 2), padding='same', strides=(1, 1)))
    cnn.add(LeakyReLU())
    cnn.add(Dropout(0.3))

    #cnn.add(Convolution2D(128, (3, 3), padding='same', strides=(2, 2)))
    #cnn.add(LeakyReLU())
    #cnn.add(Dropout(0.3))

    #cnn.add(Convolution2D(256, (3, 3), padding='same', strides=(1, 1)))
    #cnn.add(LeakyReLU())
    #cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(1, seq, 40))

    features = cnn(image)

    fake = Dense(1, activation='linear', name='generation')(features)
    aux = Dense(Classnum, activation='softmax', name='auxiliary')(features)

    cnn.summary()
    exit()

    return Model(input=image, output=[fake, aux])

if __name__ == '__main__':

    # batch and latent size taken from the paper
    nb_epochs = 2000
    latent_size = 100    
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # build the discriminator
    for GenerateType  in [1,2,3,5,6]:   ### 1,2,3,5,6 denote the object snapping, cupboard, cutlery, drawer and glass jingling respectively.
        X_train,y_train = GetTrainData(GenerateType=GenerateType)
        mi = np.min(X_train)
        ma = np.max(X_train)
        X_train = (X_train.astype(np.float32) - mi) /(ma-mi)
        print(X_train.shape)
        print(y_train.shape)
        
        batch_size = 10
        discriminator = build_discriminator()
        discriminator.compile(
            optimizer=SGD(clipvalue=0.01),#Adam(lr=adam_lr, beta_1=adam_beta_1),
            loss=[modified_binary_crossentropy, 'sparse_categorical_crossentropy']
        )
        discriminator.summary()
    
        # build the generator
        generator = build_generator(latent_size)
        generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                          loss='binary_crossentropy')
        #generator.summary()
        latent = Input(shape=(latent_size, ))
        image_class = Input(shape=(1,), dtype='int32')
    
        # get a fake image
        fake = generator([latent, image_class])
    
        # we only want to be able to train generation for the combined model
        discriminator.trainable = False
        fake, aux = discriminator(fake)
        combined = Model(input=[latent, image_class], output=[fake,aux])
    
        combined.compile(
            optimizer='RMSprop',
            loss=[modified_binary_crossentropy, 'sparse_categorical_crossentropy']        
        )        
        
        nb_train = X_train.shape[0]
    
        train_history = defaultdict(list)
        test_history = defaultdict(list)
    
        for epoch in range(nb_epochs):
            print('Epoch {} of {}'.format(epoch + 1, nb_epochs))
    
            nb_batches = int(X_train.shape[0] / batch_size)
            progress_bar = Progbar(target=nb_batches)
    
            epoch_gen_loss = []
            epoch_disc_loss = []
    
            for index in range(nb_batches):
                if len(epoch_gen_loss) + len(epoch_disc_loss) > 1:
                    progress_bar.update(index, values=[('disc_loss',np.mean(np.array(epoch_disc_loss),axis=0)[0]), ('gen_loss', np.mean(np.array(epoch_gen_loss),axis=0)[0])])
                else:
                    progress_bar.update(index)
                # generate a new batch of noise
                #noise = np.random.uniform(-1, 1, (batch_size, latent_size))
                noise = np.random.normal(0, 1, (batch_size, latent_size))
    
                # get a batch of real images
                image_batch = X_train[index * batch_size:(index + 1) * batch_size]
                label_batch = y_train[index * batch_size:(index + 1) * batch_size]
                #print(image_batch[3,0,:,:])
                # sample some labels from p_c
                #sampled_labels = np.random.randint(0, 0, batch_size)
                sampled_labels = np.zeros((batch_size)) 
                sampled_labels = sampled_labels.astype(np.int) + GenerateType           
    
    
                generated_images = generator.predict(
                    [noise, sampled_labels.reshape((-1, 1))], verbose=0)
    
                X = np.concatenate((image_batch, generated_images))
    
                y = np.array([-1] * batch_size + [1] * batch_size)          
                aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
    
                # see if the discriminator can figure itself out...
                epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))
    
                # make new noise. we generate 2 * batch size here such that we have
                # the generator optimize over an identical number of images as the
                # discriminator                       
                noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
                #sampled_labels = np.random.randint(0, 6, 2 * batch_size)
                sampled_labels = np.zeros((2*batch_size))
                sampled_labels = sampled_labels.astype(np.int) + GenerateType
                # we want to train the genrator to trick the discriminator
                # For the generator, we want all the {fake, not-fake} labels to say
                # not-fake
                trick = -np.ones(2 * batch_size)
    
                epoch_gen_loss.append(combined.train_on_batch(
                    [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))
        
            discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)
            generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)
    
            # generate an epoch report on performance
            train_history['generator'].append(generator_train_loss)
            train_history['discriminator'].append(discriminator_train_loss)
    
            print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
                'component', *discriminator.metrics_names))
            print('-' * 65)
    
            ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
            print(ROW_FMT.format('generator (train)',*train_history['generator'][-1]))
            print(ROW_FMT.format('discriminator (train)',*train_history['discriminator'][-1]))
            
    
            # generate some digits to display
            if(epoch % 100 == 0):
                name = 'GAN_Class_' + str(GenerateType) + '/GAN_Class_' + str(GenerateType) + '_generator.hdf5'
                print(name)
                generator.save_weights(name)
                name = 'GAN_Class_' + str(GenerateType) + '/GAN_Class_' + str(GenerateType) + '_discriminator.hdf5'
                discriminator.save_weights(name)
                noise = np.random.normal(0, 1, (10000, latent_size))
                for ind in range(11):                    
                    if(ind != GenerateType):
                        continue
                    sampled_labels = np.zeros((10000))
                    sampled_labels = sampled_labels.astype(np.int) + GenerateType
                    generated_images = generator.predict([noise, sampled_labels], verbose=0)        
                    generated_images = generated_images.reshape(generated_images.shape[0],generated_images.shape[2], generated_images.shape[3])
                    generated_images = generated_images.astype(np.float32) * (ma- mi) + mi                    
                    name = 'GAN_Class_'  + str(ind) + '/GAN_Class_' + '_epoch_' + str(epoch)
                    label = np.zeros((10000, seq, 11),dtype=int)
                    label[:,:,ind] = 1
                    sio.savemat(name,{'arr_0':generated_images,'arr_1':label})
        name = 'GAN_Class_' + str(GenerateType) + '/acgan-history.pkl'
        pickle.dump({'train': train_history, 'test': test_history},
                    open(name, 'wb'))
        pickle.dump({'train': train_history, 'test': test_history},
                    open('acgan-history.pkl', 'wb'))
        
