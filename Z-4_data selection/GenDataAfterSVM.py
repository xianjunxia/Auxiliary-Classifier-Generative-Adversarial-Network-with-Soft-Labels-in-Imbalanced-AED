import numpy as np 
import scipy.io as sio
import h5py
'''
(object) rustling: 0,
(object) snapping: 1,
cupboard: 2,
cutlery: 3,
dishes: 4,
drawer: 5,
glass jingling:6,
object impact: 7,
people walking: 8,
washing dishes: 9,
water tap running: 10
'''
## Generate the 'brake squeaking' acoustic events 
### how many generated samples to be used, each sample corresponds to 20ms.
sample = 30000   ### the number of samples to be used after the SVM hyper-plane
path = '/data/users/21799506/Data/'
EventClass = 1
datapath = path + 'DCASE2016_Data/home/Evaluation/feat/GAN_After_SVM/GAN_AfterSVM_Class_1.mat'
Data = sio.loadmat(datapath)
mbe = Data['arr_0']
label = Data['arr_1']
generated_feat_file = path + 'DCASE2016_Data/home/Evaluation/feat/a111.wav_mon.npz'
np.savez(generated_feat_file, mbe[0:sample,:], label[0:sample,:])


path = '/data/users/21799506/Data/'
EventClass = 2
datapath = path + 'DCASE2016_Data/home/Evaluation/feat/GAN_After_SVM/GAN_AfterSVM_Class_2.mat'
Data = sio.loadmat(datapath)
mbe = Data['arr_0']
label = Data['arr_1']
generated_feat_file = path + 'DCASE2016_Data/home/Evaluation/feat/a222.wav_mon.npz'
np.savez(generated_feat_file, mbe[0:sample,:], label[0:sample,:])

 
path = '/data/users/21799506/Data/'
EventClass = 3
datapath = path + 'DCASE2016_Data/home/Evaluation/feat/GAN_After_SVM/GAN_AfterSVM_Class_3.mat'
Data = sio.loadmat(datapath)
mbe = Data['arr_0']
label = Data['arr_1']
generated_feat_file = path + 'DCASE2016_Data/home/Evaluation/feat/a333.wav_mon.npz'
np.savez(generated_feat_file, mbe[0:sample,:], label[0:sample,:])


path = '/data/users/21799506/Data/'
EventClass = 5
datapath = path + 'DCASE2016_Data/home/Evaluation/feat/GAN_After_SVM/GAN_AfterSVM_Class_5.mat'
Data = sio.loadmat(datapath)
mbe = Data['arr_0']
label = Data['arr_1']
generated_feat_file = path + 'DCASE2016_Data/home/Evaluation/feat/a555.wav_mon.npz'
np.savez(generated_feat_file, mbe[0:sample,:], label[0:sample,:])



path = '/data/users/21799506/Data/'
EventClass = 6
datapath = path + 'DCASE2016_Data/home/Evaluation/feat/GAN_After_SVM/GAN_AfterSVM_Class_6.mat'
Data = sio.loadmat(datapath)
mbe = Data['arr_0']
label = Data['arr_1']
generated_feat_file = path + 'DCASE2016_Data/home/Evaluation/feat/a666.wav_mon.npz'
np.savez(generated_feat_file, mbe[0:sample,:], label[0:sample,:])

