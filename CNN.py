import tensorflow as tf
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.models import Model
import numpy as np
from keras.applications.vgg19 import preprocess_input

import matplotlib.pyplot as plt

class CNN:

    model = None

    def __init__(self,model_path):

        inputs = Input(shape=(224,224,3))
        # block1
        X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block1_conv1')(inputs)
        X = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block1_conv2')(X)
        X = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block1_pool')(X)

        # block2
        X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='block2_conv1')(X)
        X = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='block2_conv2')(X)
        X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(X)

        # block3
        X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block3_conv1')(X)
        X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block3_conv2')(X)
        X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block3_conv3')(X)
        X = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block3_conv4')(X)
        X = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block3_pool')(X)

        # # block4
        # X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='block4_conv1')(X)
        # X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='block4_conv2')(X)
        # X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='block4_conv3')(X)
        # X = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', name='block4_conv4')(X)
        # X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(X)
        #
        # # block5
        # # 32,32,512 -> 32,32,512
        # X = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block5_conv1')(X)
        # X = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block5_conv2')(X)
        # X = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block5_conv3')(X)
        # X = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same', name='block5_conv4')(X)
        # X = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool')(X)

        self.model=Model(inputs=inputs, outputs=X)
        self.model.load_weights(model_path, by_name=True)

        print(self.model.summary())

    def extract(self,pic_path):
        #读取指定图像数据
        img = image.load_img(pic_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        #利用第一个模型预测出特征数据，并对特征数据进行切片
        # print(type(x))
        feature_map = self.model.predict(x)
        T = np.array(feature_map)
        # print(T.shape)
        return T[0,:,:,:]


    def show_extract_pic(self,T):
        print(T.shape)
        # plt.imshow(T[:,:,:], cmap=plt.get_cmap('brg'))
        # for i in range(3):
        #     plt.subplot(1,3,i+1)
        #     plt.imshow(T[:,:,i], cmap=plt.get_cmap('gray'))

        # plt.show()
        # n = T.shape[2]
        # for iter in range(n//4):
        #     for i in range(4):
        #         plt.subplot(2,2,i+1)
        #         plt.imshow(T[:,:,iter*4+i], cmap=plt.get_cmap('gray'))
        #
        #     plt.show()

if __name__ == '__main__':
    cnn = CNN("./mmm/vgg19_weights_tf_dim_ordering_tf_kernels.h5")
    image = cnn.extract("./pic/1539109346908_final.jpg")

    img1 = np.mean(image[:,:,:86],2);
    img2 = np.mean(image[:,:,86:171],2);
    img3 = np.mean(image[:,:,171:],2);

    cnn.show_extract_pic(np.stack((img1,img2,img3),axis=2))
    # print(cnn.extract("./pic/1539109346908_final.jpg"))