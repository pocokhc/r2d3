from keras.layers import *

class ImageModel():
    """ Abstract base class for all implemented ImageModel. """
    def create_image_model(self, c, enable_lstm):
        raise NotImplementedError()
    

class DQNImageModel(ImageModel):
    """ native dqn image model
    https://arxiv.org/abs/1312.5602
    """

    def create_image_model(self, c, enable_lstm):
        
        if enable_lstm:
            c = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same"), name="conv_1")(c)
            c = Activation("relu")(c)
            
            c = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same"), name="conv_2")(c)
            c = Activation("relu")(c)
            
            c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="same"), name="conv_3")(c)
            c = Activation("relu")(c)
            
            c = TimeDistributed(Flatten())(c)

        else:
                
            c = Conv2D(32, (8, 8), strides=(4, 4), padding="same", name="conv_1")(c)
            c = Activation("relu")(c)

            c = Conv2D(64, (4, 4), strides=(2, 2), padding="same", name="conv_2")(c)
            c = Activation("relu")(c)

            c = Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv_3")(c)
            c = Activation("relu")(c)

            c = Flatten()(c)

        return c


