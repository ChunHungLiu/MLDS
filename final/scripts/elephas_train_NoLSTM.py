
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
from elephas import optimizers as elephas_optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
from pyspark import SparkContext, SparkConf

import Keras_model_deep_noLSTM as Akar
import vqa

batch_size = 200
nb_epoch = 10
verbose = 1
validation_split = 0.1
shuffle = True
show_accuracy = True

MODEL_ROOT = '../models/elephas/'
MODEL = 'elephas_noLSTM'
MODEL_FILE_NAME = MODEL_ROOT + MODEL + '.h5'
print 'Loading data...'

(X_train,Y_train),(X_test,Y_test) = vqa.load_data()

print 'Building model...'

model = Akar.keras_model(1)

#print 'Setting callbacks...'
#checkpointer = ModelCheckpoint(filepath=MODEL_ROOT+MODEL+".h5", verbose=1, save_best_only=False)
#early_stopping = EarlyStopping(monitor='val_acc', patience=5)
#print 'Start training...'
#model.fit( X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[checkpointer],validation_split=validation_split, shuffle=shuffle,show_accuracy=show_accuracy)

# Create Spark Context
conf = SparkConf().setAppName(MODEL)
sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, X_train, Y_train)

# Initialize SparkModel from Keras model and Spark Context

rmsprop = elephas_optimizers.RMSprop()

spark_model = SparkModel(sc,\
                        model,\
                        optimizer=rmsprop,\
                        frequency='epoch',\
                        mode='asynchronous',\
                        num_workers=3)

spark_model.train(rdd,\
                    nb_epoch=nb_epoch,\
                    batch_size=batch_size,\
                    verbose=2,\
                    validation_split=validation_split)

spark_model.get_network().save_weights(MODEL_FILE_NAME)

