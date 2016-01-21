from keras.callbacks import ModelCheckpoint,EarlyStopping

import Keras_model_deep_noLSTM as Akar
from vqa import sent_load_data


# Parameters
batch_size = 200
nb_epoch = 20
verbose = 1
validation_split = 0.0
shuffle = True
show_accuracy = True

MODEL_ROOT = '../models/elephas/'
MODEL = 'sent_overfit_noLSTM_batch_{}'.format(batch_size)

MODEL_NAME = MODEL_ROOT + MODEL  + ".h5"

print 'Loading data...'
(X_train,Y_train) = sent_load_data()[0]

print 'Building model...'
model = Akar.keras_model(1)
#print 'Defining callbacks...'
#earlystopping = EarlyStopping(monitor='val_loss', patience=2, verbose=0)

print 'Start training...'

for epoch in [20,40,60,80,100,120,140,160,180,200]:
    model.fit( X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose, callbacks=[],validation_split=validation_split, shuffle=shuffle,show_accuracy=show_accuracy)
    model.save_weights(MODEL_ROOT+MODEL+"_{}.h5".format(epoch))
