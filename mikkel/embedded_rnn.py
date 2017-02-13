import pickle
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

from mikkel import data_parser

trainX = pickle.load(open("train_data_ann_representation.pickle", "rb"))
trainY = pickle.load(open("train_labels_one_hots.pickle", "rb"))
testX = pickle.load(open("test_data_ann_representation.pickle", "rb"))
testY = pickle.load(open("test_labels_one_hots.pickle", "rb"))

n_classes = 5
save_path = './keras_model_saves/model.h5'
train = True
save = True

trainX = pad_sequences(trainX, maxlen=100)
testX = pad_sequences(testX, maxlen=100)

if train:
    print("Building model...")
    model = Sequential()
    model.add(Embedding(50000, 128))
    model.add(LSTM(128, dropout_U=0.2))
    # model.add(Dense(512, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training model...")
    model.fit(trainX, trainY, validation_data=(testX, testY), nb_epoch=2)
    if save:
        model.save(save_path)
else:
    print("Loading model...")
    model = load_model(save_path)

print("Predicting stuff...")
res_pred = model.predict(testX)
res_pred = np.argmax(res_pred, axis=1)+1
res_true = np.argmax(testY, axis=1)+1
acc = np.sum(res_true == res_pred) / len(testY)
mse = ((res_true - res_pred) ** 2).mean()
print("Acc:", acc)
print("MSE:", mse)
data_parser.draw_heatmap(res_true, res_pred)


print("Finished!")
