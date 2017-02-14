import pickle
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.cross_validation import train_test_split

from mikkel import data_parser

n_classes = 5
save_path = './keras_model_saves/model_best.h5'
train = True
save = True
load_training_set = True

if load_training_set:
    trainX = pickle.load(open("train_data_ann_representation_balanced.pickle", "rb"))
    trainY = pickle.load(open("train_labels_one_hots_balanced.pickle", "rb"))
    trainX, testX, trainY, testY = train_test_split(trainX, trainY,  test_size=0.2)

    trainX = pad_sequences(trainX, maxlen=100)
    testX = pad_sequences(testX, maxlen=100)

if train:
    print("Building model...")
    model = Sequential()
    model.add(Embedding(50000, 128))
    model.add(LSTM(128, dropout_U=0.2))
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
res_pred = np.argmax(res_pred, axis=1) + 1
res_true = np.argmax(testY, axis=1) + 1
acc = np.sum(res_true == res_pred) / len(testY)
mse = ((res_true - res_pred) ** 2).mean()
print("Acc:", acc)
print("MSE:", mse)
data_parser.draw_heatmap(res_true, res_pred)

my_bad_review = "This product was really shit! I hate everything about it. Do not buy it!"
my_good_review = "This have to be the best thing I've ever tried. The colours and performance are great. Fits my everyday use perfectly."

x_bad, _ = data_parser.generate_index_representation([my_bad_review], None)
x_good, _ = data_parser.generate_index_representation([my_good_review], None)

bad_result = model.predict(x_bad, batch_size=1)
good_result = model.predict(x_good, batch_size=1)

print(my_bad_review, "->", np.argmax(bad_result[0]) + 1, bad_result)
print(my_good_review, "->", np.argmax(good_result[0]) + 1, good_result)

print("Finished!")
