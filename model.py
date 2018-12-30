import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense,Dropout
from keras.optimizers import Adam,SGD,RMSprop
from GenreFeatureData import GenreFeatureData 

genre_features = GenreFeatureData()
# genre_features.load_preprocess_data()
genre_features.load_deserialize_data()

opt = Adam(lr=0.001)
batch_size = 25
nb_epochs = 300
model = Sequential()t
model.add(LSTM(units=128, dropout=0.05, recurrent_dropout=0.25, 
               return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.35, 
               return_sequences=False))
model.add(Dense(units=32))
model.add(Dropout(0.05))
model.add(Dense(units=6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()
model.fit(genre_features.train_X, genre_features.train_Y, batch_size=25, epochs=300,validation_data=(genre_features.dev_X,genre_features.dev_Y))
score, accuracy = model.evaluate(genre_features.dev_X, genre_features.dev_Y, batch_size=25, verbose=1)
print("Validation loss:  ", score)
print("Validation accuracy:  ", accuracy)

score, accuracy = model.evaluate(genre_features.test_X, genre_features.test_Y, batch_size=30)
print("Test loss:  ", score)
print("Test accuracy:  ", accuracy)

model.save('70_6_30.h5')
