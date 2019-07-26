import os
import librosa
import numpy as np	
from keras.models import load_model

model=load_model('70_6_30.h5')
os.system('clear')
os.system('figlet Welcome to BE Project')
print("---------------------------------------------------")
os.system('figlet "Music Genre Classifier"')

while (1!=0):

	file=input('\n\nEnter the file name: ')
	if os.path.exists(file) == True:
		timeseries_length = 128
		data = np.zeros((1, timeseries_length, 33), dtype=np.float64)
		y, sr = librosa.load(file)
		mfcc = librosa.feature.mfcc(y=y, sr=sr,n_mfcc=13)
		spectral_center = librosa.feature.spectral_centroid(y=y, sr=sr)
		chroma = librosa.feature.chroma_stft(y=y, sr=sr)
		spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

		data[0, :, 0:13] = mfcc.T[0:timeseries_length, :]
		data[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
		data[0, :, 14:26] = chroma.T[0:timeseries_length, :]
		data[0, :, 26:33] = spectral_contrast.T[0:timeseries_length, :]

		y_prob = model.predict(data) 
		y_classes = y_prob.argmax(axis=-1)
		if y_classes[0]==0:
			print('Genre is Classical')
		elif y_classes[0]==1:
			print('Genre is Hip Hop')
		elif y_classes[0]==2:
			print('Genre is Jazz')
		elif y_classes[0]==3:
			print('Genre is Metal')
		elif y_classes[0]==4:
			print('Genre is Pop')
		elif y_classes[0]==5:
			print('Genre is Reggae')
	else:
		print('\n #Please enter correct file name #')
		continue

del model
gc.collect()
