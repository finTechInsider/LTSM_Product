import pandas as pd
import numpy as np
from pickle import dump
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, SpatialDropout1D, Embedding
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from settings import FEATURES, extend_to_feature_length

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def load_test_data(file:str = 'input_data.txt'): 
	raw_text = load_doc(file)
	lines = raw_text.split('\n')

	# integer encode sequences of characters
	unique_chars = sorted(list(set(raw_text + 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-!@#$%^&*() ')))
	# unique_chars = unique_chars.append([])
	print(unique_chars)
	return lines, unique_chars

def create_feature_mapping(chars: list):
	feature_mapping = dict((c, i) for i, c in enumerate(chars))
	dump(feature_mapping, open('feature_mapping.pkl', 'wb'))
	return feature_mapping

def create_output_mapping(values:list):
	output_map = dict((c, i) for i, c in enumerate(values))
	dump(output_map, open('output_map.pkl', 'wb'))
	return output_map

def encode_input(lines:list, feature_mapping:list, output_map:dict):
	sequences = list()
	outputs = list()

	num_input_rows = len(lines)
	print("There are {} rows of input data".format(num_input_rows))
	for line in lines:
		tokens = line.split(',')
		
		if len(tokens) != 2:
			num_input_rows -= 1
			print(tokens)	
			continue
		else:
			encoded_seq = np.array([feature_mapping[char] for char in tokens[0]])
			encoded_seq = extend_to_feature_length(encoded_seq)
			sequences.append(encoded_seq)
			outputs.append(output_map[tokens[1]])

	new_output = np.concatenate(sequences).reshape((num_input_rows,len(encoded_seq)))

	x = new_output
	y = outputs
	return x, y

def wrapped_to_categorical(input, num_classes):
    # print(input)
    return to_categorical(input, num_classes)

def get_x_y_data(encoded_data:np.array, y, num_features:int, num_classes:int):
	X = encoded_data
	sequences = [wrapped_to_categorical(x, num_classes=num_features) for x in X]
	X = np.array(sequences)
	print('Input Shape: {}'.format(X.shape))

	y = to_categorical(y, num_classes=num_classes)
	return X, y

def train_model(X:np.array, y:np.array, num_classes:int):
	callback = EarlyStopping(monitor='loss', patience=6)

    # # define model
	model = Sequential()
	model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dense(num_classes, activation='softmax'))
	print(model.summary())
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])#, callbacks=[callback])
	# fit model
	model.fit(X, y, epochs=50, verbose=2)

	# save the model to file
	model.save('model.h5')
	# save the mapping
	return model

if __name__ == "__main__":
	df = pd.read_csv('input_data.txt')
	df.columns = ['desc','product'] 
	lines, chars = load_test_data()
	feature_mapping = create_feature_mapping(chars)
	output_map = create_output_mapping(list(df['product'].unique()))
	x, y = encode_input(lines, feature_mapping, output_map)

	# vocabulary size
	vocab_size = len(feature_mapping)
	print('Vocabulary Size: %d' % vocab_size)

	X,y = get_x_y_data(x, y, vocab_size, num_classes=len(output_map))
	model = train_model(X, y, num_classes=len(output_map))

	# TEST
	from run_model import predict
	
	for s in ['US129393939_', 'ffe54a24-ea7b-11ea-ab1b-1094bbdb9afe']:
		prediction_encoding = predict(model, s, feature_mapping)
		prediction = dict(map(reversed, output_map.items()))[int(prediction_encoding)]
		print("{} is an {}".format(s, prediction))
