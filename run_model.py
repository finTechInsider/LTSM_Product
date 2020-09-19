from pickle import load
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np 
from settings import extend_to_feature_length

def load_nn_model():
	# load the model
	model = load_model('model.h5')
	# load the mapping
	feature_mapping = load(open('feature_mapping.pkl', 'rb'))
	output_map = load(open('output_map.pkl', 'rb'))
	return model, feature_mapping, output_map

def predict(model, input_str, feature_mapping:dict):
	encoded = [feature_mapping[char] for char in input_str]
	encoded = extend_to_feature_length(encoded)
	encoded = to_categorical(encoded, num_classes=len(feature_mapping))
	encoded = encoded[None, :, :]
	# encoded = encoded.reshape(encoded.shape[1], encoded.shape[2])
	predictions = model.predict_classes(encoded, verbose=0)

	return list(predictions)[0]

if __name__ == "__main__":
	model, feature_mapping, output_map = load_nn_model()
	print(model.input_shape)

	feat_mapping = dict(map(reversed, feature_mapping.items()))
	output_map = dict(map(reversed, output_map.items()))

	print(feat_mapping)

	for s in ['US129393939_', '31203-a6b3829-2184848-22']:
		prediction_encoding = predict(model, s, feature_mapping)
		prediction = output_map[int(prediction_encoding)]
		print("{} is probably: {}".format(s, prediction))
