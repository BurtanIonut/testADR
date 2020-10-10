# something like
# model = load_model(path)

import numpy as np
import tensorflow as tf
from tensorflow import keras

tf.config.experimental.set_visible_devices([], 'GPU')

model_path = "models\cnn_fix1"
# Characters to be used
all_characters = [' ', '\t', '\n', '-', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                  'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'â', 'î', 'ă', 'ș', 'ț']
# Dictionary token to index
token_index = dict([(char, i) for i, char in enumerate(all_characters)])
# Dictionary index to token
reverse_token_index = dict((i, char) for i, char in enumerate(all_characters))
# Dictionary token to index
reverse_char_index = dict(
    (i, char) for char, i in token_index.items())

# Batch size for training
#batch_size = 32
# Max sentence length
max_len = 128
# Max number of samples
#max_samples = 137
# Number of characters
num_tokens = len(all_characters)  # Number of tokens

# LOAD MODEL
model = keras.models.load_model(model_path)


def predict_text(text):
    # PREPARE DATA
    test_texts = [text]

    # ENCODE TEST DATA
    encoder_test_data = np.zeros(
        (1, max_len, num_tokens), dtype="float32"
    )
    for i, test_text in enumerate(test_texts):
        for t, char in enumerate(test_text):
            encoder_test_data[i, t, token_index[char]] = 1.0
        encoder_test_data[i, t + 1:, token_index[" "]] = 1.0

    in_encoder = encoder_test_data
    in_decoder = np.zeros(
        (len(in_encoder), max_len, num_tokens),
        dtype='float32')

    in_decoder[:, 0, token_index["\t"]] = 1

    predict = np.zeros(
        (len(in_encoder), max_len),
        dtype='float32')

    for i in range(max_len - 1):
        predict = model.predict([in_encoder, in_decoder])
        predict = predict.argmax(axis=-1)
        predict_ = predict[:, i].ravel().tolist()
        for j, x in enumerate(predict_):
            in_decoder[j, i + 1, x] = 1

    for seq_index in range((len(in_encoder))):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        output_seq = predict[seq_index, :].ravel().tolist()
        decoded = []
        for x in output_seq:
            if reverse_char_index[x] == "\n":
                break
            else:
                decoded.append(reverse_char_index[x])
        decoded_sentence = "".join(decoded)

    return decoded_sentence
