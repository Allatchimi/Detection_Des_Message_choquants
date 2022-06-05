import numpy as np
import tensorflow as tf


from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



app = Flask(__name__)
model = load_model('sar.h5')
modelAnglais = load_model('sar.h5')

embedding_dim = 128
max_length = 128
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
#training_size = 20000

top_k = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  lower=False,
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':

        choix_langue = str(request.form["choix-de-langue"])

        if choix_langue == "Anglais":
            int_features = str(request.form["interview_score"])
            int_features = [int_feature.split() for int_feature in int_features]
            tokenizer.fit_on_texts(int_features)
            pred_sequences = tokenizer.texts_to_sequences(int_features)
            pred_padded = pad_sequences(pred_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

            pred_LSTM = (modelAnglais.predict(pred_padded) > 0.5).astype("int32")


            if pred_LSTM[0] == 0:
                return render_template('index.html', prediction_text="This message is not offensive ")
            else:
                return render_template('index.html', prediction_text=" This message is offensive")
        else:
            int_features = str(request.form["interview_score"])
            int_features = [int_feature.split() for int_feature in int_features]
            tokenizer.fit_on_texts(int_features)
            pred_sequences = tokenizer.texts_to_sequences(int_features)
            pred_padded = pad_sequences(pred_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

            pred_LSTM = (model.predict(pred_padded) > 0.5).astype("int32")

            if pred_LSTM[0] == 0:
                return render_template('index.html', prediction_text = "ce message n'est pas choquant ")
            else:
                return render_template('index.html', prediction_text = " message choquant ")



if __name__ == '__main__':
    app.run()

