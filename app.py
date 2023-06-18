import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenisasi():
    data = pd.read_csv('depresi (3).csv')
    data.text = data.text.astype(str)
    text = data['text']
    tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    #print("berikut adalah tokenisasi per kata -> ",word_index)

    return tokenizer

token = tokenisasi()

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

@st.cache_resource
def load_models():
    """
    @st.cache_resource decorator digunakan untuk menyimpan resource model.

    Fungsi load_models() akan membuat model FCDUG dan menerapkan weights dari file .h5 

    """

    # model_LSTM = tf.keras.Sequential([
    # tf.keras.layers.Embedding(input_dim=10000, output_dim=40),
    # tf.keras.layers.LSTM(64),
    # tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(2, activation='softmax')
    # ])
    
    embed_dim = 32  # Embedding size for each token
    num_heads = 6  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer
    maxlen = 200
    vocab_size = 10000

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.Dropout(0.1)(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model_TRANSFORMERS = keras.Model(inputs=inputs, outputs=outputs)

    model_TRANSFORMERS.load_weights("deploydepression.h5")

    return model_TRANSFORMERS

model_TRANSFORMERS = load_models()


def preprocess_text(sentence):
    sentence = [sentence]
    print(sentence)

    # Preprocessing
    # melakukan tokenisasi
    #tokenizer.fit_on_texts(sentence)

    sequences = token.texts_to_sequences(sentence)
    #sequences = tokenizer.fit_on_texts(sentence)
    print(sequences)
    padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')
    print("Padded in preprocess ->", padded)

    return padded

def predict(text_predict):
    """
    @st.cache_data decorator berfungsi untuk caching / menyimpan data prediksi sementara

    Fungsi predict digunakan untuk melakukan prediksi data
    """
    padded = preprocess_text(text_predict)
    print(padded)
    prediction = int(np.argmax(model_TRANSFORMERS.predict(padded)))

    print("Prediction -> ", prediction)
    if prediction == 0:
        prediction = "Tidak Depresi"
    else:
        prediction = "Depresi"
    
    return prediction

def main():
    st.title("Detecting Depression")
    st.subheader("Architecture used -> 6attnheadTransformer+Augmentasi+LayerNorm+TripleDropout")

    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        with st.form(key='nlpForm'):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

            col1  = st.columns(1)
            if submit_button:
                
                result = predict(raw_text)
                st.info("Results")
                st.write(result)
    
    else:
        st.subheader("About")

if __name__=='__main__':
    main()