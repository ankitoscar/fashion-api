import numpy as np 
import pandas as pd
import pickle as pkl 

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input

class ImageRecommender():

    def __init__(self):
        self.img_width, self.img_height, self._ = 224, 224, 3
        self.map_embeddings = pd.read_csv('map_embeddings.csv')
        images = pd.read_csv('images.csv')
        self.images = images['0']
        self.df = pd.read_csv('images_df.csv')
        MODEL_PATH = 'embedding_model.h5'
        self.model = keras.models.load_model(MODEL_PATH)

    def find_cosine_similarity(self, img_emb):
        cos_sim = np.dot(self.map_embeddings, img_emb)/(np.linalg.norm(self.map_embeddings)*np.linalg.norm(img_emb))
        return 1-cos_sim

    def get_custom_embedding(self, img_path):
        # Reshape
        img = image.load_img(img_path, target_size=(self.img_width, self.img_height))
        # img to Array
        x   = image.img_to_array(img)
        # Expand Dim (1, w, h)
        x   = np.expand_dims(x, axis=0)
        # Pre process Input
        x   = preprocess_input(x)
        return self.model.predict(x).reshape(-1)

    def get_similar_images(self, img_path):
        emb = self.get_custom_embedding(img_path)
        most_similar = np.argsort(self.find_cosine_similarity(emb))[:5]
        image_paths = []
        for x in most_similar:
            image_paths.append(self.images[x])
        links = []
        for path in image_paths:
            for i in range(44420):
                if self.df['filename'].iloc[i] == path:
                    links.append(self.df['link'].iloc[i])
        
        return links
