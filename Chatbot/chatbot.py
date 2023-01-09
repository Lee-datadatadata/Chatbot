import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
import pickle

#from tensorflow.keras.models import load_model


lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
#model = load_model(r"C:\Users\Leham\OneDrive\Desktop\Chatbot for the lonely\chatbot.h5")

with open("chatbot.pkl", "rb") as f:
    model = pickle.load(f)

def clean_up_sentince(sentince):
    sentince_words = nltk.word_tokenize(sentince)
    sentince_words = [lemmatizer.lemmatize(word) for word in sentince_words]
    return sentince_words

def bag_of_words(sentince):
    sentince_words = clean_up_sentince(sentince)
    bag = [0] * len(words)
    for w in sentince_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_classes(sentince):
    bow = bag_of_words(sentince)
    res = model.predict(np.array([bow]))
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if (r > ERROR_THRESHOLD).any()]


    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print('Go!, Bot is running!')

while True:
    message = input('')
    ints = predict_classes(message)
    res = get_response(ints, intents)
    print(res)