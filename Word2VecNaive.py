from sklearn import cluster
import gensim
import numpy as np

new_model = gensim.models.KeyedVectors.load('working.embedded')


words = []
for i, term in enumerate(new_model.wv.vocab):
    words.append(term)
    if i > 30:
        break


words = np.array(['beans', 'broccoli', 'carrot', 'celery', 'corn', 'cucumber', 'asparagus', 'eggplant', 'lettuce', 'cabbage', 'onion', 'peas', 'potato', 'pumpkin', 'radish', 'spinach', 'tomato', 'turnip',
         'alligator', 'ant', 'bear', 'bee', 'bird', 'camel', 'cat', 'cheetah', 'chicken', 'chimpanzee', 'cow',
         'crocodile', 'deer', 'dog', 'dolphin', 'duck', 'eagle', 'elephant', 'fish', 'fly', 'fox', 'frog', 'giraffe',
         'goat', 'goldfish', 'hamster',
         'accountant', 'actor', 'actress', 'athlete', 'author', 'baker', 'banker', 'barber', 'beautician', 'broker',
         'burglar', 'butcher', 'carpenter', 'chauffeur', 'chef', 'clerk', 'coach', 'craftsman', 'criminal', 'crook',
         'dentist', 'doctor', 'editor', 'engineer', 'farmer'])

X = np.zeros((len(words),len(words)))
print(new_model.similarity('vegetable', words[0]), new_model.similarity('animal', words[0]), new_model.similarity('profession', words[0]))
for i, iV, in enumerate(words):
    for j, jV in enumerate(words):
        try:
            X[i, j] = new_model.similarity(iV, jV)
        except  KeyError:
            X[i,j] = 2

for i in range(X.shape[0]):
    if X[i,i] == 2:
        words[i] = "empty"

groups = {}
checkingArray = np.arange(words.shape[0])
for i, iPos in enumerate(X):
    flag = True
    for j, jPos in enumerate(iPos):
        if jPos > 0.7 and (checkingArray[j] != -1) and jPos != 2 and checkingArray[i] != -1 and i != j:
            if flag:
                groups[words[i]] = [words[j]]
                flag = False
            else:
                groups[words[i]].append(words[j])
            checkingArray[j] = -1
print (groups)


