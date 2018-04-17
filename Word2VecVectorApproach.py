from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
import gensim
#import nltk
#from nltk.corpus import wordnet
import numpy as np



def clusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]

#new_model = gensim.models.KeyedVectors.load('working.embedded')
new_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

print("finished")


words = np.array(['beans', 'broccoli', 'carrot', 'celery', 'corn', 'cucumber', 'asparagus', 'eggplant', 'lettuce',
                  'cabbage', 'onion', 'peas', 'potato', 'pumpkin', 'radish', 'spinach', 'tomato', 'turnip',
         'alligator', 'ant', 'bear', 'bee', 'bird', 'camel', 'cat', 'cheetah', 'chicken', 'chimpanzee', 'cow',
         'crocodile', 'deer', 'dog', 'dolphin', 'duck', 'eagle', 'elephant', 'fish', 'fly', 'fox', 'frog', 'giraffe',
         'goat', 'goldfish', 'hamster',
         'accountant', 'actor', 'actress', 'athlete', 'author', 'baker', 'banker', 'barber', 'beautician', 'broker',
         'burglar', 'butcher', 'carpenter', 'chauffeur', 'chef', 'clerk', 'coach', 'craftsman', 'criminal', 'crook',
         'dentist', 'doctor', 'editor', 'engineer', 'farmer'])

X = []
delPos = []
for ipos, i in enumerate(words):
    try:
        X.append(new_model[i])
    except :
        delPos.append(ipos)
words = np.delete(words, delPos)
print(words)
'''
for ipos, i in enumerate(words):
    print(ipos, i, end=' ')

13 36 60
'''
X = np.array(X)
s = np.zeros(X.shape[1])
a = np.zeros(X.shape[1])
for i in range(0, 14):
    s += new_model[words[i]] - new_model['vegetables']

for i in range(14, 37):
    s += new_model[words[i]] - new_model['animals']

for i in range(37, 61):
    s += new_model[words[i]] - new_model['professions']
s = s/60

kmeans = cluster.KMeans(n_clusters=8)
kmeans.fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

for i in range(8):
    wordsTemp = words[clusterIndicesNumpy(i, labels)]
    '''
    fruitArray = np.full((int(wordsTemp.shape[0]/1.7),), 'fruit')
    appleArray = np.full((int(wordsTemp.shape[0]/1.7), ), 'apple')
    wordsPlusFruits = np.append(wordsTemp, fruitArray)
    similarWord = new_model.most_similar(positive=wordsPlusFruits, negative=appleArray, topn=1)
    print(wordsTemp)
    print (similarWord[0][0])
    for j in wordsTemp:
        print("{:3.2}".format(new_model.similarity(j, similarWord[0][0])), end=' ')
    '''
    print(wordsTemp)
    print('-', new_model.most_similar(positive=[centers[i]-s], negative=[], topn=4))
    #print('?', new_model.most_similar(positive=[centers[i]], negative=[], topn=4))
    #print('+', new_model.most_similar(positive=[centers[i]+s], negative=[], topn=4))