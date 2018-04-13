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
'''
from nltk.corpus import brown
nltk.download('brown')
model = gensim.models.Word2Vec(brown.sents())
nltk.download('word2vec_sample')
model.save('brown.embedding')

new_model = gensim.models.Word2Vec.load('brown.embedding')


from nltk.data import find
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
model.save ('working.embedded')


'''


#new_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
new_model = gensim.models.KeyedVectors.load('working.embedded')
#print(new_model.most_similar(positive=['vegetable'], negative=[], topn=5))

print("finished")
#print(words)

words = np.array(['beans', 'broccoli', 'carrot', 'celery', 'corn', 'cucumber', 'asparagus', 'eggplant', 'lettuce', 'cabbage', 'onion', 'peas', 'potato', 'pumpkin', 'radish', 'spinach', 'tomato', 'turnip',
         'alligator', 'ant', 'bear', 'bee', 'bird', 'camel', 'cat', 'cheetah', 'chicken', 'chimpanzee', 'cow',
         'crocodile', 'deer', 'dog', 'dolphin', 'duck', 'eagle', 'elephant', 'fish', 'fly', 'fox', 'frog', 'giraffe',
         'goat', 'goldfish', 'hamster',
         'accountant', 'actor', 'actress', 'athlete', 'author', 'baker', 'banker', 'barber', 'beautician', 'broker',
         'burglar', 'butcher', 'carpenter', 'chauffeur', 'chef', 'clerk', 'coach', 'craftsman', 'criminal', 'crook',
         'dentist', 'doctor', 'editor', 'engineer', 'farmer'])


#words = ['salami', 'meat', 'pork', 'candy', 'dessert', 'chocolate', 'rice', 'curry', 'chicken', 'wheat', 'potato']
#print (new_model.most_similar(positive=words, negative=[], topn=1))

X = np.zeros((len(words),len(words)))
for i, iV, in enumerate(words):
    for j, jV in enumerate(words):
        try:
            X[i, j] = new_model.similarity(iV, jV)
        except  KeyError:
            X[i,j] = 2

for i in range(X.shape[0]):
    if X[i,i] == 2:
        words[i] = "empty"

for i, iV, in enumerate(words):
    for j, jV in enumerate(words):
        try:
            X[i, j] = new_model.similarity(iV, jV)
        except  KeyError:
            X[i,j] = 2

#for i in range(1,15):
kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(X)
labels = kmeans.labels_
flag = False
#for i in range(4):

wordsTemp = words[clusterIndicesNumpy(2, labels)]
s = np.zeros(len(new_model[words[0]]))
for i in wordsTemp:
    s += new_model[i]
s = s / len(wordsTemp)
print("asdasd", new_model.most_similar(positive=[s], negative=[], topn=1))


for i in range(4):
    wordsTemp = words[clusterIndicesNumpy(i, labels)]
    print(wordsTemp)
    similarWord = new_model.most_similar(positive=wordsTemp, negative=[], topn=1)
    print (similarWord[0][0])
similarArray = []
unSimilarArray = []
shittyArray = []
for i in wordsTemp:
    temp =  new_model.similarity(i, similarWord[0][0])
    if temp > 0.7:
        similarArray.append(i)
    elif temp > 0.5:
        unSimilarArray.append(i)
    else:
        shittyArray.append(i)
print ("sim: ", similarArray)
print ("unsim: ", unSimilarArray)
print ("shitty: ", shittyArray)
if len(similarArray) > 0:
    print(new_model.most_similar(positive=similarArray, negative=[], topn=1))
if len(unSimilarArray) > 0:
    print(new_model.most_similar(positive=unSimilarArray, negative=[], topn=1))
if len(shittyArray) > 0:
    print(new_model.most_similar(positive=shittyArray, negative=[], topn=1))
