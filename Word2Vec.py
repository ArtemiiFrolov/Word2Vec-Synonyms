from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import metrics
import gensim
import nltk
from nltk.corpus import wordnet
import numpy as np
import nltk
from nltk.corpus import wordnet
from gap_statistic import OptimalK
import matplotlib.pyplot as plt

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


new_model = gensim.models.KeyedVectors.load('working.embedded')
print (new_model.most_similar(positive=['chocolate'], negative=[], topn=5))
words = []
for i, term in enumerate(new_model.wv.vocab):
    words.append(term)
    if i > 30:
        break
#print(words)

words = ['beans', 'broccoli', 'carrot', 'celery', 'corn', 'cucumber', 'asparagus', 'eggplant', 'lettuce', 'cabbage', 'onion', 'peas', 'potato', 'pumpkin', 'radish', 'spinach', 'tomato', 'turnip',
         'alligator', 'ant', 'bear', 'bee', 'bird', 'camel', 'cat', 'cheetah', 'chicken', 'chimpanzee', 'cow',
         'crocodile', 'deer', 'dog', 'dolphin', 'duck', 'eagle', 'elephant', 'fish', 'fly', 'fox', 'frog', 'giraffe',
         'goat', 'goldfish', 'hamster',
         'accountant', 'actor', 'actress', 'athlete', 'author', 'baker', 'banker', 'barber', 'beautician', 'broker',
         'burglar', 'butcher', 'carpenter', 'chauffeur', 'chef', 'clerk', 'coach', 'craftsman', 'criminal', 'crook',
         'dentist', 'doctor', 'editor', 'engineer', 'farmer']


#words = ['salami', 'meat', 'pork', 'candy', 'dessert', 'chocolate', 'rice', 'curry', 'chicken', 'wheat', 'potato']
X = np.zeros((len(words),len(words)))
print(new_model.similarity('vegetable', words[0]), new_model.similarity('animal', words[0]), new_model.similarity('profession', words[0]))
for i, iV, in enumerate(words):
    for j, jV in enumerate(words):
        try:
            X[i, j] = new_model.similarity(iV, jV)
        except  KeyError:
            X[i,j] = 2
            #print(jV, "!")
#print(X)
optimalK = OptimalK(parallel_backend='None')
n_clusters = optimalK(X, cluster_array=np.arange(1, 20))
print('Optimal clusters: ', n_clusters)


plt.plot(optimalK.gap_df.n_clusters, optimalK.gap_df.gap_value, linewidth=3)
plt.scatter(optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].n_clusters,
            optimalK.gap_df[optimalK.gap_df.n_clusters == n_clusters].gap_value, s=250, c='r')
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()

kmeans = cluster.KMeans(n_clusters=4)
kmeans.fit(X)

labels = kmeans.labels_
dictX = {}
for i, label in enumerate(kmeans.labels_):
    if label in dictX:
        dictX[label].append(words[i])
    else:
        dictX[label] = [words[i]]
print(dictX)



print("Cluster id labels for inputted data")
print(labels)
'''
sentences = [['salami', 'lunchmeat', 'bologna', 'turkey', 'candy', 'dessert', 'chocolate']]
model = Word2Vec(sentences, min_count=1)
word_vectors = model.wv.syn0

n_words = word_vectors.shape[0]
vec_size = word_vectors.shape[1]
print("#words = {0}, vector size = {1}".format(n_words, vec_size))

kmeans = KMeans(n_clusters=3)
idx = kmeans.fit_predict(word_vectors)
print("finished")



word_centroid_list = list(zip(model.wv.index2word, idx))
word_centroid_list_sort = sorted(word_centroid_list, key=lambda el: el[1], reverse=False)
for word_centroid in word_centroid_list_sort:
    print( word_centroid[0] + '\t' + str(word_centroid[1]) + '\n')
'''

'''
sentences = [['salami', 'lunchmeat', 'bologna', 'turkey', 'candy', 'dessert', 'chocolate']]
model = Word2Vec(sentences, min_count=1)
print (list(model.wv.vocab))
#print (model.most_similar(positive=['chocolate'], negative=[], topn=22))
X = model[model.wv.vocab]

kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

print("Cluster id labels for inputted data")
print(labels)
'''
'''
w1 = wordnet.synset('salami.n.01')
w2 = wordnet.synset('bologna.n.01')
w3 = wordnet.synset('chocolate.n.01')
w4 = wordnet.synset('dessert.n.01')
print(w1.lch_similarity(w2))
print(w1.lch_similarity(w3))
print(w1.lch_similarity(w4))
print(w2.lch_similarity(w3))
print(w2.lch_similarity(w4))
print(w3.lch_similarity(w4))
'''