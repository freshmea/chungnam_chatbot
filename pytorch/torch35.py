# 1 one-hot encoding
import pandas as pd
from sklearn import preprocessing

class2 = pd.read_csv("data2/class2.csv")
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

train_x = label_encoder.fit_transform(class2)
print(train_x)


# 2. counter vector
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "This is the first document.",
    "This is the second second document.",
    "And the third one.",
    "Is this the first document?",
]
vect = CountVectorizer()
vect.fit(corpus)
vect.vocabulary_


# 3. array cvt
vect.transform(["This is the first document."]).toarray()


# 4.
vect = CountVectorizer(stop_words=["and", "is", "please", "this"]).fit(corpus)
vect.vocabulary_

# 5 tfidf
from sklearn.feature_extraction.text import TfidfVectorizer

doc = ["I like machine learning", "I love deep learning", "I run everyday"]
tfidf_vectorize = TfidfVectorizer(min_df=1)
tfidf_matrix = tfidf_vectorize.fit_transform(doc)
doc_distance = tfidf_matrix * tfidf_matrix.T
print(
    "유사도를 위한",
    str(doc_distance.get_shape()[0]),
    "x",
    str(doc_distance.get_shape()[1]),
    "행렬을 만들었습니다.",
)
print(doc_distance.toarray())

# 6 word2vec

from nltk.tokenize import word_tokenize, sent_tokenize
import warnings

warnings.filterwarnings(action="ignore")
import gensim
from gensim.models import Word2Vec

sample = open("data2/peter.txt", "r", encoding="utf-8")
s = sample.read()

f = s.replace("\n", " ")
data = []

for i in sent_tokenize(f):
    data = []
    for j in word_tokenize(i):
        data.append(j.lower())

data

# 7
model1 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=0)
print("Cosine similarity between 'peter' " + "and 'rabbit' - CBOW : ", end="")
print(model1.wv.similarity("peter", "rabbit"))
print(model1.wv.similarity("peter", "hook"))

# 8
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5, sg=1)
print(model2.wv.similarity("peter", "wendy"))

# 9
from gensim.test.utils import common_texts
from gensim.models import FastText

model = FastText("data/peter.txt", size=4, window=3, min_count=1, epochs=10)

sim_score = model.wv.similarity("peter", "rabbit")
print(sim_score)

sim_score = model.wv.similarity("peter", "hook")
print(sim_score)

# 10
from __future__ import print_function
from gensim.models import KeyedVectors

model_kr = KeyedVectors.load_word2vec_format("data/wiki.ko.vec")

# 11
find_similar_to = "노력"

for smilar_word in model_kr.similar_by_word(find_similar_to):
    print("Word : ", smilar_word[0], "\nSimilarity : ", smilar_word[1])

similarities = model_kr.most_similar(positive=["동물", "육식동물"], negative=["사람"])
print(similarities)
