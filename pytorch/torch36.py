# 1
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")
from sklearn.decomposition import PCA
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# 2
glove_file = datapath("C:\chungnam_chatbot\pytorch\data2\glove.6B.100d.txt")
word2vec_glove_file = get_tmpfile(
    "C:\chungnam_chatbot\pytorch\data2\glove.6B.100d.vector.txt"
)
glove2word2vec(glove_file, word2vec_glove_file)

# 3
model = KeyedVectors.load_word2vec_format(word2vec_glove_file)
model.most_similar("bill")

# 4
model.most_similar("cherry")

# 5
result = model.most_similar(positive=["woman", "king"], negative=["man"])
for re in result:
    print(f"{re[0]} {re[1]:.4f}")


# 6
def analogy(x1, x2, y1):
    result = model.most_similar(positive=[y1, x2], negative=[x1])
    return result[0][0]


print(analogy("australia", "beer", "france"))
print(analogy("tail", "tallest", "long"))

print(model.doesnt_match("breakfast cereal dinner lunch".split()))
