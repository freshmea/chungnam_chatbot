# 1 
from nltk import sent_tokenize
text_sample = '''
Share full article


About a half dozen writers and actors carrying signs, and some wearing hats, picket outside Paramount studios in Los Angeles.
Striking Hollywood writers and actors picketing outside Paramount Studios in Los Angeles last week.Credit...Mario Tama/Getty Images

Brooks BarnesJohn Koblin
By Brooks Barnes and John Koblin
Sept. 25, 2023, 1:09 a.m. ET
'''
tokenized_sentences = sent_tokenize(text_sample)
print(tokenized_sentences)

# 2
from nltk import word_tokenize
sentence = "The faster Harry got to the store, the faster Harry, the faster, would get home."
words = word_tokenize(sentence)
words

# 3 
from nltk.tokenize import WordPunctTokenizer
sentence = "it's nothing that you don't already know the answer to most people aren't aware of how their inner world works."
words = WordPunctTokenizer().tokenize(sentence)
words

# 4
import csv
from konlpy.tag import Okt
from gensim.models import word2vec

f = open('data/ratings_train.txt', 'r', encoding='utf-8')
rdr = csv.reader(f, delimiter='\t')
rdw = list(rdr)
f.close()

# 5
twitter = Okt()

result = []
for line in rdw:
    malist = twitter.pos(line[1], norm=True, stem=True)
    r = []
    for word in malist:
        if not word[1] in ['Josa', 'Eomi', 'Punctuation']:
            r.append(word[0])
    rl = (" ".join(r)).strip()
    result.append(rl)
    print(rl)

# 6
with open("NaverMovie.nlp", 'w', encoding='utf-8') as fp:
    fp.write("\n".join(result))
    
# 7
mData = word2vec.LineSentence("NaverMovie.nlp")
mModel = word2vec.Word2Vec(mData, vector_size=200, window=10, hs=1, min_count=2, sg=1)
mModel.save("NaverMovie.model")


# 8 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

sample_text = "One of first things I noticed about the book was the unusual way author chose to narrate story. Instead of telling it from perspective of one or two characters, author chose to tell it from perspective of an entire town."
text_tokens = word_tokenize(sample_text)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
print(f'불용어 제거 미적용: {text_tokens}', '\n')
print(f'불용어 제거 적용: {tokens_without_sw}')

# 9
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

print(stemmer.stem('obsesse'), stemmer.stem('obsessed'))
print(stemmer.stem('standardizes'), stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))

# 10 lancaster
from nltk.stem import LancasterStemmer
stemmer = LancasterStemmer()

print(stemmer.stem('obsesse'), stemmer.stem('obsessed'))
print(stemmer.stem('standardizes'), stemmer.stem('standardization'))
print(stemmer.stem('national'), stemmer.stem('nation'))

