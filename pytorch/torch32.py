# 1 
import nltk
nltk.download()
text = nltk.word_tokenize("Is it possible distinguishing cats and dogs?")
text

# 2 
nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(text)

# 3
!curl -s https://raw.githubusercontent.com/teddylee777/machine-learning/master/99-Misc/01-Colab/mecab-colab.sh | bash

from konlpy.tag import Komoran
komoran = Komoran()
print(komoran.morphs('딥러닝이 쉽나요? 어렵나요?'))