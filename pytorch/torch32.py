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

# 4
print(komoran.pos('소파 위에 있는 것이 고양이인가요? 강아지인가요?'))

# 5
import pandas as pd
df = pd.read_csv('data/class2.csv')
df

# 6
df.isnull().sum()
# df.isna().sum()
print(f'결측치의 비율은 {df.isnull().sum()/len(df)*100:.2f}% 이다.')

# 7
print(df.dropna(how='all'))
print(df)

# 8
df1 = df.dropna()
print(df1)

# 9 
df2 = df.fillna()
print(df2)

# 10
df['x'].fillna(df['x'].mean(), inplace=True)
print(df)