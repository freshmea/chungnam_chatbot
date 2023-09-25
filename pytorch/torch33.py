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