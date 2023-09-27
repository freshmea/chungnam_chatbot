# 1 multilingual bert
import torch
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# 2 sentence tokenization
text = "나는 파이토치를 이용한 딥러닝을 학습중이다."
marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
print(tokenized_text)

# 3 text define
text = "과수원에 사과가 많았다." "친구가 나에게 사과한다." "백설공주는 독이 든 사과를 먹었다."
print(text)
marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer.tokenize(marked_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
for tup in zip(tokenized_text, indexed_tokens):
    print("{:<12} {:>6,}".format(tup[0], tup[1]))

# 4 recognize unit
segments_ids = [1] * len(tokenized_text)
print(segments_ids)

# 5 to_tensor
token_tensor = torch.tensor([indexed_tokens])
segments_tensor = torch.tensor([segments_ids])


# 6 model load
model = BertModel.from_pretrained(
    "bert-base-multilingual-cased", output_hidden_states=True
)
model.eval()


# 7 model predition
with torch.no_grad():
    outputs = model(token_tensor, segments_tensor)
    hidden_states = outputs[2]


# 8 hidden states information
print(f"Number of layers: {len(hidden_states)} initial embeddings + 12 BERT layers")
layer_i = 0
print(f"Number of batches: {len(hidden_states[layer_i])}")
batch_i = 0
print(f"Number of tokens: {len(hidden_states[layer_i][batch_i])}")
token_i = 0
print(f"Number of hidden units: {len(hidden_states[layer_i][batch_i][token_i])}")


# 9 hidden states type, size
print(f"hidden state type: {type(hidden_states)}")
print(f"hidden state size: {hidden_states[0].size()}")


# 10 change type of tensor
token_embeddings = torch.stack(hidden_states, dim=0)
print(token_embeddings.size())


# 10.5 squeeze tensor
token_embeddings = torch.squeeze(token_embeddings, dim=1)
print(token_embeddings.size())

# 11 change dimension of tensor
token_embeddings = token_embeddings.permute(1, 0, 2, -1)
print(token_embeddings.size())


# 12 token type check
token_vecs_cat = []
for token in token_embeddings:
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    token_vecs_cat.append(cat_vec)
print(f"shape is: {len(token_vecs_cat)} X {len(token_vecs_cat[0])}")


# 13 bind layers to final token
token_vecs_sum = []
for token in token_embeddings:
    sum_vec = torch.sum(token[-4:], dim=0)
    token_vecs_sum.append(sum_vec)
print(f"shape is: {len(token_vecs_sum)} X {len(token_vecs_sum[0])}")


# 14 sentence vector
token_vecs = hidden_states[-2][0]
sentence_embedding = torch.mean(token_vecs, dim=0)
print(f"shape is: {sentence_embedding.size()}")


# 15 print index
for i, token_str in enumerate(tokenized_text):
    print(i, token_str)


# 16 check token vector
print("사과가 많았다.", str(token_vecs_sum[6][:5]))
print("나에게 사과했다.", str(token_vecs_sum[10][:5]))
print("사과를 먹었다.", str(token_vecs_sum[19][:5]))


# 17 check sentence vector cosine similarity
from scipy.spatial.distance import cosine

diff_apple = 1 - cosine(token_vecs_sum[5], token_vecs_sum[27])
sample_apple = 1 - cosine(token_vecs_sum[5], token_vecs_sum[16])
print(f"유사한 의미에 대한 벡터 유사도: {diff_apple}")
print(f"다른 의미에 대한 벡터 유사도: {sample_apple}")
