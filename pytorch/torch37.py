# 1
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import os, re, random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 2
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 20


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# 3
def normalizeString(df, lang):
    sentence = df[lang].str.lower()
    sentence = sentence.str.replace("[^A-Za-z\s]+", " ")
    sentence = sentence.str.nomalize("NFD")
    sentence = sentence.str.encode("ascii", errors="ignore").str.decode("utf-8")
    return sentence


def read_sentence(df, lang1, lang2):
    sentence1 = normalizeString(df, lang1)
    sentence2 = normalizeString(df, lang2)
    return sentence1, sentence2


def read_file(loc, lang1, lang2):
    df = pd.read_csv(loc, delimiter="\t", header=None, names=[lang1, lang2])
    return df


def process_data(lang1, lang2):
    df = read_file("data/%s-%s.txt" % (lang1, lang2), lang1, lang2)
    sentence1, sentence2 = read_sentence(df, lang1, lang2)

    input_lang = Lang()
    output_lang = Lang()
    pairs = []
    for i in range(len(df)):
        if (
            len(sentence1[i].split(" ")) < MAX_LENGTH
            and len(sentence2[i].split(" ")) < MAX_LENGTH
        ):
            full = [sentence1[i], sentence2[i]]
            input_lang.addSentence(sentence1[i])
            output_lang.addSentence(sentence2[i])
            pairs.append(full)
    return input_lang, output_lang, pairs


# 4
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    output_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, output_tensor)


# 5 encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers)

    def forward(self, src):
        embedded = self.embedding(src).view(1, 1, -1)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


# 6 decoder
class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embbed_dim = embbed_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_dim, embbed_dim)
        self.gru = nn.GRU(embbed_dim, hidden_dim, num_layers=num_layers)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = input.view(1, -1)
        embedded = F.relu(self.embedding(input))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.softmax(self.out(output[0]))
        return prediction, hidden


# 7 seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, MAX_LENGTH=MAX_LENGTH):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, input_lang, output_lang, teacher_forcing_ratio=0.5):
        input_length = input_lang.size(0)
        batch_size = output_lang.shape[1]
        target_length = output_lang.shape[0]
        vocab_size = self.decoder.output_dim
        outputs = torch.zeros(target_length, batch_size, vocab_size).to(self.device)

        for i in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_lang[i])
        decoder_hidden = encoder_hidden.to(device)
        decoder_input = torch.tensor([[SOS_token]], device=device)

        for t in range(target_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1)
            input = output_lang[t] if teacher_force else topi
            if teacher_force == False and input.item() == EOS_token:
                break
        return outputs


# 8 model, loss
teacher_forcing_ratio = 0.5


def makeModel(model, input_tensor, target_tenser, model_optimizer, criterion):
    model_optimizer.zero_grad()
    input_length = input_tensor.size(0)
    loss = 0
    epoch_loss = 0
    output = model(input_tensor, target_tenser)
    num_iter = output.size(0)

    for ot in range(num_iter):
        loss += criterion(output[ot], target_tenser[ot])
    loss.backward()
    model_optimizer.step()
    epoch_loss = loss.item() / num_iter
    return epoch_loss


# 9 train
def trainModel(model, input_lang, output_lang, pairs, num_iteration=20000):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.NLLLoss()
    total_loss_iterations = 0

    training_pairs = [
        tensorFromPair(input_lang, output_lang, random.choice(pairs))
        for i in range(num_iteration)
    ]

    for iter in range(1, num_iteration + 1):
        training_pairs = training_pairs[iter - 1]
        input_tensor = training_pairs[0]
        target_tensor = training_pairs[1]
        loss = makeModel(model, input_tensor, target_tensor, optimizer, criterion)
        total_loss_iterations += loss

        if iter % 5000 == 0:
            average_loss = total_loss_iterations / 5000
            total_loss_iterations = 0
            print("loss: %.4f" % (average_loss))

    torch.save(model.state_dict(), "data/mytraining.pt")
    return model


# 10 evaluate
def evaluate(model, input_lang, output_lang, sentences, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentences[0])
        output_tenser = tensorFromSentence(output_lang, sentences[1])
        decoded_words = []
        output = model(input_tensor, output_tenser)

        for ot in range(output.size(0)):
            topv, topi = output[ot].topk(1)
            if topi[0].item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi[0].item()])
    return decoded_words


def evaluateRandomly(model, input_lang, output_lang, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print("input", pair[0])
        print("output", pair[1])
        output_words = evaluate(model, input_lang, output_lang, pair)
        output_sentence = " ".join(output_words)
        print("predicted : ", output_sentence)


# 11 main
lang1 = "eng"
lang2 = "fra"
input_lang, output_lang, pairs = process_data(lang1, lang2)

randomize = random.choice(pairs)
print("randomize sentence", randomize)

input_size = input_lang.n_words
output_size = output_lang.n_words
print("input size", input_size, "output size", output_size)

embed_size = 256
hidden_size = 512
num_layers = 1
num_iteration = 75000

encoder = Encoder(input_size, hidden_size, num_layers).to(device)
decoder = Decoder(output_size, hidden_size, embed_size, num_layers).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

print(encoder)
print(decoder)
