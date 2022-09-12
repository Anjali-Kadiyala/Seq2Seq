from ctypes.wintypes import tagRECT
# from fileinput import _HasReadlineAndFileno
from pickletools import optimize
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k  #importing German to Eng dataset
from torchtext.data import Field, BucketIterator #for preprocessing
import numpy as np
import spacy  #for tokenizer
import random
from torch.utils.tensorboard import SummaryWriter  #to print to tensorboard for loss plots
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')

def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger(text)]

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng(text)]

#preprocessing
german = Field(tokenizer=tokenizer_ger, lower = True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenizer=tokenizer_eng, lower = True, init_token='<sos>', eos_token='<eos>')

#using dataset
train_data, val_data, test_data = Multi30k.split(exts=('.de','.en'), fields=(german, english))

german.build_vocab(train_data, max_size=1000, min_freq=2) #if a word is not in the input at least twice, we don't add it to the vocabulary.
english.build_vocab(train_data, max_size=1000, min_freq = 2)

class Encoder(nn.Module): #first LSTM

    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p) #p is the number of values to dropout
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    
    def forward(self, x):  #x is a vector of indices
        #x shape : (seq_length, N)
        embedding = self.droupout(self.exbedding(x)) #embeedding shape : (seq_length, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell

class Decoder(nn.Module): #second LSTM

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p): #input size and output size are of same size
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p) #hidden_size of the encoder and decoder are the same.
        self.fc = nn.Linear(hidden_size, output_size) #fully connected layer

    def forward(self, x, hidden, cell):
        #shape of x: (N) but we want (1,N) as decoder predicts one word at a time, so 1 represets that we have N words, 1 word at a time
        x = x.unsqueeze(0)
        embedding = self.droupout(self.embedding(x)) 
        #embedding shape: (1, N, embedding_size)
        outputs, (hidden, cell) = self.RNN(embedding, (hidden, cell))
        #shape of outputs: (1, N, hidden_size)
        predictions = self.fc(outputs) 
        #shape of predictions : (1, N, length_of_vocab) and we want to remove 1 now. 
        predictions.squeeze(0)
        return predictions, hidden, cell


class Seq2Seq(nn.Module): #This combines the encoder and decoder

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        #shape of source (target_len, N)
        target_length = target.shape[0]
        target_vocab_size = len(english.vocab)
        outputs = torch.zeros(target_length, batch_size, target_vocab_size).to(device)
        hidden, cell = self.encoder(source)
        x = target[0] #grabbing start token
        for t in range(1, target_length):
            output, hidden, cell = self.decoder(x, hidden, cell)
            outputs[t] = output
            #output size: (N, english_vocab_size) 
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return output


#Now we have model
### We are ready to do training ###

#training hyperparameters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

#model hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300 #100-300 is a good size.
decoder_embedding_size = 300
hidden_size = 1024 #used in SEq2Seq papaer. It is a lil smaller (medium size) for today
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.3

#tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, validation_iterator, test_iterator = BucketIterator.splits((train_data, val_data, test_data), batch_size = batch_size, sort_within_batch = True, sort_key = lambda x: len(x.src), device=device)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')
    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)
        output = model(inp_data, target) #output shape: (trg_len, batch_size, output_dim)
        #reshaping output
        output = output[1:].reshape(-1, output.shape[2]) # removing start token
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step = step)
        step += 1