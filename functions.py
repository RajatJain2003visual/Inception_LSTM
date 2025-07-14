import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
# from google.colab.patches import cv2_imshow
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from PIL import Image
from IPython.display import display
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import torch.nn.functional as F
import numpy as np 
import random
import math
import seaborn as sns
from tqdm.notebook import tqdm, trange
import pickle

class Encoder(nn.Module):
  def __init__(self, embedding_size, finetune = False):
    super(Encoder, self).__init__()
    self.base_model = models.inception_v3(pretrained=True)
    self.feature_map = {}
    def hook_fn(module, input, output): # hook_fn to get the feature map of image (2048,8,8)
      self.feature_map['spatial'] = output
    self.base_model.Mixed_7c.register_forward_hook(hook_fn)
    self.linear = nn.Linear(2048, embedding_size)
    self.base_model.fc = nn.Identity()
    self.dropout = nn.Dropout(0) # Use dropout initially to stabilize training
    self.relu = nn.ReLU()
    self.finetune = finetune

    for param in self.base_model.parameters():
      param.requires_grad = False


    # Finetuning layers Mixed_7b and Mixed_7c
    if self.finetune is True:
      for param in self.base_model.Mixed_7b.parameters():
        param.requires_grad = True

      for param in self.base_model.Mixed_7c.parameters():
        param.requires_grad = True

  def forward(self, img):

    for module in self.base_model.modules():
      if isinstance(module, nn.BatchNorm2d): # We have to keep batch norm layer to eval mode so that it does not change itself according to our training batch
        module.eval()

    if self.training:
      feature, _ = self.base_model(img)
    else:
      feature = self.base_model(img)

    feature = self.relu(feature)
    embedding = self.dropout(self.linear(feature))
    return embedding, self.feature_map['spatial']


# I am using Bahdanau Attention here
class Attention(nn.Module):
    def __init__(self, feature_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.attn_feat = nn.Linear(feature_dim, attention_dim)     # Project image features
        self.attn_hidden = nn.Linear(hidden_dim, attention_dim)    # Project LSTM hidden state
        self.attn_score = nn.Linear(attention_dim, 1)
        self.attention_dim = attention_dim

    def forward(self, features, hidden,  batch_size):
        # Project image features and LSTM hidden state
        feat_proj = self.attn_feat(features)
        hidden_proj = self.attn_hidden(hidden)

        feat_proj = feat_proj.reshape((batch_size,64,self.attention_dim))
        hidden_proj = hidden_proj.reshape((batch_size,1,self.attention_dim))

        # Combine and score
        score = self.attn_score(torch.tanh(feat_proj + hidden_proj))

        attention_weights = F.softmax(score, dim=1)

        # Weighted sum (context vector)
        context = (attention_weights * features).sum(dim=1)

        return context, attention_weights.squeeze(-1)


class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size):
    super(Decoder, self).__init__()
    self.lstm = nn.LSTMCell(input_size, hidden_size)
    self.context_dropout = nn.Dropout(0.2)
    self.hidden_dropout = nn.Dropout(0.2)

  def forward(self, context, word_embedding, states):
    if len(word_embedding.shape) == 1:
      word_embedding = word_embedding.unsqueeze(0)

    resultant = torch.cat((context,word_embedding), dim=1)
    h_t, c_t = self.lstm(resultant, states)

    return h_t, c_t

class EncoderDecoder(nn.Module):
  def __init__(self, embedding_size, hidden_size, attention_size, padding_idx, vocab, finetune_encoder = False):
    super(EncoderDecoder, self).__init__()
    self.encoder = Encoder(embedding_size, finetune_encoder)
    self.attention = Attention(2048, hidden_size, attention_size)
    self.decoder = Decoder(2048 + embedding_size, hidden_size)
    self.embed = nn.Embedding(len(vocab.vocab), embedding_size, padding_idx=padding_idx)
    self.output_layer = nn.Linear(hidden_size, len(vocab.vocab))
    self.hidden_size = hidden_size
    self.vocab = vocab

    self.hiddenStates_init = nn.Linear(embedding_size, hidden_size)
    self.teacher_forcing_ratio = 1.0

  def forward(self, img, numeric_captions, max_len, device):
    batch_size = img.size(0)

    embedding, feature_map = self.encoder(img)
    features = feature_map.view(batch_size,2048,-1).permute(0,2,1)

    # initial word embedding replaced by image embedding to provide global context to lstm (512)
    word_embedding = embedding.squeeze(0)

    # original caption for teacher forcing
    numeric_caption = numeric_captions
    output_caption = []
    embeddings = self.embed(numeric_caption)

    h_t = self.hiddenStates_init(embedding) # initializing hidden states to image embedding
    c_t = self.hiddenStates_init(embedding)

    context,_ = self.attention(features, h_t, batch_size)


    for i in range(max_len):
      h_t, c_t = self.decoder(context, word_embedding, (h_t, c_t))

      context,_ = self.attention(features, h_t, batch_size)
      logits = self.output_layer(h_t)
      max_index = logits.argmax(1)
      predicted_word_embedding = self.embed(max_index)
      output_caption.append(logits)

      # Teacher Forcing
      if self.training:
        use_teacher_forcing = random.random() < self.teacher_forcing_ratio
        if use_teacher_forcing:
          if i <= embeddings.size(1):
            word_embedding = embeddings[:, i, :]
          else:
            word_embedding = self.embed(torch.tensor([self.vocab.vocab["<PAD>"]], device=device))
        else:
          word_embedding = predicted_word_embedding

      else:
        word_embedding = predicted_word_embedding

    return torch.stack(output_caption,dim=1)

class vocabulary:
  def __init__(self, document):
    self.document = document
    self.tokens = word_tokenize(document.lower())
    self.vocab = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<EOS>': 3}
    for token in Counter(self.tokens).keys():
      if token not in self.vocab:
        self.vocab[token] = len(self.vocab)
    self.itos = {}
    for word, index in self.vocab.items():
        self.itos[index] = word

  def text_to_numeric(self, text):
    tokenized_list = word_tokenize(text.lower())
    numeric_text = []
    for item in tokenized_list:
      if item not in self.vocab:
        numeric_text.append(self.vocab['<UNK>'])
      else:
        numeric_text.append(self.vocab[item])
    numeric_text.append(self.vocab['<EOS>'])
    while len(numeric_text) < 80:
      numeric_text.append(self.vocab['<PAD>'])
    return numeric_text

class CustomDataset(Dataset):
  def __init__(self, df, vocab):
    print("start \n")
    self.paths, self.captions = df['image_name'], df['comment']
    print(f"{self.paths[0]}")
    self.vocab = vocab
    self.transform = transforms.Compose([
        transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2,contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

  def __len__(self):
    return len(self.captions)

  def __getitem__(self, idx):
    img_path = os.path.join('/content/flickr30k_images',self.paths[idx])
    img = Image.open(img_path).convert("RGB")
    caption = self.captions[idx]
    numeric_caption = self.vocab.text_to_numeric(caption)
    img = self.transform(img)
    numeric_caption = torch.tensor(numeric_caption)
    return img,numeric_caption

def generate_captions_beam_search(model, img_path, beam_size=3, blue_score=False, maxlen=79, batch_size=1, device='cuda'):
  model.eval()
  with torch.no_grad():

    img = Image.open(img_path).convert("RGB")
    if blue_score is False:
      display(img)
    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)

    embedding, feature_map = model.encoder(img)
    features = feature_map.view(batch_size,2048,-1).permute(0,2,1)

    hidden = embedding.squeeze(1)

    h_t = model.hiddenStates_init(embedding)
    c_t = model.hiddenStates_init(embedding)

    context,_ = model.attention(features, h_t, batch_size)

    beams = [[0,[],(h_t,c_t)]]

    for _ in range(maxlen):
      new_beam = []

      for beam in beams:
        if len(beam[1]) == 0:
          h_t, c_t = model.decoder(context, hidden, (h_t, c_t))

        else:
          h_t, c_t = beam[2]
          if h_t is None:
            new_beam.append([beam[0], beam[1], beam[2]])
            continue

          context, _ = model.attention(features, h_t, batch_size)
          word_embedding = model.embed(torch.tensor(model.vocab.vocab[beam[1][-1]], device=device))
          word_embedding = word_embedding.unsqueeze(0)

          h_t, c_t = model.decoder(context, word_embedding, (h_t, c_t))

        output_logits = model.output_layer(h_t)
        log_probs = torch.log_softmax(output_logits, dim=1)
        top_log_probs, top_indices = log_probs.topk(beam_size, dim=1)

        for i in range(beam_size):
            max_index = top_indices[0][i].item()
            word = model.vocab.itos[max_index]
            new_caption = beam[1] + [word]
            new_score = beam[0] + top_log_probs[0][i].item()
            if word in ['<EOS>','.']:
              new_beam.append([new_score, new_caption, (None, None)])
              continue
            new_beam.append([new_score,new_caption, (h_t,c_t)])

        new_beam.sort(key=lambda x: x[0], reverse=True)
        beams = new_beam[:beam_size]

    best_caption = []
    for i in range(len(beams)):
      best_caption.append([beams[i][0],beams[i][1]])

    if blue_score is True:
      return best_caption[0][1]

    return best_caption
  
@st.cache_resource
def initialize(device):
  with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)

  embedding_size = 300
  hidden_size = 512
  attention_size = 256
  model = EncoderDecoder(embedding_size=embedding_size, hidden_size=hidden_size, attention_size=attention_size, padding_idx=0, vocab=vocab)
  model.to(device)
  exist = True
  if os.path.exists('../new_best_modelweights1.pth.tar'):
    checkpoint = torch.load('../new_best_modelweights1.pth.tar',map_location=torch.device(device))
    model.load_state_dict(checkpoint['state_dict'])
    print("=> Loading checkpoint")
  else:
    exist = False
    print("=> No checkpoint found")
  return vocab,model,exist

