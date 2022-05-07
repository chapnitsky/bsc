import torch
import torch.nn.functional as F
import torchtext
import time
import random
import pandas as pd


RANDOM_SEED = 58
torch.manual_seed(RANDOM_SEED)

VOCABULARY_SIZE = 200
LEARNING_RATE = 0.005
BATCH_SIZE = 25
NUM_EPOCHS = 15
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 2  # Default or not


df = pd.read_csv('defaults.csv')
TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')


LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)
fields = [('TEXT_COLUMN_NAME', TEXT), ('LABEL_COLUMN_NAME', LABEL)]

dataset = torchtext.legacy.data.TabularDataset(path='defaults.csv', format='csv', skip_header=True, fields=fields)
