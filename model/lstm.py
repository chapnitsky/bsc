import spacy
import time
import numpy as np
import pandas as pd
import torchtext
import torch
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F


class SenDataSet(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sen = self.data['sen'][index]
        typ = self.data['isdefault'][index]
        label = torch.tensor(typ, dtype=torch.long)

        return sen, label


class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)
        # self.rnn = torch.nn.RNN(embedding_dim,
        #                        hidden_dim,
        #                        nonlinearity='relu')
        self.rnn = torch.nn.LSTM(embedding_dim,
                                 hidden_dim)

        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text dim: [sentence length, batch size]

        embedded = self.embedding(text)
        # embedded dim: [sentence length, batch size, embedding dim]

        output, (hidden, cell) = self.rnn(embedded)
        # output dim: [sentence length, batch size, hidden dim]
        # hidden dim: [1, batch size, hidden dim]

        hidden.squeeze_(0)
        # hidden dim: [batch size, hidden dim]

        output = self.fc(hidden)
        return output


def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float() / num_examples * 100


def predict_def(model, sentence):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed).to(DEVICE)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.nn.functional.softmax(model(tensor), dim=1)
    return prediction[0][0].item()


if __name__ == "__main__":
    df = pd.read_csv('defaults.csv')

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

    TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
    LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)
    fields = [('sen', TEXT), ('isdefault', LABEL)]

    dataset = SenDataSet(data_frame=df)
    TOTAL_SIZE = len(dataset)
    TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT = .33333, .33333, .33333

    TRAIN_SIZE = int(np.ceil(TRAIN_PERCENT * TOTAL_SIZE))
    VAL_SIZE = int(np.ceil(VAL_PERCENT * TOTAL_SIZE))
    TEST_SIZE = int(np.ceil(TEST_PERCENT * TOTAL_SIZE))
    tttt1 = dataset[:TRAIN_SIZE]
    tttt2 = dataset[TRAIN_SIZE: TRAIN_SIZE + VAL_SIZE]
    tttt3 = dataset[TRAIN_SIZE+VAL_SIZE: TRAIN_SIZE+VAL_SIZE+TEST_SIZE]
    train_data, valid_data, test_data = random_split(dataset, [int(np.ceil(TRAIN_SIZE)), int(np.ceil(VAL_SIZE)),
                                                               int(np.ceil(TEST_SIZE))])
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    train_loader, valid_loader, test_loader = torchtext.legacy.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=False,
        sort_key=lambda x: len(x.TEXT_COLUMN_NAME),
        device=DEVICE
    )

    model = RNN(input_dim=len(TEXT.vocab),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=NUM_CLASSES  # could use 1 for binary classification
    )

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            text = batch_data.TEXT_COLUMN_NAME.to(DEVICE)
            labels = batch_data.LABEL_COLUMN_NAME.to(DEVICE)

            ### FORWARD AND BACK PROP
            logits = model(text)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()

            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 50:
                print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                      f'Loss: {loss:.4f}')

        with torch.set_grad_enabled(False):
            print(f'training accuracy: '
                  f'{compute_accuracy(model, train_loader, DEVICE):.2f}%'
                  f'\nvalid accuracy: '
                  f'{compute_accuracy(model, valid_loader, DEVICE):.2f}%')

        print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min')

    print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
    print(f'Test accuracy: {compute_accuracy(model, test_loader, DEVICE):.2f}%')

    nlp = spacy.blank("en")

    print('Probability positive:')
    predict_def(model, "This is such an awesome movie, I really love it!")
