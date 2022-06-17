import spacy
import time
import random
import pandas as pd
import numpy as np
# nltk.download('punkt')
# import ntlk
# !pip uninstall torch
import torchtext
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import seaborn as sn
from torch.utils.data import Dataset, random_split
import torch.nn.functional as F
from cv2 import imshow


class SenDataSet(Dataset):
    def __init__(self, data_frame):
        self.data = data_frame

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sen = self.data['sen'][index]
        typ = self.data['isdefault'][index]
        label = torch.tensor(np.array(typ), dtype=torch.long)

        return sen, label


class RNN(torch.nn.Module):

    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        self.embedding = torch.nn.Embedding(input_dim, embedding_dim)

        self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim)

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
    y_target = []
    y_pred = []
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for i, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.float().to(device)

            logits = model(features)
            _, predicted_labels = torch.max(logits, 1)

            y_pred.extend(predicted_labels.data.cpu().numpy())
            y_target.extend(targets.data.cpu().numpy())

            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()

    return correct_pred.float() / num_examples * 100, (y_target, y_pred)


def predict_def(model, sentence):
    global NLP, DEVICE
    model.eval()
    with torch.no_grad():
        tokenized = [tok.text for tok in NLP.tokenizer(sentence)]
        indexed = [TEXT.vocab.stoi[t] for t in tokenized]
        length = [len(indexed)]
        tensor = torch.LongTensor(indexed).to(DEVICE)
        tensor = tensor.unsqueeze(1)
        length_tensor = torch.LongTensor(length)
        _, predictions = model(tensor).max(dim=1)
        return predictions.tolist()[0]


def split_data(TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT, fields):
    # dataset = SenDataSet(data_frame=df)

    dataset = torchtext.legacy.data.TabularDataset(path="defaults.csv", format="csv", skip_header=True, fields=fields)

    train_data, test_data = dataset.split(split_ratio=[TRAIN_PERCENT + VAL_PERCENT, TEST_PERCENT],
                                          random_state=random.seed(RANDOM_SEED))

    train_data, valid_data = train_data.split(
        split_ratio=[TRAIN_PERCENT, VAL_PERCENT],
        random_state=random.seed(RANDOM_SEED))

    return train_data, valid_data, test_data


def visualize_model(model, test_loader, class_names, num_images=6):
    global DEVICE

    model.eval()
    images_so_far = 1
    corrects = 0
    wrongs = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for j in range(inputs.size()[0]):
                # ****status = |pred - actual| *****
                status = int(abs(preds[j] - labels[j]))  # Not-Default = 0, Default = 1
                if corrects + wrongs >= num_images:
                    # model.train(mode=was_training)
                    return

                if status == 0 and corrects < num_images // 2:  # Correct
                    corrects += 1
                elif status == 1 and wrongs < num_images // 2:  # Incorrect
                    wrongs += 1
                else:
                    continue

                ax = plt.subplot(num_images // 2, 2, images_so_far)
                images_so_far += 1

                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}, is_correct = {str(status == 0)}')
                imshow(inputs.cpu().data[j])


if __name__ == "__main__":
    class_names = {0: "Not-Default", 1: "Default"}
    NLP = spacy.blank("en")
    RANDOM_SEED = 58
    torch.manual_seed(RANDOM_SEED)
    VOCABULARY_SIZE = 2000
    LEARNING_RATE = .005
    BATCH_SIZE = 25
    NUM_EPOCHS = 30
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    EMBEDDING_DIM = 128
    HIDDEN_DIM = 256
    NUM_CLASSES = 2  # Default or Not-Default

    TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT = .7, .1, .2
    TEXT = torchtext.legacy.data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
    LABEL = torchtext.legacy.data.LabelField(dtype=torch.long)
    fields = [('sen', TEXT), ('isdefault', LABEL)]

    train_data, valid_data, test_data = split_data(TRAIN_PERCENT, VAL_PERCENT, TEST_PERCENT, fields)

    print(f'Num Train: {len(train_data)}')
    print(f'Num Validation: {len(valid_data)}')
    print(f'Num Test: {len(test_data)}')

    TEXT.build_vocab(train_data, max_size=VOCABULARY_SIZE)
    LABEL.build_vocab(train_data)

    # print(vars(train_data.examples[0]))
    # print(TEXT.vocab.freqs.most_common(20))

    train_loader, valid_loader, test_loader = torchtext.legacy.data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_within_batch=False,
        sort_key=lambda x: len(x.sen),
        device=DEVICE
    )

    model = RNN(input_dim=len(TEXT.vocab),
                embedding_dim=EMBEDDING_DIM,
                hidden_dim=HIDDEN_DIM,
                output_dim=NUM_CLASSES
                )
    # model.load_state_dict(torch.load('TRAIN_98.1915__VALID53.7313.pt'))
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_time = time.time()

    loss_data = []
    train_acc = []
    valid_acc = []
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_idx, batch_data in enumerate(train_loader):

            text = batch_data.sen.to(DEVICE)
            labels = batch_data.isdefault.to(DEVICE)

            ### FORWARD AND BACK PROP
            logits = model(text)
            loss = F.cross_entropy(logits, labels)
            loss_data.append(float(loss.item()))
            optimizer.zero_grad()

            loss.backward()

            ### UPDATE MODEL PARAMETERS
            optimizer.step()

            ### LOGGING
            if not batch_idx % 50:
                print(f'Epoch: {epoch + 1:03d}/{NUM_EPOCHS:03d} | '
                      f'Batch {batch_idx:03d}/{len(train_loader):03d} | '
                      f'Loss: {loss:.4f}')

        with torch.no_grad():
            cur_train_acc = round(float(compute_accuracy(model, train_loader, DEVICE)[0]), 4)
            cur_valid_acc = round(float(compute_accuracy(model, valid_loader, DEVICE)[0]), 4)
            train_acc.append(cur_train_acc)
            valid_acc.append(cur_valid_acc)
            print(f'training accuracy: '
                  f'{cur_train_acc:.2f}%'
                  f'\nvalid accuracy: '
                  f'{cur_valid_acc:.2f}%')

        print(f'Time elapsed: {(time.time() - start_time) / 60:.2f} min\n')

    print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')

    test_acc, ys = compute_accuracy(model, test_loader, DEVICE)
    print(f'Test accuracy: {test_acc:.2f}%')

    cf_matrix = confusion_matrix(ys[0], ys[1])
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[str(i) for i in class_names.values()],
                         columns=[str(i) for i in class_names.values()])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
    plt.savefig('conf.png')

    max_train_acc = max(train_acc)
    max_valid_acc = max(valid_acc)
    min_loss_val = min(loss_data)

    plt.style.use('fivethirtyeight')
    plt.title(f'Cross Entropy Loss, minimum: {min_loss_val}')
    plt.plot(loss_data)
    plt.show()
    plt.figure(figsize=(15, 15))
    plt.title(f'Train & Valid Accuracy')
    plt.plot(train_acc)
    plt.plot(valid_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend([f'Train with max: {max_train_acc}', f'Valid with max: {max_valid_acc}'], loc='best')
    plt.show()
    torch.save(model.state_dict(), f'TRAIN_{max_train_acc}__VALID{max_valid_acc}.pt')
    stringtest = "When it is rainning outside you need an umbrella."
    pred = predict_def(model, stringtest)
    print(f'{stringtest}\nis type of: {class_names[pred]}')
