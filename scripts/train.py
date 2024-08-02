import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy.data import Field, TabularDataset, BucketIterator

# シンプルなRNNモデルの定義
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# 学習データのロードと前処理
TEXT = Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
datafields = [("text", TEXT)]
train_data = TabularDataset.splits(path='./data', train='train.csv', format='csv', fields=datafields)

TEXT.build_vocab(train_data)

# モデル、損失関数、オプティマイザの定義
input_dim = len(TEXT.vocab)
hidden_dim = 256
output_dim = len(TEXT.vocab)

model = SimpleRNN(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 学習ループ
def train_model(model, data, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in data:
            optimizer.zero_grad()
            output = model(batch.text)
            loss = criterion(output, batch.text)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data)}')

train_iterator = BucketIterator(train_data, batch_size=32, device=torch.device('cpu'))
train_model(model, train_iterator, optimizer, criterion)

# モデルの保存
torch.save(model.state_dict(), './model/trained_model.pth')