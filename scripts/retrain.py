import torch
from train import SimpleRNN, TEXT, train_model

# 学習済みモデルのロード
model = SimpleRNN(len(TEXT.vocab), 256, len(TEXT.vocab))
model.load_state_dict(torch.load('./model/trained_model.pth'))

# 再学習のためのデータロード
train_data = TabularDataset.splits(path='./data', train='train.csv', format='csv', fields=[("text", TEXT)])[0]
train_iterator = BucketIterator(train_data, batch_size=32, device=torch.device('cpu'))

# オプティマイザと損失関数の再定義
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# モデルの再学習
train_model(model, train_iterator, optimizer, criterion)

# 再学習済みモデルの保存
torch.save(model.state_dict(), './model/trained_model.pth')