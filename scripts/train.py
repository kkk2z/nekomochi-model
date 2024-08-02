import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader

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

# データセットのクラス
class TextDataset(Dataset):
    def __init__(self, file_paths):
        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.vocab = None
        self.data = self.load_data(file_paths)
        
    def load_data(self, file_paths):
        texts = []
        for file_path in file_paths:
            with open(file_path, 'r') as f:
                texts.extend(f.readlines())
        return texts
    
    def build_vocab(self):
        tokenized_texts = [self.tokenizer(text) for text in self.data]
        self.vocab = build_vocab_from_iterator(tokenized_texts, specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        tokens = self.tokenizer(text)
        token_indices = torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)
        return token_indices

# データセットの準備
data_files = ['./data/novel1.txt', './data/novel2.txt']  # 例
dataset = TextDataset(data_files)
dataset.build_vocab()

# データローダーの設定
def collate_fn(batch):
    lengths = [len(x) for x in batch]
    padded_batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return padded_batch, lengths

data_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# モデル、損失関数、オプティマイザの定義
input_dim = len(dataset.vocab)
hidden_dim = 256
output_dim = len(dataset.vocab)

model = SimpleRNN(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 学習ループ
def train_model(model, data_loader, optimizer, criterion, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch, lengths in data_loader:
            optimizer.zero_grad()
            output = model(batch.float())
            # ラベルの準備（今回は例示のため、バッチのインデックスを使用）
            labels = batch[:, 1:].contiguous().view(-1)
            predictions = output.view(-1, len(dataset.vocab))
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(data_loader)}')

train_model(model, data_loader, optimizer, criterion)

# モデルの保存
torch.save(model.state_dict(), './model/trained_model.pth')