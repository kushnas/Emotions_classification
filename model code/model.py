import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\w*\d\w*', '', text)
    tokens = word_tokenize(text)
    # Remove stopwords and stem tokens
    cleaned_tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    # Join the tokens back into a single string
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class CNNModel(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        super(CNNModel, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = torch.nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.pool = torch.nn.AdaptiveMaxPool1d(1)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        conv_out = self.conv1(embedded.transpose(1, 2))
        pooled = self.pool(conv_out).squeeze(2)
        fc1_out = torch.relu(self.fc1(pooled))
        output = torch.softmax(self.fc2(fc1_out), dim=1)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
