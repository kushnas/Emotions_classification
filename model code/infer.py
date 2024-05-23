import hydra
from omegaconf import DictConfig
import mlflow
import gdown
import torch
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model
from model import clean_text
from model import TextDataset
from model import CNNModel

@hydra.main(version_base=None, config_name="config.yaml")
def infer_model(cfg: DictConfig):
    mlflow.set_tracking_uri(" http://127.0.0.1:5000")
    mlflow.set_experiment("infer_detecting_emotions")
    output = 'data.csv'
    gdown.download(cfg.data.url, output, quiet=False)

    df = pd.read_csv(output)
    df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates()
    unique_review = df['text'].unique()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    df['cleaned_text'] = df['text'].apply(clean_text)
    df['cleaned_text'] = df['cleaned_text'].str.replace("http", "").str.replace("href", "").str.replace("img","").str.replace("irc", "")
    unique_review = df['cleaned_text'].unique()

    mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
    df['Emotion'] = df['label'].map(mapping)

    X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=cfg.data.test_size, random_state=cfg.data.random_state)


    # Предобработка текста для модели
    tokenizer = Tokenizer(num_words=cfg.model.vocab_size)
    tokenizer.fit_on_texts(X_train)
    X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=cfg.model.maxlen, padding=cfg.model.padding)
    X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=cfg.model.maxlen, padding=cfg.model.padding)

    # Преобразование данных в тензоры PyTorch
    X_train_tensor = torch.tensor(X_train_padded)
    X_test_tensor = torch.tensor(X_test_padded)
    y_train_tensor = torch.tensor(y_train.values)
    y_test_tensor = torch.tensor(y_test.values)

    train_dataset = TextDataset(X_train_tensor, y_train_tensor)
    test_dataset = TextDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size)

    model = CNNModel(vocab_size=cfg.model.vocab_size, embedding_dim=cfg.model.embedding_dim,output_dim=cfg.model.embedding_dim)
    model.load_state_dict(torch.load("cnn_model.pth"))

    # Перевод модели в режим оценки (evaluation mode)
    model.eval()

    # Оценка качества модели на тестовой выборке
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy on test set: {accuracy}')

if __name__ == "__main__":
    infer_model()




