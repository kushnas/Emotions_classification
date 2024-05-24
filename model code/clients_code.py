import hydra
import pickle
from omegaconf import DictConfig
import torch
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model
from model import clean_text
from model import CNNModel

@hydra.main(version_base=None, config_name="config.yaml")
def clients_code(cfg: DictConfig) -> None:
    with open('tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    # Загрузите вашу модель и tokenizer
    model = CNNModel(vocab_size=cfg.model.vocab_size, embedding_dim=cfg.model.embedding_dim,output_dim=cfg.model.embedding_dim)
    model.load_state_dict(torch.load("cnn_model.pth"))
    model.eval()

    def predict_emotion(text):
        cleaned_text = clean_text(text)
        cleaned_text = cleaned_text.replace("http", "").replace("href", "").replace("img", "").replace("irc", "")
        tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
        padded_text = pad_sequences(tokenized_text, maxlen=cfg.model.maxlen, padding=cfg.model.padding)
        input_tensor = torch.tensor(padded_text)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted_class = torch.max(output, 1)
            mapping = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
            predicted_emotion = mapping[predicted_class.item()]

        return predicted_emotion

    while True:
        user_input = input("Введите текст для предсказания эмоции (для выхода введите 'exit'): ")

        if user_input.lower() == 'exit':
            break

        predicted_emotion = predict_emotion(user_input)
        print(f"Предсказанная эмоция: {predicted_emotion}")

if __name__ == "__main__":
    clients_code()


