import re

# Константы
DATA_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
MODEL_FILE = "sentiment_model.pkl"
METRICS_FILE = "metrics.json"

# Вот она, та самая функция, которую python не может найти
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
