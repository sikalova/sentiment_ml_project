import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.utils import clean_text, DATA_URL, MODEL_FILE, METRICS_FILE

def train_process():
    df = pd.read_csv(DATA_URL)
    df.rename(columns={'review': 'Review', 'sentiment': 'Sentiment'}, inplace=True)
    df = df.sample(50000, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['Review'], df['Sentiment'], test_size=0.2, random_state=42
    )

    my_stop_words = list(ENGLISH_STOP_WORDS)
    for word in ['not', 'no', 'never', 'none', 'neither']:
        if word in my_stop_words:
            my_stop_words.remove(word)

    model = Pipeline([
        ('tfidf', TfidfVectorizer(
            preprocessor=clean_text, 
            stop_words=my_stop_words,
            max_features=10000, 
            ngram_range=(1, 2)
        )),
        ('clf', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics_data = {
        "accuracy_test": accuracy_score(y_test, y_pred),
        "accuracy_train": accuracy_score(y_train, y_train_pred), 
        "f1_pos": report['positive']['f1-score'], 
        "f1_neg": report['negative']['f1-score'],
        "dataset_size": len(df)
    }
    
    joblib.dump(model, MODEL_FILE)
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics_data, f)
    print(f"Точность: {metrics_data['accuracy_train']:.2%}")

if __name__ == "__main__":
    train_process()
