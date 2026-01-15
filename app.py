import streamlit as st
import joblib
import json
import pandas as pd
import os
from src.utils import clean_text, MODEL_FILE, METRICS_FILE


st.set_page_config(page_title="Анализ Отзывов", layout="wide")

def load_assets():
    model = None
    metrics = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f: metrics = json.load(f)
    if os.path.exists(MODEL_FILE):
         model = joblib.load(MODEL_FILE)
    return model, metrics

def get_feature_importance(model, text):
    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['clf']
    vector = tfidf.transform([text])
    feature_names = tfidf.get_feature_names_out()
    indices = vector.nonzero()[1]
    
    data = []
    for idx in indices:
        weight = clf.coef_[0][idx] * vector[0, idx]
        data.append({'Слово': feature_names[idx], 'Вес': weight})
    
    return pd.DataFrame(data).sort_values(by='Вес', ascending=False)

model, metrics = load_assets()

with st.sidebar:
    st.header("Инфо")
    if metrics:
        st.metric("Точность", f"{metrics.get('accuracy_train', 0):.1%}")
        st.caption("Logistic Regression + N-grams")

st.title("ML Sentiment Analysis")

if not model:
    st.error("Модель не найдена")
    st.stop()

text = st.text_area("Введите отзыв (Eng):", height=150)

if st.button("Анализировать", type="primary"):
    if text.strip():
        probs = model.predict_proba([text])[0]
        pos_score = probs[1]
        
        st.write(f"Позитивность: **{pos_score*100:.1f}%**")
        st.progress(pos_score)
        
        if pos_score > 0.7: st.success("ПОЗИТИВНЫЙ ОТЗЫВ")
        elif pos_score < 0.3: st.error("НЕГАТИВНЫЙ ОТЗЫВ")
        else: st.info("НЕЙТРАЛЬНО")
        st.subheader("Влияние слов")
        df_features = get_feature_importance(model, text)
        if not df_features.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.write("Тянут в позитив")
                st.dataframe(df_features[df_features['Вес'] > 0].head(5), hide_index=True)
            with col2:
                st.write("Тянут в негатив")
                st.dataframe(df_features[df_features['Вес'] < 0].sort_values('Вес').head(5), hide_index=True)
    else:
        st.warning("Введите текст")
