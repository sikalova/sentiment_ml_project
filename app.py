import streamlit as st
import joblib
import json
import os
# Импортируем логику
from src.utils import clean_text, MODEL_FILE, METRICS_FILE

# Настройка страницы
st.set_page_config(page_title="Анализ Отзывов", page_icon="🧠", layout="wide")

def load_assets():
    model = None
    metrics = None
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
    if os.path.exists(MODEL_FILE):
        try:
            model = joblib.load(MODEL_FILE)
        except Exception as e:
            st.error(f"Ошибка модели: {e}")
    return model, metrics

model, metrics = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.header("📊 Инфо")
    if metrics:
        st.metric("Точность", f"{metrics.get('accuracy', 0):.1%}")
        st.caption("Machine Learning: Logistic Regression")
    else:
        st.warning("Нет метрик. Запустите train.py")

# --- MAIN ---
st.title("🧠 ML Sentiment Analysis")

if not model:
    st.error("Модель не найдена! Сначала запустите обучение.")
    st.stop()

text = st.text_area("Введите отзыв (Eng):", height=150)

if st.button("Анализировать", type="primary"):
    if text.strip():
        probs = model.predict_proba([text])[0]
        pos_score = probs[1]
        
        st.write(f"Позитивность: **{pos_score*100:.1f}%**")
        st.progress(pos_score)
        
        if pos_score > 0.6:
            st.success("ПОЗИТИВНЫЙ ОТЗЫВ 😄")
        elif pos_score < 0.4:
            st.error("НЕГАТИВНЫЙ ОТЗЫВ 😡")
        else:
            st.info("НЕЙТРАЛЬНО 😐")
    else:
        st.warning("Введите текст!")
