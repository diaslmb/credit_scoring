import streamlit as st
import pandas as pd
import joblib

# ===== КОНФИГУРАЦИЯ СТРАНИЦЫ =====
# Должна быть первой командой Streamlit!
st.set_page_config(
    page_title="Кредитный скоринг",
    page_icon="🏦",
    layout="centered"
)

# ===== ЗАГРУЗКА МОДЕЛИ =====
# @st.cache_resource: модель загружается один раз и кэшируется
# При каждом изменении UI скрипт перезапускается, но модель не перезагружается
@st.cache_resource
def load_model():
    return joblib.load('credit_scoring_model.joblib')

model = load_model()

# ===== ЗАГОЛОВОК =====
st.title("🏦 Кредитный скоринг")
st.write("Введите данные клиента — модель предскажет решение по кредиту")
st.divider()

# ===== ФОРМА ВВОДА =====
st.subheader("📋 Данные клиента")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "Город",
        options=["Алматы", "Астана", "Шымкент", "Караганда", "Актобе"]
    )
    age = st.slider("Возраст", min_value=18, max_value=70, value=35)
    income = st.number_input(
        "Ежемесячный доход (₸)",
        min_value=80000,
        max_value=2000000,
        value=450000,
        step=10000
    )
    credit_amount = st.number_input(
        "Сумма кредита (₸)",
        min_value=100000,
        max_value=15000000,
        value=3000000,
        step=100000
    )

with col2:
    employment_type = st.selectbox(
        "Тип занятости",
        options=["Наёмный", "ИП", "Госслужба", "Безработный", "Фриланс"]
    )
    credit_history_years = st.slider(
        "Кредитная история (лет)",
        min_value=0.0,
        max_value=25.0,
        value=5.0,
        step=0.5
    )
    has_property_str = st.selectbox(
        "Наличие недвижимости",
        options=["Нет", "Да"]
    )
    num_dependents = st.number_input(
        "Количество иждивенцев",
        min_value=0,
        max_value=6,
        value=2
    )

monthly_expenses = st.slider(
    "Ежемесячные расходы (₸)",
    min_value=30000,
    max_value=500000,
    value=150000,
    step=5000
)

st.divider()

# ===== КНОПКА ПРЕДСКАЗАНИЯ =====
if st.button("🔍 Проверить клиента", type="primary", use_container_width=True):
    
    # Конвертируем "Да"/"Нет" в 1/0
    has_property = 1 if has_property_str == "Да" else 0
    
    # Собираем данные в DataFrame (именно такой формат ожидает наш Pipeline)
    client_data = pd.DataFrame([{
        'city': city,
        'employment_type': employment_type,
        'age': age,
        'income': income,
        'credit_amount': credit_amount,
        'credit_history_years': credit_history_years,
        'has_property': has_property,
        'num_dependents': num_dependents,
        'monthly_expenses': monthly_expenses
    }])
    
    # Делаем предсказание
    prediction = model.predict(client_data)[0]
    probabilities = model.predict_proba(client_data)[0]
    prob_reject = probabilities[0]
    prob_approve = probabilities[1]
    
    # ===== РЕЗУЛЬТАТ =====
    st.subheader("📊 Результат")
    
    if prediction == 1:
        st.success("✅ КРЕДИТ ОДОБРЕН")
    else:
        st.error("❌ КРЕДИТ ОТКЛОНЁН")
    
    # Три метрики
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Вероятность одобрения", f"{prob_approve*100:.1f}%")
    with m2:
        st.metric("Сумма кредита", f"{credit_amount:,} ₸")
    with m3:
        st.metric("Ежемесячный доход", f"{income:,} ₸")
    
    # Прогресс-бар вероятности одобрения
    st.write("**Уверенность модели:**")
    st.progress(float(prob_approve))
    
    st.divider()
    
    # ===== АНАЛИЗ ФИНАНСОВОЙ НАГРУЗКИ =====
    st.subheader("📈 Анализ финансовой нагрузки")
    
    annual_income = income * 12
    debt_ratio = credit_amount / annual_income
    
    if debt_ratio > 0.5:
        st.warning(
            f"⚠️ Кредит составляет {debt_ratio*100:.1f}% годового дохода — "
            f"высокая нагрузка (более 50%)"
        )
    elif debt_ratio > 0.3:
        st.info(
            f"ℹ️ Кредит составляет {debt_ratio*100:.1f}% годового дохода — "
            f"умеренная нагрузка (30–50%)"
        )
    else:
        st.success(
            f"✅ Кредит составляет {debt_ratio*100:.1f}% годового дохода — "
            f"комфортная нагрузка (до 30%)"
        )
    
    # ===== ПОДРОБНЫЕ ДАННЫЕ =====
    with st.expander("🔎 Подробные данные клиента"):
        st.dataframe(client_data)
        st.write(f"**Вероятность одобрения:** {prob_approve*100:.2f}%")
        st.write(f"**Вероятность отказа:** {prob_reject*100:.2f}%")

# ===== ПОДВАЛ =====
st.divider()
st.caption("🏦 Система кредитного скоринга | Модель: XGBoost Pipeline | Курс ML & DS, 2025")
