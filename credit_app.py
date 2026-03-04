import streamlit as st
import pandas as pd
import joblib

# ─── Конфигурация страницы ─────────────────────────────────────
st.set_page_config(
    page_title="Кредитный скоринг",
    page_icon="🏦",
    layout="wide"
)

# ─── Загрузка модели (кэшируется — загружается один раз!) ──────
@st.cache_resource
def load_model():
    return joblib.load('credit_scoring_model.joblib')

model = load_model()

# ─── Заголовок ────────────────────────────────────────────────
st.title("🏦 Система кредитного скоринга")
st.markdown("""
Введите данные клиента и получите мгновенное решение по кредиту.  
*Демо-модель на основе XGBoost | Казахстан 🇰🇿*
""")
st.divider()

# ─── Форма ввода в 2 колонки ───────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("👤 Личные данные")
    city = st.selectbox(
        "🏙️ Город",
        ['Алматы', 'Астана', 'Шымкент', 'Актобе', 'Атырау']
    )
    employment_type = st.selectbox(
        "💼 Тип занятости",
        ['Наёмный', 'ИП', 'Госслужба', 'Безработный', 'Фриланс']
    )
    age = st.number_input(
        "🎂 Возраст", min_value=21, max_value=65, value=34, step=1
    )
    has_property = st.checkbox("🏠 Есть недвижимость")
    num_dependents = st.number_input(
        "👨‍👧 Количество иждивенцев", min_value=0, max_value=6, value=1, step=1
    )

with col_right:
    st.subheader("💰 Финансовые данные")
    income = st.number_input(
        "💵 Ежемесячный доход (₸)",
        min_value=80_000, max_value=2_000_000,
        value=350_000, step=10_000
    )
    credit_amount = st.number_input(
        "🏧 Сумма кредита (₸)",
        min_value=100_000, max_value=15_000_000,
        value=2_000_000, step=50_000
    )
    credit_history_years = st.slider(
        "📅 Кредитная история (лет)",
        min_value=0.0, max_value=25.0, value=5.0, step=0.5
    )
    monthly_expenses = st.number_input(
        "🧾 Ежемесячные расходы (₸)",
        min_value=50_000, max_value=500_000,
        value=150_000, step=5_000
    )

st.divider()

# ─── Кнопка предсказания ───────────────────────────────────────
if st.button("🔍 Проверить клиента", type="primary", use_container_width=True):

    # Формируем DataFrame для модели
    input_data = pd.DataFrame([{
        'city':                  city,
        'employment_type':       employment_type,
        'age':                   age,
        'income':                income,
        'credit_amount':         credit_amount,
        'credit_history_years':  credit_history_years,
        'has_property':          int(has_property),
        'num_dependents':        num_dependents,
        'monthly_expenses':      monthly_expenses
    }])

    # Предсказание
    prediction    = model.predict(input_data)[0]
    probability   = model.predict_proba(input_data)[0][1]

    # ─── Результат ──────────────────────────────────────────────
    if prediction == 1:
        st.success("✅ КРЕДИТ ОДОБРЕН")
    else:
        st.error("❌ КРЕДИТ ОТКЛОНЁН")

    # ─── Метрики ────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "🎯 Вероятность одобрения",
        f"{probability:.1%}"
    )
    m2.metric(
        "💳 Сумма кредита",
        f"{credit_amount:,} ₸"
    )
    m3.metric(
        "💵 Ежемесячный доход",
        f"{income:,} ₸"
    )

    # ─── Прогресс-бар нагрузки ──────────────────────────────────
    st.subheader("📊 Кредитная нагрузка")
    # Ориентировочный платёж (3 года, ~12%)
    monthly_payment = credit_amount * 0.033
    load_ratio = monthly_payment / income
    st.progress(min(load_ratio, 1.0))
    st.write(f"Ежемесячный платёж: ~{monthly_payment:,.0f} ₸ ({load_ratio:.1%} от дохода)")

    if load_ratio > 0.6:
        st.warning("⚠️ Высокая долговая нагрузка (более 60% дохода)")
    elif load_ratio > 0.4:
        st.info("ℹ️ Умеренная нагрузка (40–60% дохода)")
    else:
        st.success("✅ Комфортная нагрузка (менее 40% дохода)")

    # ─── Предупреждения (hard_reject логика) ────────────────────
    if employment_type == 'Безработный':
        st.error("🚫 Автоматический отказ: статус 'Безработный'")
    if income / credit_amount < 0.05:
        st.error("🚫 Автоматический отказ: доход < 5% от суммы кредита")

    # ─── Детали решения ─────────────────────────────────────────
    with st.expander("📋 Детали введённых данных"):
        display_df = input_data.T.reset_index()
        display_df.columns = ["Параметр", "Значение"]
        display_df["Значение"] = display_df["Значение"].astype(str)
        st.dataframe(display_df, width='stretch')

# ─── Подвал ──────────────────────────────────────────────────
st.caption("🏦 Кредитный скоринг | XGBoost Pipeline | Курс ML, Неделя 19 | 🇰🇿")
