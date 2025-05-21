# Bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import io

st.set_page_config(page_title="Dashboard Petróleo Brent", layout="wide")

@st.cache_data(show_spinner=True)
def carregar_dados():
    url = "https://datalaketech4.blob.core.windows.net/dados-ipea/cotacao_petroleo_ipea.csv"
    df = pd.read_csv(url, encoding="ISO-8859-1", delimiter=";", decimal=",")
    df.columns = df.columns.str.strip().str.lower()
    df['data'] = pd.to_datetime(df['data'], format='%d/%m/%Y')
    df['ano'] = df['data'].dt.year
    df['mes'] = df['data'].dt.month
    df['dia_da_semana'] = df['data'].dt.dayofweek
    df['media_movel_30'] = df['preco'].rolling(window=30).mean()
    df['preco_lag_1'] = df['preco'].shift(1)
    df['variacao_percentual'] = df['preco'].pct_change() * 100
    df['volatilidade_30d'] = df['preco'].rolling(window=30).std()
    df['retorno_log'] = np.log(df['preco']) - np.log(df['preco'].shift(1))
    df['retorno_log_acumulado'] = df['retorno_log'].cumsum()
    scaler = StandardScaler()
    df['preco_normalizado'] = scaler.fit_transform(df[['preco']])
    df['preco_log'] = np.log(df['preco'])
    df.bfill(inplace=True)
    return df

df = carregar_dados()

# Sidebar filtros
st.sidebar.header("Filtros")
anos = st.sidebar.multiselect("Selecione o(s) ano(s):", sorted(df['ano'].unique()), default=sorted(df['ano'].unique()))
data_min, data_max = df['data'].min(), df['data'].max()

# Validação simples: garantir que data_inicio <= data_fim
data_inicio, data_fim = st.sidebar.date_input("Intervalo de datas:", (data_min, data_max), min_value=data_min, max_value=data_max)
if data_inicio > data_fim:
    st.sidebar.error("Data de início não pode ser maior que data final. Ajuste, por favor.")

df_filtrado = df[
    (df['ano'].isin(anos)) &
    (df['data'] >= pd.to_datetime(data_inicio)) &
    (df['data'] <= pd.to_datetime(data_fim))
]

st.title("🛢️ Dashboard Interativo - Preço do Petróleo Brent")

# KPIs principais com st.metric
col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
col_kpi1.metric("Preço Atual", f"US$ {df_filtrado['preco'].iloc[-1]:.2f}")
col_kpi2.metric("Média Móvel 30d", f"US$ {df_filtrado['media_movel_30'].iloc[-1]:.2f}")
col_kpi3.metric("Volatilidade 30d", f"{df_filtrado['volatilidade_30d'].iloc[-1]:.2f}")

tabs = st.tabs(["📊 Visão Geral", "📉 Análises", "🧪 Testes Estatísticos", "🔍 Correlações", "🤖 Previsão ARIMA", "⚠️ Anomalias"])

with tabs[0]:
    st.subheader("Evolução do Preço ao Longo do Tempo")
    fig = px.line(df_filtrado, x='data', y='preco', title="Preço do Petróleo Brent")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Exibir dados brutos"):
        st.dataframe(df_filtrado[['data', 'preco', 'media_movel_30', 'variacao_percentual']].reset_index(drop=True))

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Preço vs Média Móvel (30 dias)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['preco'], mode='lines', name='Preço'))
        fig.add_trace(go.Scatter(x=df_filtrado['data'], y=df_filtrado['media_movel_30'], mode='lines', name='Média Móvel (30d)', line=dict(color='orange')))
        fig.update_layout(title="Preço vs Média Móvel", xaxis_title="Data", yaxis_title="Preço")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Distribuição do Preço")
        fig2 = px.histogram(df_filtrado, x='preco', nbins=50, marginal="box", title="Histograma e Boxplot do Preço")
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("📉 Variação Percentual Diária")
        fig3 = px.line(df_filtrado, x='data', y='variacao_percentual', title="Variação % Diária", color_discrete_sequence=['green'])
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("📉 Volatilidade Móvel (Desvio Padrão 30 dias)")
        fig4 = px.line(df_filtrado, x='data', y='volatilidade_30d', title="Volatilidade (30 dias)", color_discrete_sequence=['red'])
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("📈 Retorno Logarítmico Acumulado")
    fig5 = px.line(df_filtrado, x='data', y='retorno_log_acumulado', title="Retorno Logarítmico Acumulado", color_discrete_sequence=['purple'])
    st.plotly_chart(fig5, use_container_width=True)

with tabs[2]:
    st.subheader("📆 Decomposição Sazonal")
    serie = df_filtrado.set_index('data')['preco'].asfreq('D').interpolate()
    # Limitar decomposição para no máximo 2 anos para performance
    if len(serie) > 730:
        serie = serie[-730:]
    decomposicao = seasonal_decompose(serie, model='multiplicative', period=365)
    fig_decomp = decomposicao.plot()
    st.pyplot(fig_decomp)

    st.subheader("📐 Teste de Estacionariedade (ADF)")
    adf = adfuller(df_filtrado['preco'])
    st.write(f"Estatística ADF: {adf[0]:.4f}")
    st.write(f"p-valor: {adf[1]:.4f}")
    if adf[1] <= 0.05:
        st.success("✅ Série é estacionária (p-valor <= 0.05)")
    else:
        st.warning("⚠️ Série NÃO é estacionária (p-valor > 0.05)")

    st.subheader("📊 Autocorrelograma (ACF)")
    lag_acf = acf(df_filtrado['preco'], nlags=40)
    fig_acf = px.bar(x=range(len(lag_acf)), y=lag_acf, labels={'x':"Lag", 'y':"ACF"}, title="Função de Autocorrelação (ACF)")
    st.plotly_chart(fig_acf, use_container_width=True)

with tabs[3]:
    st.subheader("📌 Matriz de Correlação")
    corr = df_filtrado.corr(numeric_only=True)
    st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Matriz de Correlação")
    st.plotly_chart(fig_corr, use_container_width=True)

with tabs[4]:
    st.subheader("📅 Previsão de Preço com ARIMA")
    periodo_previsao = st.slider("Quantidade de dias para prever:", 7, 90, 30)
    serie_arima = df_filtrado.set_index('data')['preco'].asfreq('D').interpolate()

    with st.spinner("Ajustando modelo ARIMA..."):
        try:
            # Definindo parâmetros p,d,q; aqui uso (5,1,0) como exemplo — pode ajustar conforme seu dado
            modelo = ARIMA(serie_arima, order=(5,1,0))
            modelo_fit = modelo.fit()
            previsao_result = modelo_fit.get_forecast(steps=periodo_previsao)
            previsao_mean = previsao_result.predicted_mean
            conf_int = previsao_result.conf_int()

            future_dates = pd.date_range(serie_arima.index[-1], periods=periodo_previsao+1, freq='D')[1:]

            # Gráfico interativo da previsão
            fig_previsao = go.Figure()
            fig_previsao.add_trace(go.Scatter(x=serie_arima.index, y=serie_arima, mode='lines', name='Histórico'))
            fig_previsao.add_trace(go.Scatter(x=future_dates, y=previsao_mean, mode='lines', name='Previsão', line=dict(color='red')))
            fig_previsao.add_trace(go.Scatter(
                x=list(future_dates) + list(future_dates[::-1]),
                y=list(conf_int.iloc[:,1]) + list(conf_int.iloc[:,0][::-1]),
                fill='toself',
                fillcolor='rgba(255, 182, 193, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='Intervalo de Confiança'
            ))
            fig_previsao.update_layout(
                title=f'Previsão ARIMA para os próximos {periodo_previsao} dias',
                xaxis_title='Data',
                yaxis_title='Preço'
            )
            st.plotly_chart(fig_previsao, use_container_width=True)

            st.markdown("**Resumo do Modelo ARIMA:**")
            st.write(modelo_fit.summary())

            if len(serie_arima) > periodo_previsao:
                test = serie_arima[-periodo_previsao:]
                pred_test = modelo_fit.predict(start=test.index[0], end=test.index[-1])
                mse = mean_squared_error(test, pred_test)
                rmse = mse ** 0.5
                st.metric("RMSE da Previsão", f"{rmse:.4f}")

        except Exception as e:
            st.error(f"Erro na modelagem ARIMA: {e}")

with tabs[5]:
    st.subheader("⚠️ Detecção de Anomalias com Isolation Forest")
    n_estimators = st.slider("Número de estimadores:", 50, 500, 100)
    contamination = st.slider("Contaminação (proporção de anomalias):", 0.01, 0.1, 0.05, step=0.01)
    isolation = IsolationForest(n_estimators=n_estimators, contamination=contamination, random_state=42)
    
    # Evitar warning do SettingWithCopyWarning
    df_filtrado.loc[:, 'anomalia'] = isolation.fit_predict(df_filtrado[['preco']])
    df_filtrado['anomalia'] = df_filtrado['anomalia'].map({1: 0, -1: 1})

    # Mapear para texto para legenda mais clara
    df_filtrado['anomalia_texto'] = df_filtrado['anomalia'].map({0: 'Normal', 1: 'Anomalia'})

    fig_anomalias = px.scatter(
        df_filtrado, x='data', y='preco', 
        color='anomalia_texto',
        title="Detecção de Anomalias no Preço",
        color_discrete_map={'Normal':'blue', 'Anomalia':'red'},
        hover_data={'anomalia_texto':False, 'anomalia':True}
    )
    st.plotly_chart(fig_anomalias, use_container_width=True)

    st.write(df_filtrado[df_filtrado['anomalia'] == 1][['data', 'preco']])

# Exportar dados filtrados para CSV
st.sidebar.markdown("---")
st.sidebar.subheader("Exportar Dados")
csv_buffer = io.StringIO()
df_filtrado.to_csv(csv_buffer, index=False)
st.sidebar.download_button(
    label="⬇️ Baixar CSV dos Dados Filtrados",
    data=csv_buffer.getvalue(),
    file_name="dados_petroleo_filtrados.csv",
    mime="text/csv"
)
