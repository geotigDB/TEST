# Importamos las bibliotecas necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

##########################################################
# CONFIGURACIÓN DEL DASHBOARD
##########################################################

# Configuración básica de la página
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

# Configuración simple para los gráficos
sns.set_style("whitegrid")

##################################################
# CARGA DE DATOS
##################################################

# Función para cargar datos con cache para mejorar rendimiento
@st.cache_data
def cargar_datos():
    # Carga el archivo CSV con datos macroeconómicos
    df = pd.read_csv("USMacroG_v2.csv")
    # Usamos solo el año como referencia temporal
    df["Fecha"] = df["Year"]
    return df

# Cargamos los datos
df = cargar_datos()

##############################################
# CONFIGURACIÓN DE LA BARRA LATERAL
##############################################

# Simplificamos la barra lateral con solo lo esencial
st.sidebar.header('Filtros del Dashboard')

# Selector de rango de años
anio_inicio, anio_fin = st.sidebar.slider(
    'Rango de Años', 
    int(df['Year'].min()), 
    int(df['Year'].max()), 
    (1950, 2000)
)

# Selector de componentes del PIB (solo para el gráfico de área)
componentes_pib = st.sidebar.multiselect(
    'Componentes del PIB', 
    options=['consumption', 'invest', 'government'],
    default=['consumption', 'invest'],
    help="Selecciona los componentes para visualizar en el gráfico de área"
)

# ##################################################
# # FILTRADO DE DATOS
# ##################################################
df[df['Year'] == 2000].head()
# Filtramos los datos según el rango de años seleccionado
df_filtrado = df[(df['Year'] >= anio_inicio) & (df['Year'] <= anio_fin)]

# Título principal del dashboard
st.title('📊 Dashboard Macroeconómico')
st.write(f"Datos económicos de EE.UU. ({anio_inicio}-{anio_fin})")

# #######################################################
# # SECCIÓN DE MÉTRICAS (PRIMERA FILA)
# #######################################################

# Mostramos métricas del último trimestre disponible
st.subheader("Último cuarter registrado")

# Obtenemos los datos del último trimestre
ultimo = df_filtrado.iloc[-1]
fecha_ultimo = f"Q{int(ultimo['Quarter'])} {int(ultimo['Year'])}"

# Creamos tres columnas para las métricas principales
col1, col2, col3 = st.columns(3)

# Mostramos las métricas con formato adecuado
col1.metric("PIB (GDP)", f"${ultimo['gdp']:,.0f} Bill", help=f"Producto Interno Bruto en {fecha_ultimo}")
col2.metric("Desempleo", f"{ultimo['unemp']:.1f}%", help=f"Tasa de desempleo en {fecha_ultimo}")
col3.metric("Inflación", f"{ultimo['inflation']:.1f}%", help=f"Tasa de inflación en {fecha_ultimo}")

#########################################################
# SECCIÓN DE GRÁFICOS (SEGUNDA FILA)
#########################################################

# Sección: Composición del PIB
st.subheader('Composición del PIB')

# Dividimos la pantalla en dos columnas (proporción 7:3)
c1_f1, c2_f1 = st.columns((7, 3))

# Columna 1: Gráfico de área para componentes del PIB
with c1_f1:
    if componentes_pib:
        # Creamos un gráfico de área para mostrar la evolución temporal
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Graficamos los componentes seleccionados
        # Agrupamos por año para simplificar
        df_anual = df_filtrado.groupby('Year')[componentes_pib].mean()
        df_anual.plot.area(
            ax=ax
        )
        
        # Etiquetas y cuadrícula
        ax.set_ylabel("Billones $")
        ax.set_title("Evolución de componentes del PIB")
        # ax.grid(True, alpha=0.3)
        
        # Mostramos el gráfico en Streamlit
        st.pyplot(fig)
    else:
        st.info("Selecciona al menos un componente del PIB")

# Columna 2: Gráfico de torta para distribución trimestral
with c2_f1:
    if componentes_pib:
        # Calculamos el promedio por trimestre
        pie_data = df_filtrado.groupby('Quarter')[componentes_pib].mean().sum(axis=1)
        
        # Creamos el gráfico de torta
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            pie_data, 
            labels=[f'Trim {int(q)}' for q in pie_data.index],
            autopct='%1.1f%%',  # Mostrar porcentajes
            # colors=sns.color_palette('viridis', 4)  # Usar misma paleta que gráfico de área
        )
        
        # Título del gráfico
        ax.set_title('Distribución por trimestre')
        
        # Mostramos el gráfico en Streamlit
        st.pyplot(fig)
    else:
        st.info("Selecciona al menos un componente del PIB")

###################################################
# SECCIÓN DE ANÁLISIS ECONÓMICO (TERCERA FILA)
###################################################

# Sección: Análisis de Tendencias Económicas
st.subheader('Análisis Económico')
st.write('Visualización de tendencias y relaciones entre indicadores económicos')

# Creamos una fila con dos gráficos: PIB y Variables Porcentuales
c1_f2, c2_f2 = st.columns(2)

# Diccionario para traducir nombres de variables
nombres = {
    'gdp': 'PIB', 
    'unemp': 'Desempleo', 
    'inflation': 'Inflación'
}

# Columna 1: Gráfico exclusivo para el PIB
with c1_f2:
    st.write("### Evolución del PIB")
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Graficamos el PIB agrupado por año
    df_anual_pib = df_filtrado.groupby('Year')['gdp'].mean().reset_index()
    sns.lineplot(
        data=df_anual_pib, 
        x='Year', 
        y='gdp', 
        color='#1f77b4',
        ax=ax
    )
    
    # Configuración del gráfico
    ax.set_ylabel('Billones $')
    ax.set_title("Tendencia del Producto Interno Bruto")
    ax.grid(True, alpha=0.3)
    
    # Mostramos el gráfico
    st.pyplot(fig)
    st.write("*El gráfico muestra la evolución del PIB a lo largo del tiempo, permitiendo identificar ciclos económicos y tendencias de crecimiento.*")

# Columna 2: Gráfico para variables porcentuales (Desempleo e Inflación)
with c2_f2:
    st.write("### Desempleo e Inflación")
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Colores para cada variable
    colores = {'unemp': '#ff7f0e', 'inflation': '#2ca02c'}
    
    # Graficamos las variables porcentuales agrupadas por año
    df_anual_vars = df_filtrado.groupby('Year')[['unemp', 'inflation']].mean().reset_index()
    for var in ['unemp', 'inflation']:
        sns.lineplot(
            data=df_anual_vars, 
            x='Year', 
            y=var, 
            label=nombres.get(var),
            color=colores.get(var),
            ax=ax
        )
    
    # Configuración del gráfico
    ax.set_ylabel('Porcentaje (%)')
    ax.set_title("Tendencias de Desempleo e Inflación")
    ax.legend()
    ax.grid(True, alpha=0)
    
    # Mostramos el gráfico
    st.pyplot(fig)
    st.write("*Comparación entre tasas de desempleo e inflación, útil para analizar posibles compensaciones en política económica.*")

########################################################
# SECCIÓN DE ANÁLISIS DE RELACIONES (CUARTA FILA)
########################################################

# Nueva fila: Gráfico de dispersión (Inflación vs Desempleo) e Histograma
c1_f3, c2_f3 = st.columns(2)

# Gráfico de dispersión: Desempleo vs Inflación (Curva de Phillips)
with c1_f3:
    st.write("### Relación Inflación-Desempleo")
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Crear gráfico de dispersión
    scatter = ax.scatter(
        df_filtrado['unemp'], 
        df_filtrado['inflation'],
        alpha=0.7,
        c=df_filtrado['Year'],  # Colorear por año
        cmap='viridis'
    )
    
    # # Añadir línea de tendencia
    # z = np.polyfit(df_filtrado['unemp'], df_filtrado['inflation'], 1)
    # p = np.poly1d(z)
    # ax.plot(df_filtrado['unemp'], p(df_filtrado['unemp']), "r--", alpha=0.7)
    
    # Configuración del gráfico
    ax.set_xlabel('Tasa de Desempleo (%)')
    ax.set_ylabel('Tasa de Inflación (%)')
    ax.set_title('Curva de Phillips: Inflación vs Desempleo')
    ax.grid(True, alpha=0.3)
    
    # Mostrar gráfico
    st.pyplot(fig)
    st.write("*Explora la relación entre inflación y desempleo. La teoría de la Curva de Phillips sugiere una relación inversa entre ambas variables.*")

# Histograma de Inflación
with c2_f3:
    st.write("### Distribución de la Inflación")
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Crear histograma
    ax.hist(
        df_filtrado['inflation'],
        bins=15,
        color='#2ca02c',
        alpha=0.7,
        edgecolor='black'
    )
    
    # Configuración del gráfico
    ax.set_xlabel('Tasa de Inflación (%)')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de la Inflación')
    ax.grid(True, alpha=0.3)
    
    # Mostrar línea vertical en la media
    media = df_filtrado['inflation'].mean()
    ax.axvline(media, color='red', linestyle='dashed', linewidth=1, label=f'Media: {media:.2f}%')
    ax.legend()
    
    # Mostrar gráfico
    st.pyplot(fig)
    st.write("*Visualiza la distribución de las tasas de inflación en el período seleccionado, mostrando su frecuencia y dispersión.*")

# Pie de página simple
st.markdown("---")
st.caption("Dashboard Macroeconómico Simple | Datos: USMacroG_v2.csv")
