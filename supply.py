# --- CÓDIGO PARA dashboard_tarea_grupo_X.py ---
# (Este bloque NO se ejecuta directamente en Jupyter)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import pandas as pd
from io import StringIO
import plotly.express as px
import folium
from streamlit_folium import st_folium
import geopandas as gpd
from folium.features import GeoJson, GeoJsonTooltip, GeoJsonPopup

# Configuración básica de la página
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
tab1, tab2, tab3 = st.tabs(["Resumen", "Base de datos", "Mapa"])
# Configuración simple para los gráficos
sns.set_style("whitegrid")
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
##################################################
# CARGA DE DATOS
##################################################
st.sidebar.header('Filtros del Dashboard')

token = st.sidebar.text_input(
        "Ingresa la contraseña 👇:",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        type = 'password',
    )

##############################################
# CONFIGURACIÓN DE LA BARRA LATERAL
##############################################

if token:
    url = 'https://raw.githubusercontent.com/geotigDB/Supply_DB/refs/heads/main/Base_AvanceCosecha_GEOTIG_CSV.csv'
    url_json = "https://raw.githubusercontent.com/geotigDB/TEST/refs/heads/main/RevMensual_supply_202401Act_resumen.json"
    headers = {'Authorization': f'token {token}'}
    response = requests.get(url, headers=headers)
    response_json = requests.get(url_json, headers=headers)
else:
    st.sidebar.info("Ingrese contraseña")
    response = []
    st.stop()

if response.status_code == 200:
    # Función para cargar datos con cache para mejorar rendimiento
    @st.cache_data
    def cargar_datos():
        # Carga el archivo CSV con datos macroeconómicos
        df = pd.read_csv(StringIO(response.text))
        return df
    # Cargamos los datos
    df = cargar_datos()
    df['COD_HITO_R'] = pd.to_datetime(df['Periodo de revisión']).dt.strftime('%Y %m').tolist()
    df['CODIGO'] = df['ROL']+"_"+df['COD_HITO_R']
    
    roles = df["ROL"]
    roles_lista = roles.dropna().drop_duplicates().sort_values(ascending=True).tolist()
else:
    st.error(f"Error al cargar los datos: {response.status_code} - {response.text}")
    st.stop()

if response_json.status_code == 200:
    @st.cache_data
    def cargar_json():
        # Carga el archivo CSV con datos macroeconómicos
        gdf = gpd.read_file(StringIO(response_json.text))
        return gdf
    # Cargamos los datos
    gdf = cargar_json()
    gdf['CODIGO'] = gdf['rol']+"_"+gdf['COD_HITO_R']
    
else:
    st.sidebar.info("Sin acceso a los datos.")
    st.stop()

mostrar_mapa = st.sidebar.toggle("Activar mapas?", help = "Puede afectar el rendimiento del dashboard")

opciones = ["Rol", "Periodo"]
option = st.sidebar.pills("Tipo de busqueda", opciones, selection_mode="single", default = "Rol")
      
# Selector de tipo de producto
if option == 'Rol':
    rol = st.sidebar.multiselect(
        'Seleccionar rol:',
        options=roles_lista,
        default=[],
        help="Selecciona un rol (Debe tener seleccionado un periodo de tiempo)"
    )

    df_filtrado = df[
        df['ROL'].isin(rol)
    ]

    fechas = pd.to_datetime(df_filtrado["Periodo de revisión"], errors='coerce')
    fechas = pd.to_datetime(fechas, format='%Y-%m-%d')
    fechas_unicas = fechas.dropna().drop_duplicates().sort_values(ascending=False)
    fechas_ordenadas_str = fechas_unicas.dt.strftime('%Y-%m-%d').tolist()
    all_fechas = ['Todas'] + fechas_ordenadas_str
    
    periodo_corta = []
    if rol:
        periodo_corta = st.sidebar.multiselect(
            'Periodo de corta',
            options=all_fechas,
            default=fechas_ordenadas_str,
            help="Selecciona un periodo de tiempo"
        )
        if 'Todas' in periodo_corta:
            periodo_corta = fechas_ordenadas_str
    else:
        st.sidebar.info("Primero selecciona un rol para ver los periodos donde hubo revisión.")
        periodo_corta = []  # O puedes usar None si prefieres
    
    if len(periodo_corta)==0:
        st.sidebar.info("Debe seleccionar un periodo de tiempo")
        st.stop()
        
    df_filtrado = df[
        (df['Periodo de revisión'].isin(periodo_corta)) &
        (df['ROL'].isin(rol))
    ]
else:
    fechas = pd.to_datetime(df["Periodo de revisión"], errors='coerce')
    fechas = pd.to_datetime(fechas, format='%Y-%m-%d')
    fechas_unicas = fechas.dropna().drop_duplicates().sort_values(ascending=False)
    fechas_ordenadas_str = fechas_unicas.dt.strftime('%Y-%m-%d').tolist()
    all_fechas = ['Todas'] + fechas_ordenadas_str
    
    periodo_corta = st.sidebar.multiselect(
        'Periodo de corta',
        options=all_fechas,
        default=fechas_ordenadas_str[0],
        help="Selecciona un periodo de tiempo"
    )
    if 'Todas' in periodo_corta:
            periodo_corta = fechas_ordenadas_str

    df_filtrado = df[
        (df['Periodo de revisión'].isin(periodo_corta))
    ]

    roles = df_filtrado["ROL"]
    roles_lista = roles.dropna().drop_duplicates().sort_values(ascending=True).tolist()

    rol = st.sidebar.multiselect(
        'Seleccionar rol:',
        options=roles_lista,
        default=[],
        help="Selecciona un rol (Debe tener seleccionado un periodo de tiempo)"
    )
    
    if rol:
        df_filtrado = df_filtrado[df_filtrado["ROL"].isin(rol)]
    
df_filtrado_respaldo = df_filtrado.drop(columns=["COD_HITO_R", "CODIGO"])

# Sección: Graficos basicos

with tab1:
    st.subheader('Información relevante')
    df_revisado = df_filtrado[df_filtrado['OBS'] == 'Revisado']
    df_raleado = df_filtrado[df_filtrado['OBS'].isin(['Posible cosecha incipiente o raleo', 'Revisado (con cosecha y raleo)'])]
    superficie_raleo = round(df_raleado['SUPERFICIE (ha)'].sum(), 1)
    superficie = round(df_revisado['SUPERFICIE (ha)'].sum(), 1)
    volumen = round(df_revisado['VOLUMEN PULPABLE(m3)'].sum(), 1)
    superficie_6m = round(df_filtrado['SUPERFICIE 6 MESES (ha)'].sum(), 1)
    volumen_6m = round(df_filtrado['VOLUMEN PULPABLE 6 MESES (m3)'].sum(), 1)
    
    st.markdown(f"""
    **Para el rol y periodo seleccionado las superficies y volumenes son los siguientes:**  
    """)
    c1_f1, c2_f1 = st.columns([1, 1])
    with c1_f1:
        st.markdown(f"""
        **Superficie (ha):**  
        Total cosechado (revisión mensual): {superficie}  
        Total raleado o con cosecha incipiente (revisión mensual): {superficie_raleo}  
        Total cosechado o raleado en 6 meses (ha): {superficie_6m}  
        """)
    with c2_f1:
        st.markdown(f"""
        **Volumen pulpable aproximado (m3):**  
        Total (revision mensual): {volumen}  
        Total en 6 meses: {volumen_6m}
        """)
   
    df_resumen = df_filtrado[['Periodo de revisión', 'ROL', 'DSC_ESPECIE','OBS', 'SUPERFICIE (ha)', 'VOLUMEN PULPABLE(m3)', 'SUPERFICIE 6 MESES (ha)',
       'VOLUMEN PULPABLE 6 MESES (m3)', 'COMUNA']]
        
    c1_f2, c2_f2 = st.columns([1, 1])
    if option == 'Rol' and len(rol)==1:
        fecha_completa = pd.date_range(start=df_filtrado['Periodo de revisión'].min(), end=df_filtrado['Periodo de revisión'].max(), freq='MS').strftime('%Y-%m-%d')
        df_filtrado = df_filtrado.set_index('Periodo de revisión')
        df_filtrado = df_filtrado.reindex(fecha_completa)
        df_filtrado = df_filtrado.reset_index().rename(columns={'index':'Periodo de revisión'})
        # Convertir columnas a tipos adecuados
        df_filtrado['Fecha'] = pd.to_datetime(df_filtrado['Periodo de revisión'], dayfirst=True, errors='coerce')
        df_filtrado['SUPERFICIE (ha)'] = pd.to_numeric(df_filtrado['SUPERFICIE (ha)'], errors='coerce')
        df_filtrado['VOLUMEN PULPABLE(m3)'] = pd.to_numeric(df_filtrado['VOLUMEN PULPABLE(m3)'], errors='coerce')
        df_filtrado['DSC_ESPECIE'] = df_filtrado['DSC_ESPECIE'].fillna('-')
        df_filtrado['DSC_ESPECIE'] = df_filtrado['DSC_ESPECIE'].replace('-', 'Sin información')
        
        # Filtrar filas con volumen válido
        df_sup = df_filtrado.dropna(subset=['SUPERFICIE (ha)'])
        df_vol = df_filtrado.dropna(subset=['VOLUMEN PULPABLE(m3)'])
        
        if len(df_sup)>0:
            with c1_f2:
                fig2 = px.line(
                df_sup,
                x='Periodo de revisión',
                y='SUPERFICIE (ha)',
                color='DSC_ESPECIE',
                markers=True,
                title='Superficie evaluada (ha)',
                labels={'SUPERFICIE (ha)': 'Superficie (ha)', 'Periodo de revisión': 'Mes de cosecha', 'DSC_ESPECIE': 'Especie'},
                symbol='OBS',
                hover_data=['OBS', 'SUPERFICIE (ha)']
                )

                fig2.update_traces(marker=dict(size=12))
                fig2.update_layout(xaxis_title='Fecha', yaxis_title='Superficie')
                fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.8, xanchor="center", x=0.5))
                fig2.update_xaxes(dtick="M1")
                st.plotly_chart(fig2, use_container_width=True)
            
            with c2_f2:
                fig = px.line(
                df_vol,
                x='Periodo de revisión',
                y='VOLUMEN PULPABLE(m3)',
                color='DSC_ESPECIE',
                markers=True,
                title='Volumen estimado (m3)',
                labels={'VOLUMEN PULPABLE(m3)': 'Volumen (m3)', 'Periodo de revisión': 'Mes de cosecha', 'DSC_ESPECIE': 'Especie'},
                symbol='OBS',
                hover_data=['OBS', 'SUPERFICIE (ha)']
                )

                fig.update_traces(marker=dict(size=12))
                fig.update_layout(xaxis_title='Fecha', yaxis_title='Superficie')
                fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.8, xanchor="center", x=0.5))
                fig.update_xaxes(dtick="M1")
                st.plotly_chart(fig, use_container_width=True)                 
                
    st.subheader('Tabla de datos resumida')
    st.dataframe(df_resumen, use_container_width=True, hide_index=True)

with tab2:
    st.markdown(f"""Se encuentra la base de datos filtrada al rol y periodo de revisión seleccionado  """)
    st.dataframe(df_filtrado_respaldo, hide_index=True)
    
with tab3:  
    st.markdown(
    """
    <style>
    .leaflet-popup-content {
        font-size: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    col1, col2, col3 = st.columns([1, 7, 1])
    
    if mostrar_mapa == True:
        with col2:
            periodo = df_filtrado['COD_HITO_R'].dropna().drop_duplicates().sort_values(ascending=True).tolist()
                                    
            filtro = gdf[
            gdf["rol"].isin(rol) & 
            gdf["COD_HITO_R"].isin(periodo)]
                                    
            if len(filtro)>0:
                filtro = filtro.merge(df_filtrado[['COD_HITO_R', 'DSC_ESPECIE','OBS', 'SUPERFICIE (ha)', 'VOLUMEN PULPABLE(m3)']],
                 on='COD_HITO_R', how='left')
                             
                colores = plt.cm.get_cmap("tab20", len(periodo))
            
                hito_color = {
                hito: f'#{int(colores(i)[0]*255):02x}{int(colores(i)[1]*255):02x}{int(colores(i)[2]*255):02x}'
                for i, hito in enumerate(periodo)}
            
                def estilo_por_hito(feature):
                    hito = feature["properties"]["COD_HITO_R"]
                    return {
                        "fillColor": hito_color.get(hito, "#cccccc"),
                        "color": "black",
                        "weight": 1,
                        "fillOpacity": 0.5
                    }                
                bounds = filtro.total_bounds
                m = folium.Map(location=[(bounds[1] + bounds[3])/2, (bounds[0] + bounds[2])/2], zoom_start=13)
                folium.GeoJson(filtro, style_function=estilo_por_hito,tooltip=folium.GeoJsonTooltip(fields=["rol"]),popup=GeoJsonPopup(
        fields=["rol", "COD_HITO_R", "DSC_ESPECIE", "OBS", "SUPERFICIE (ha)", "VOLUMEN PULPABLE(m3)"],
        aliases=["ROL", "Periodo de revisión", "Especie", "Observación", "Superficie (ha)", "Volumen Pulpable (m3)"],
        labels=True,
        localize=True,
        style="font-size: 10px;"
    )).add_to(m)
                st_folium(m, width=700, height=500)               
                
            else:
                st.write("No se encuentra figura asociada al rol en el periodo seleccionado")
            
    
    
