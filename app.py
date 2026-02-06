import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import base64  # Librería clave para los GIFs

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Ing. Condoi - UCE",
    page_icon="🦅",
    layout="wide"
)

if not api_key:
    st.error("❌ ERROR: No encontré la API Key. Revisa tu archivo .env")
    st.stop()

genai.configure(api_key=api_key)

PDF_FOLDER = 'archivos_pdf'
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

# --- RECURSOS GRÁFICOS ---
LOGO_URL = "UCELOGO.png"
AVATAR_URL = "avatar_uce.gif"          # GIF animado 1 (Chat)
AVATAR_URL_GESTION = "avatar_uce2.gif" # GIF animado 2 (Gestión)

# --- 2. FUNCIONES DE LÓGICA (Backend) ---

def get_img_as_base64(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def conseguir_modelo_disponible():
    try:
        modelos = list(genai.list_models())
        modelos_chat = [m for m in modelos if 'generateContent' in m.supported_generation_methods]
        if not modelos_chat: return None, "Sin modelos compatibles."
        nombres = [m.name for m in modelos_chat]
        preferidos = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        for pref in preferidos:
            if pref in nombres: return pref, pref
        return nombres[0], nombres[0]
    except Exception as e:
        return None, str(e)

def guardar_archivo(uploaded_file):
    ruta = os.path.join(PDF_FOLDER, uploaded_file.name)
    with open(ruta, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return uploaded_file.name

def eliminar_archivo(nombre_archivo):
    ruta = os.path.join(PDF_FOLDER, nombre_archivo)
    if os.path.exists(ruta):
        os.remove(ruta)

@st.cache_resource
def leer_pdfs_locales():
    textos, fuentes = [], []
    if not os.path.exists(PDF_FOLDER): return [], []

    archivos = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not archivos: return [], []
    
    for archivo in archivos:
        try:
            ruta_completa = os.path.join(PDF_FOLDER, archivo)
            reader = PyPDF2.PdfReader(ruta_completa)
            for i, page in enumerate(reader.pages):
                texto = page.extract_text()
                if texto:
                    texto_limpio = re.sub(r'\s+', ' ', texto).strip()
                    chunks = [texto_limpio[i:i+1000] for i in range(0, len(texto_limpio), 800)]
                    for chunk in chunks:
                        textos.append(chunk)
                        fuentes.append(f"{archivo} (Pág {i+1})")
        except: pass
    return textos, fuentes

def buscar_informacion(pregunta, textos, fuentes):
    if not textos: return ""
    try:
        vectorizer = TfidfVectorizer().fit_transform(textos + [pregunta])
        vectors = vectorizer.toarray()
        cosine_sim = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1]).flatten()
        indices = cosine_sim.argsort()[:-5:-1]
        contexto = ""
        hay_relevancia = False
        for i in indices:
            if cosine_sim[i] > 0.15:
                hay_relevancia = True
                contexto += f"\n- {textos[i]} [Fuente: {fuentes[i]}]\n"
        return contexto if hay_relevancia else ""
    except: return ""

# --- 3. DISEÑO VISUAL (Hacks CSS) ---

def footer_personalizado():
    estilos = """
    <style>
        /* Footer fijo */
        .footer-credits {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #ffffff;
            color: #444;
            text-align: center;
            font-size: 13px;
            padding: 12px;
            border-top: 2px solid #C59200;
            z-index: 9999;
            font-family: sans-serif;
            box-shadow: 0px -2px 5px rgba(0,0,0,0.1);
        }
        .footer-names {
            font-weight: bold;
            color: #002F6C;
            margin-bottom: 4px;
        }
        .footer-tech {
            font-size: 11px;
            color: #666;
        }
        div[data-testid="stBottom"] {
            padding-bottom: 70px;
        }
        footer {visibility: hidden;}

        /* CSS Avatar Chat (Burbujas) */
        [data-testid="stChatMessageAvatar"] {
            width: 85px !important;
            height: 85px !important;
            border-radius: 50% !important;
            margin-right: 15px !important;
        }
        [data-testid="stChatMessageAvatar"] img {
            width: 85px !important;
            height: 85px !important;
            object-fit: contain !important;
        }
        [data-testid="stChatMessageAvatar"] svg {
            width: 50px !important;
            height: 50px !important;
        }

        /* Traducción Uploader */
        [data-testid="stFileUploader"] section > div > div > span,
        [data-testid="stFileUploader"] section > div > div > small {
            display: none !important;
        }
        [data-testid="stFileUploader"] section > div > div::after {
            content: "📂 Arrastra y suelta tus archivos PDF aquí";
            display: block;
            font-weight: bold;
            color: #444;
            margin-bottom: 5px;
            font-size: 16px;
        }
        [data-testid="stFileUploader"] section > div > div::before {
            content: "Límite: 200MB por archivo • Formato PDF";
            display: block;
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
    </style>

    <div class="footer-credits">
        <div class="footer-names">
            Hecho por: Altamirano Isis, Castillo Alexander, Chalán David, Flores Bryan, Cabezas Jhampierre
        </div>
        <div class="footer-tech">
            Proyecto Académico | Powered by Google Gemini API | Algoritmos: TF-IDF, Cosine Similarity & RAG Architecture.
        </div>
    </div>
    """
    st.markdown(estilos, unsafe_allow_html=True)

# --- NUEVA FUNCIÓN: ENCABEZADO INSTITUCIONAL ---
# Esto pone el logo y el texto arriba, en el cuerpo principal
def encabezado_institucional():
    # Creamos columnas: Logo (pequeño) | Texto (Grande)
    col_logo, col_texto = st.columns([1, 6])
    
    with col_logo:
        try:
            st.image(LOGO_URL, width=130) # Tamaño del logo institucional
        except:
            st.error("Logo no encontrado")
            
    with col_texto:
        st.markdown("## Universidad Central del Ecuador")
        st.markdown("#### FICA - Facultad de Ingeniería y Ciencias Aplicadas")
        st.markdown("**Carrera de Sistemas de Información**")
    
    st.divider() # Línea separadora elegante

# --- 4. INTERFACES GRÁFICAS ---

def sidebar_uce():
    with st.sidebar:
        # Ya quitamos el logo y el texto de aquí, ahora solo queda la navegación
        st.title("Navegación")
        opcion = st.radio("Selecciona una opción:", ["💬 Chat con Ing. Condoi", "📂 Gestión de Bibliografía"])
        
        st.divider()
        st.caption("© 2026 UCE - Ingeniería en Sistemas")
        return opcion

def interfaz_gestor_archivos():
    footer_personalizado()
    encabezado_institucional() # <--- AQUI LLAMAMOS AL NUEVO ENCABEZADO
    
    col_img, col_txt = st.columns([1.5, 3.5]) # Damos más espacio a la izquierda para el avatar grande
    
    with col_img:
        if os.path.exists(AVATAR_URL_GESTION):
            img_b64 = get_img_as_base64(AVATAR_URL_GESTION)
            # Aumentado a 350px
            st.markdown(f'<img src="data:image/gif;base64,{img_b64}" style="width:350px; max-width: 100%;">', unsafe_allow_html=True)
        elif os.path.exists(AVATAR_URL):
            img_b64 = get_img_as_base64(AVATAR_URL)
            st.markdown(f'<img src="data:image/gif;base64,{img_b64}" style="width:350px; max-width: 100%;">', unsafe_allow_html=True)
        else:
            st.markdown("📂")
            
    with col_txt:
        st.header("📂 Gestión de Bibliografía") 
        st.info("Ayuda al Ing. Condoi a aprender subiendo los sílabos y libros aquí.") 
    
    st.markdown("---") 
    
    col1, col2 = st.columns([1, 2]) 
    with col1: 
        uploaded_files = st.file_uploader("Cargar documentos PDF", type="pdf", accept_multiple_files=True) 
        if uploaded_files: 
            if st.button("Procesar Documentos", type="primary"): 
                contador = 0 
                for file in uploaded_files: 
                    guardar_archivo(file) 
                    contador += 1 
                leer_pdfs_locales.clear()
                st.success(f"✅ {contador} documentos aprendidos por el sistema.") 
                st.rerun() 
    with col2: 
        st.subheader("📚 Memoria del Ing. Condoi:") 
        archivos = os.listdir(PDF_FOLDER) 
        if not archivos: 
            st.warning("Memoria vacía. Sube archivos.") 
        else: 
            for f in archivos: 
                c1, c2 = st.columns([4, 1]) 
                c1.text(f"📄 {f}") 
                if c2.button("🗑️", key=f, help="Borrar"): 
                    eliminar_archivo(f) 
                    leer_pdfs_locales.clear()
                    st.toast(f"Olvidando: {f}") 
                    st.rerun() 

def interfaz_chat():
    footer_personalizado()
    encabezado_institucional() # <--- AQUI LLAMAMOS AL NUEVO ENCABEZADO
    
    # Ajustamos las columnas para que el Avatar tenga más espacio
    col_avatar, col_texto = st.columns([1.5, 3.5])
    
    with col_avatar:
        if os.path.exists(AVATAR_URL):
            img_b64 = get_img_as_base64(AVATAR_URL)
            # Aumentado a 350px (Más grande e impactante)
            st.markdown(f'<img src="data:image/gif;base64,{img_b64}" style="width:350px; max-width: 100%;">', unsafe_allow_html=True)
        else:
            st.markdown("🤖")
            
    with col_texto:
        st.header("💬 Ing. Condoi")
        st.caption("Tu Tutor Virtual de la FICA - UCE")

    modelo, status = conseguir_modelo_disponible()
    if not modelo:
        st.error(f"Error de conexión: {status}")
        st.stop()
    
    st.info("""
    **🦅 ¡Hola compañero! Soy el Ing. Condoi.**
  
    * Si quieres conversar sobre algún tema en general, ¡escribe abajo!
    * Si necesitas que revise información específica, ve a **"Gestión de Bibliografía"** y dame los archivos.
    """)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    avatar_bot = AVATAR_URL if os.path.exists(AVATAR_URL) else "assistant"
    avatar_user = "👤"

    for message in st.session_state.messages:
        icono = avatar_bot if message["role"] == "assistant" else avatar_user
        with st.chat_message(message["role"], avatar=icono):
            st.markdown(message["content"])

    if prompt := st.chat_input("Pregúntale al Ing. Condoi..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=avatar_user):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=avatar_bot):
            placeholder = st.empty()
            placeholder.markdown("🦅 *El Ing. Condoi está pensando...*")
            
            try:
                textos, fuentes = leer_pdfs_locales()
                contexto_pdf = buscar_informacion(prompt, textos, fuentes)
                
                prompt_sistema = f"""
                Tienes una identidad definida: Eres el **Ing. Condoi**.
                Eres el tutor virtual oficial (un águila/cóndor ingeniero) de la FICA (Facultad de Ingeniería y Ciencias Aplicadas) de la Universidad Central del Ecuador.
                
                Tu personalidad es:
                1. Profesional pero amigable y accesible para **cualquier estudiante** de la universidad.
                2. Motivador, usas frases como "¡Vamos compañero!", "Excelente pregunta", "Estamos aquí para aprender".
                3. Tratas al usuario como un **compañero universitario** en general, no solo como ingeniero.
                4. Siempre mencionas "según la documentación" si usas los PDFs.
                
                CONTEXTO (RAG):
                {contexto_pdf}
                
                PREGUNTA DEL ESTUDIANTE: {prompt}
                """
                
                model = genai.GenerativeModel(modelo)
                response = model.generate_content(prompt_sistema)
                
                placeholder.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
                
            except Exception as e:
                st.error(f"Error del sistema: {e}")

# --- 4. MAIN ---

def main():
    opcion = sidebar_uce()

    if opcion == "📂 Gestión de Bibliografía":
        interfaz_gestor_archivos()
    elif "Chat" in opcion:
        interfaz_chat()

if __name__ == "__main__":
    main()
