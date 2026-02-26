import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import base64

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()

# Prioridad: Streamlit Secrets (para la nube) o .env (local)
api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Ing. Condoi - UCE",
    page_icon="🦅",
    layout="wide"
)

if not api_key:
    st.error("❌ ERROR: No encontré la API Key. Revisa tu configuración.")
    st.stop()

genai.configure(api_key=api_key)

PDF_FOLDER = 'archivos_pdf'
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

# --- RECURSOS GRÁFICOS ---
LOGO_URL = "UCELOGO.png"
AVATAR_URL = "avatar_uce.gif"
AVATAR_URL_GESTION = "avatar_uce2.gif"

# --- 2. FUNCIONES DE LÓGICA (Backend) ---

def get_img_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except: return ""

def conseguir_modelo_disponible():
    try:
        modelos = list(genai.list_models())
        modelos_chat = [m for m in modelos if 'generateContent' in m.supported_generation_methods]
        if not modelos_chat: return None, "Sin modelos compatibles."
        nombres = [m.name for m in modelos_chat]
        preferidos = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro']
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
    if os.path.exists(ruta): os.remove(ruta)

@st.cache_resource
def leer_pdfs_locales():
    textos, fuentes = [], []
    if not os.path.exists(PDF_FOLDER): return [], []
    archivos = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
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

# --- 3. DISEÑO VISUAL ---

def estilos_globales():
    estilos = """
    <style>
        .block-container { padding-top: 3rem !important; padding-bottom: 0rem !important; }
        
        .footer-credits {
            position: fixed; left: 0; bottom: 0; width: 100%;
            background-color: #ffffff; text-align: center;
            font-size: 11px; padding: 5px; border-top: 2px solid #C59200; z-index: 99999;
        }
        
        div[data-testid="stBottom"] { padding-bottom: 35px; background-color: transparent; }
        
        /* Estilo para el uploader traducido */
        [data-testid="stFileUploader"] section > div > div > span,
        [data-testid="stFileUploader"] section > div > div > small { display: none !important; }
        [data-testid="stFileUploader"] section > div > div::after {
            content: "📂 Arrastra y suelta tus archivos PDF aquí";
            display: block; font-weight: bold; color: #444;
        }
    </style>
    <div class="footer-credits">
        <div style="font-weight: bold; color: #002F6C;">Hecho por: Altamirano Isis, Castillo Alexander, Chalán David, Flores Bryan, Cabezas Jhampierre</div>
        <div style="font-size: 9px; color: #666;">Proyecto Académico | Powered by Google Gemini API</div>
    </div>
    """
    st.markdown(estilos, unsafe_allow_html=True)

# --- 4. INTERFACES ---

def sidebar_uce():
    with st.sidebar:
        st.markdown("### UCE - FICA")
        st.divider()
        st.title("Navegación")
        opcion = st.radio("Ir a:", ["💬 Chat con Ing. Condoi", "📂 Gestión de Bibliografía"])
        return opcion

def interfaz_gestor_archivos():
    estilos_globales()
    col_hl, col_ht = st.columns([0.8, 5])
    with col_hl:
        if os.path.exists(LOGO_URL): st.image(LOGO_URL, width=90)
    with col_ht:
        st.header("Gestión de Bibliografía")
    
    col_avatar, col_contenido = st.columns([1, 3])
    with col_avatar:
        avatar_path = AVATAR_URL_GESTION if os.path.exists(AVATAR_URL_GESTION) else AVATAR_URL
        if os.path.exists(avatar_path):
            img_b64 = get_img_as_base64(avatar_path)
            st.markdown(f'<img src="data:image/gif;base64,{img_b64}" style="width:100%; max-width: 300px; border-radius: 15px;">', unsafe_allow_html=True)

    with col_contenido:
        st.info("Ayuda al Ing. Condoi a aprender subiendo los sílabos y libros aquí.")
        uploaded_files = st.file_uploader("Cargar PDFs", type="pdf", accept_multiple_files=True)
        if uploaded_files and st.button("Procesar Documentos", type="primary"):
            for f in uploaded_files: guardar_archivo(f)
            leer_pdfs_locales.clear()
            st.success("✅ Documentos aprendidos.")
            st.rerun()
        
        st.subheader("📚 Memoria:")
        for f in os.listdir(PDF_FOLDER):
            c1, c2 = st.columns([4, 1])
            c1.text(f"📄 {f}")
            if c2.button("🗑️", key=f):
                eliminar_archivo(f)
                leer_pdfs_locales.clear()
                st.rerun()

def interfaz_chat():
    estilos_globales()
    
    # === ENCABEZADO ALINEADO (Logo | Texto | Avatar) ===
    col_logo, col_titulo, col_avatar_head = st.columns([1.2, 3, 1.2])

    with col_logo:
        if os.path.exists(LOGO_URL):
            st.markdown('<div style="margin-top: 15px;">', unsafe_allow_html=True)
            st.image(LOGO_URL, width=150)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_titulo:
        st.markdown("""
            <div style="padding-top: 30px;">
                <h1 style='margin-bottom: 0px; color: #002F6C; font-size: 2.5rem;'>Asistente Virtual</h1>
                <p style='margin-top: 0px; color: gray; font-size: 16px;'>Ing. Condoi - Tu Tutor Virtual de la FICA</p>
            </div>
        """, unsafe_allow_html=True)

    with col_avatar_head:
        if os.path.exists(AVATAR_URL):
            st.markdown('<div style="margin-top: 10px;">', unsafe_allow_html=True)
            st.image(AVATAR_URL, width=160)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Bienvenida con estilo institucional
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #C59200;">
        <strong>🦅 ¡Hola compañero! Soy el Ing. Condoi.</strong><br>
        Estoy aquí para ayudarte con tus dudas académicas. Si necesitas que analice un documento específico, súbelo en la sección de bibliografía.
    </div>
    """, unsafe_allow_html=True)

    contenedor_chat = st.container(height=380, border=True)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with contenedor_chat:
        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar=AVATAR_URL if m["role"]=="assistant" else "👤"):
                st.markdown(m["content"])

    if prompt := st.chat_input("Escribe tu consulta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        modelo, _ = conseguir_modelo_disponible()
        with contenedor_chat:
            with st.chat_message("assistant", avatar=AVATAR_URL):
                placeholder = st.empty()
                placeholder.markdown("🦅 *Consultando archivos...*")
                try:
                    textos, fuentes = leer_pdfs_locales()
                    contexto = buscar_informacion(st.session_state.messages[-1]["content"], textos, fuentes)
                    model = genai.GenerativeModel(modelo)
                    prompt_final = f"Eres el Ing. Condoi de la FICA-UCE. Contexto: {contexto}. Pregunta: {st.session_state.messages[-1]['content']}"
                    response = model.generate_content(prompt_final)
                    placeholder.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error: {e}")

def main():
    opcion = sidebar_uce()
    if opcion == "📂 Gestión de Bibliografía":
        interfaz_gestor_archivos()
    else:
        interfaz_chat()

if __name__ == "__main__":
    main()
