import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- 1. CONFIGURACIÓN INICIAL ---
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configuración de página
st.set_page_config(
    page_title="Asistente Académico UCE", 
    page_icon="🏛️", 
    layout="wide"
)

if not api_key:
    st.error("❌ ERROR: No encontré la API Key. Revisa tu archivo .env")
    st.stop()

genai.configure(api_key=api_key)

PDF_FOLDER = 'archivos_pdf'
if not os.path.exists(PDF_FOLDER):
    os.makedirs(PDF_FOLDER)

# Logo UCE
LOGO_URL = "UCELOGO.png"

# --- 2. FUNCIONES DE LÓGICA (Backend) ---

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

def leer_pdfs_locales():
    textos, fuentes = [], []
    archivos = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    if not archivos: return [], []
    for archivo in archivos:
        try:
            reader = PyPDF2.PdfReader(os.path.join(PDF_FOLDER, archivo))
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

# --- 3. DISEÑO VISUAL (Footer Personalizado) ---

def footer_personalizado():
    estilos = """
    <style>
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

# --- 4. INTERFACES GRÁFICAS ---

def sidebar_uce():
    with st.sidebar:
        try:
            st.image(LOGO_URL, width=150)
        except:
            st.header("UCE")
            
        # --- SECCIÓN MODIFICADA: DATOS DE FACULTAD Y CARRERA ---
        st.markdown("## Universidad Central del Ecuador")
        
        st.markdown("### FICA")
        st.markdown("**Facultad de Ingeniería y Ciencias Aplicadas**")
        st.markdown("Carrera de Sistemas de Información")
        # -------------------------------------------------------
        
        st.divider()
        
        st.title("Navegación")
        opcion = st.radio("Selecciona una opción:", ["💬 Chat Estudiantil", "📂 Gestión de Bibliografía"])
        
        st.divider()
        st.caption("© 2026 UCE - Ingeniería en Sistemas")
        return opcion

def interfaz_gestor_archivos():
    st.header("📂 Gestión de Bibliografía UCE")
    st.info("Sube aquí los sílabos, libros o papers para que los estudiantes puedan consultarlos.")
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
                st.success(f"✅ {contador} documentos añadidos a la base de conocimiento.")
                st.rerun()

    with col2:
        st.subheader("📚 Documentos Disponibles:")
        archivos = os.listdir(PDF_FOLDER)
        if not archivos:
            st.warning("No hay material cargado aún.")
        else:
            for f in archivos:
                c1, c2 = st.columns([4, 1])
                c1.text(f"📄 {f}")
                if c2.button("🗑️", key=f, help="Borrar"):
                    eliminar_archivo(f)
                    st.toast(f"Documento eliminado: {f}")
                    st.rerun()
    
    footer_personalizado()

def interfaz_chat():
    st.header("💬 Asistente Académico UCE")
    st.caption("Plataforma de asistencia estudiantil basada en Inteligencia Artificial.")
    
    modelo, status = conseguir_modelo_disponible()
    if not modelo:
        st.error(f"Error de conexión: {status}")
        st.stop()
    
    archivos = os.listdir(PDF_FOLDER)
    
    if not archivos:
        st.info("""
        **👋 ¡Bienvenido al Chat de Ingeniería!**
        
        Actualmente no hay bibliografía cargada en el sistema. Tienes dos opciones:
        1. **Chatear libremente:** Puedo responder preguntas usando mi conocimiento general.
        2. **Cargar Material:** Ve a la pestaña **"📂 Gestión de Bibliografía"** para subir los PDFs del curso.
        """)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    footer_personalizado()

    if prompt := st.chat_input("¿En qué puedo ayudarte hoy?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🔵 *Consultando base de datos UCE...*")
            
            try:
                textos, fuentes = leer_pdfs_locales()
                contexto_pdf = buscar_informacion(prompt, textos, fuentes)
                
                prompt_sistema = f"""
                Actúa como un tutor académico de la Universidad Central del Ecuador (UCE).
                Tu tono debe ser formal, académico pero cercano y motivador (estilo "Omnium Potentior Est Sapientia").
                
                CONTEXTO BIBLIOGRÁFICO:
                {contexto_pdf}
                
                INSTRUCCIONES:
                1. Si la respuesta está en los documentos, explícala con claridad y cita la fuente.
                2. Si no está, usa tu conocimiento general para guiar al estudiante.
                3. Trata al usuario como "compañero" o "estudiante".
                
                PREGUNTA: {prompt}
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
    elif opcion == "💬 Chat Estudiantil":
        interfaz_chat()

if __name__ == "__main__":
    main()
