# --- IMPORTS ---
import base64
import json
import os
import pickle
import re  # Para expresiones regulares
import time  # Import time for cache expiration
import traceback  # Import traceback for detailed error logging
import kg_embedding  # Import our new knowledge graph module
import model_config
import threading
from datetime import datetime

from flask import Flask, request, jsonify, make_response
from flask_cors import CORS  # Para permitir CORS
from openai import OpenAI
import model_config
from model_config import get_llm_context_config, use_threshold_for_llm, get_entity_search_config # <-- Añadir aquí
from SPARQLWrapper import SPARQLWrapper, DIGEST  # For Virtuoso authentication
import rdflib  # For RDF parsing

# --- GLOBAL PROGRESS TRACKING ---
INITIALIZATION_PROGRESS = {
    'status': 'idle',  # idle, running, completed, error
    'progress': 0,     # 0-100
    'current_step': '',
    'total_steps': 7,
    'start_time': None,
    'error_message': None,
    'logs': []
}
# --- GLOBAL PROGRESS TRACKING ---
INITIALIZATION_PROGRESS = {
    'status': 'idle',  # idle, running, completed, error
    'progress': 0,     # 0-100
    'current_step': '',
    'total_steps': 7,
    'start_time': None,
    'error_message': None,
    'logs': []
}
PROGRESS_LOCK = threading.Lock()

def update_progress(step_number, step_description, log_message=None, custom_progress=None):
    """Update initialization progress with smart step weighting"""
    global INITIALIZATION_PROGRESS
    with PROGRESS_LOCK:
        # Define progress ranges for each step
        # Steps 1-4: 0% → 15% (quick setup steps)
        # Step 5: 15% → 85% (training - will be updated dynamically)
        # Steps 6-7: 85% → 100% (finalization)
        step_ranges = {
            0: 0,    # Start
            1: 3,    # Ontology structure
            2: 6,    # Fetch triples  
            3: 9,    # Filter entities
            4: 12,   # Initialize KG structure
            5: 15,   # Training START (will go 15% → 85%)
            6: 85,   # Entity embeddings
            7: 100   # Complete
        }
        
        if custom_progress is not None:
            # Use custom progress (for training updates)
            progress_percent = custom_progress
        else:
            # Use predefined step ranges
            progress_percent = step_ranges.get(step_number, 0)
        
        INITIALIZATION_PROGRESS['progress'] = progress_percent
        INITIALIZATION_PROGRESS['current_step'] = step_description
        
        if log_message:
            timestamp = datetime.now().strftime("%H:%M:%S")
            INITIALIZATION_PROGRESS['logs'].append(f"[{timestamp}] {log_message}")
            # Keep only last 20 log entries
            if len(INITIALIZATION_PROGRESS['logs']) > 20:
                INITIALIZATION_PROGRESS['logs'] = INITIALIZATION_PROGRESS['logs'][-20:]
            
            # Only print important messages, not every progress update
            print(f"[{timestamp}] {log_message}")
        
        # Only print major progress milestones, not every update
        if step_number == 0 or step_number == 7 or step_number % 2 == 0:
            print(f"Progress: {progress_percent}% - {step_description}")

def update_training_progress(current_epoch, total_epochs):
    """Update progress during training with real epoch progress"""
    if total_epochs <= 0:
        return
    
    # Training occupies 15% → 85% range (70% total)
    training_start = 15
    training_end = 85
    training_range = training_end - training_start
    
    # Calculate progress within training range
    epoch_progress = min(current_epoch / total_epochs, 1.0)
    training_progress = training_start + (epoch_progress * training_range)
    
    update_progress(5, f"Training knowledge graph embeddings... (Epoch {current_epoch}/{total_epochs})", 
                   custom_progress=int(training_progress))

def reset_progress():
    """Reset progress tracking"""
    global INITIALIZATION_PROGRESS
    with PROGRESS_LOCK:
        INITIALIZATION_PROGRESS.update({
            'status': 'idle',
            'progress': 0,
            'current_step': 'Preparando inicialización...',
            'start_time': None,
            'error_message': None,
            'logs': []
        })

def set_progress_error(error_message):
    """Set error state for progress tracking"""
    global INITIALIZATION_PROGRESS
    with PROGRESS_LOCK:
        INITIALIZATION_PROGRESS['status'] = 'error'
        INITIALIZATION_PROGRESS['error_message'] = error_message
        timestamp = datetime.now().strftime("%H:%M:%S")
        INITIALIZATION_PROGRESS['logs'].append(f"[{timestamp}] ERROR: {error_message}")

def _initialize_kg_from_scratch():
    """
    Contiene toda la lógica para construir el KG y el caché desde Virtuoso.
    Esta función solo se llama si el caché no existe o ha expirado.
    """
    # Hacer globales las variables que vamos a modificar
    global ALL_TRIPLES, embedding_model, entity_embeddings, KG_EMBEDDING_READY, INITIALIZATION_PROGRESS

    # Ensure progress is completely reset at the start
    reset_progress()
    
    # Initialize progress tracking
    with PROGRESS_LOCK:
        INITIALIZATION_PROGRESS['status'] = 'running'
        INITIALIZATION_PROGRESS['start_time'] = datetime.now()
        INITIALIZATION_PROGRESS['logs'] = []
    
    update_progress(0, "Initializing Knowledge Graph system...", "Starting KG initialization from Virtuoso")

    try:
        # Step 1: Fetch ontology structure
        update_progress(1, "Fetching ontology structure...", "Retrieving class hierarchies from database")
        kg_embedding.ONTOLOGY_STRUCTURE = kg_embedding.fetch_ontology_structure(
            virtuoso_config=VIRTUOSO_CONFIG
        )

        if not kg_embedding.ONTOLOGY_STRUCTURE:
            update_progress(1, "Fetching ontology structure...", "WARNING: No ontology structure found")
        else:
            update_progress(1, "Fetching ontology structure...", f"Fetched {len(kg_embedding.ONTOLOGY_STRUCTURE)} class hierarchies")

        # Step 2: Fetch triples from Virtuoso
        update_progress(2, "Fetching triples from Virtuoso...", "Retrieving RDF triples from database")
        raw_triples = kg_embedding.fetch_triples_from_virtuoso(
            virtuoso_config=VIRTUOSO_CONFIG
        )

        if not raw_triples:
            error_msg = "No triples fetched from Virtuoso. Knowledge graph system cannot initialize."
            set_progress_error(error_msg)
            KG_EMBEDDING_READY = False
            return

        update_progress(2, "Fetching triples from Virtuoso...", f"Retrieved {len(raw_triples)} raw triples")

        # Step 3: Filter problematic entities
        update_progress(3, "Filtering problematic entities...", "Removing blank nodes and invalid entities")
        triples = []
        blank_node_count = 0

        for triple in raw_triples:
            subject = str(triple[0])
            predicate = str(triple[1])
            obj = str(triple[2])

            if (
                kg_embedding.is_blank_node(subject)
                or kg_embedding.is_blank_node(obj)
                or "NamedIndividual" in subject
                or "NamedIndividual" in obj
            ):
                blank_node_count += 1
                continue

            triples.append(triple)

        update_progress(3, "Filtering problematic entities...", 
                       f"Filtered {len(triples)} valid triples, excluded {blank_node_count} problematic entities")

        if not triples:
            error_msg = "No triples available after filtering. Knowledge graph system cannot initialize."
            set_progress_error(error_msg)
            KG_EMBEDDING_READY = False
            return

        # Step 4: Initialize KG structure
        update_progress(4, "Initializing knowledge graph structure...", f"Processing {len(triples)} filtered triples")
        kg_embedding.initialize_kg_structure(triples)
        
        # Step 5: Train KG embeddings
        update_progress(5, "Training knowledge graph embeddings...", "Loading KG model configuration")
        active_kg_model = model_config.get_active_kg_model()
        update_progress(5, "Training knowledge graph embeddings...", 
                       f"Using KG model: {active_kg_model['name']} with {active_kg_model['params']} parameters")

        # Create a background thread to update training progress
        import threading
        import time
        
        # Get training epochs from active model configuration
        training_config = active_kg_model.get('training', {})
        epochs = training_config.get('num_epochs', 150)  # Default to 150 if not found
        
        print(f"🎯 [DEBUG] Training config: {training_config}")
        print(f"🎯 [DEBUG] Extracted epochs: {epochs}")
        
        # Flag to stop progress simulation when training completes
        # Variables para tracking de épocas reales
        training_complete = threading.Event()
        real_epochs_available = threading.Event()
        current_real_epoch = 0
        
        def epoch_progress_callback(current_epoch, total_epochs):
            """Callback para recibir épocas reales del entrenamiento"""
            nonlocal current_real_epoch
            current_real_epoch = current_epoch
            real_epochs_available.set()
            update_training_progress(current_epoch, total_epochs)
            print(f"🎯 [REAL EPOCH] Epoch {current_epoch}/{total_epochs} - Progress updated")
        
        def simulate_training_progress():
            """Simulate training progress based on typical training time - FALLBACK ONLY"""
            total_duration = min(epochs * 0.2, 45)  # Estimate: 0.2 seconds per epoch, max 45 seconds
            update_interval = 1.0  # Update every 1 second (slower for real epoch priority)
            updates_needed = int(total_duration / update_interval)
            
            print(f"🎯 [DEBUG] Training simulation started as fallback: {epochs} epochs, ~{total_duration}s duration")
            
            for i in range(updates_needed):
                if training_complete.is_set():
                    print(f"🎯 [DEBUG] Training completed early at update {i}")
                    break
                
                # Wait to see if real epochs are available
                if real_epochs_available.wait(timeout=update_interval):
                    print(f"🎯 [DEBUG] Real epochs detected, stopping simulation")
                    break
                    
                # Only use simulation if no real epochs available
                progress_ratio = (i + 1) / updates_needed
                current_progress = 15 + (progress_ratio * 70)  # 70% is the training range
                estimated_epoch = int(progress_ratio * epochs)
                
                update_progress(5, f"Training knowledge graph embeddings... (Epoch ~{estimated_epoch}/{epochs})", 
                               custom_progress=int(current_progress))
                print(f"🎯 [FALLBACK] Using simulated epoch {estimated_epoch}/{epochs}")
        
        # Start progress simulation in background (as fallback)
        progress_thread = threading.Thread(target=simulate_training_progress)
        progress_thread.daemon = True
        progress_thread.start()
        
        # Start actual training with real epoch callback
        kge_model, entity_factory = kg_embedding.train_kge(
            triples, kge_config=active_kg_model, progress_callback=epoch_progress_callback
        )
        
        # Stop progress simulation
        training_complete.set()
        progress_thread.join(timeout=1)  # Wait max 1 second for thread to finish

        if not kge_model:
            error_msg = "Failed to train KG embeddings"
            set_progress_error(error_msg)
            KG_EMBEDDING_READY = False
            return

        update_progress(5, "Training knowledge graph embeddings...", "KG embeddings trained successfully", custom_progress=85)

        # Step 6: Create entity embeddings
        update_progress(6, "Creating entity embeddings...", "Loading embedding model configuration")
        active_embedding_model = model_config.get_active_embedding_model()
        update_progress(6, "Creating entity embeddings...", f"Using embedding model: {active_embedding_model}")

        embedding_model, entity_embeddings = (
            kg_embedding.create_entity_embeddings(
                triples,
                model_name=active_embedding_model,
                use_enriched=True,
                virtuoso_config=VIRTUOSO_CONFIG,
            )
        )

        if not (embedding_model and entity_embeddings):
            error_msg = "Failed to create entity embeddings"
            set_progress_error(error_msg)
            KG_EMBEDDING_READY = False
            return
        
        update_progress(6, "Creating entity embeddings...", f"Created embeddings for {len(entity_embeddings)} entities")
        
        # Step 7: Finalize initialization
        update_progress(7, "Finalizing initialization...", "Saving data and updating global state")
        ALL_TRIPLES = triples
        KG_EMBEDDING_READY = True
        
        # Save to cache
        save_to_cache()
        
        # Mark as completed
        with PROGRESS_LOCK:
            INITIALIZATION_PROGRESS['status'] = 'completed'
            INITIALIZATION_PROGRESS['progress'] = 100
            INITIALIZATION_PROGRESS['current_step'] = 'Knowledge Graph initialization completed successfully'
        
        update_progress(7, "Knowledge Graph initialization completed", 
                       f"Successfully initialized KG with {len(entity_embeddings)} entities")

    except Exception as e:
        error_msg = f"Fatal error during KG initialization: {e}"
        set_progress_error(error_msg)
        traceback.print_exc()
        KG_EMBEDDING_READY = False


# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)

# Configure CORS with explicit settings for all endpoints
CORS(app, 
     resources={
         r"/*": {
             "origins": "*",
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": ["Content-Type", "Authorization", "Access-Control-Allow-Credentials"],
             "supports_credentials": False
         }
     }
)

# --- LLM CONFIGURATION ---
MODEL_URL = "http://192.168.212.254:21000/v1/chat/completions"
MODEL_NAME = "deepHermes-3-Mistral-24B"

# Initialize OpenAI client for local model
client = OpenAI(
    base_url="http://192.168.212.254:21000/v1",
    api_key="not-needed",  # Local models typically don't need API keys
)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = (
    "Eres un asistente especializado en grafos de conocimiento y ontologías. Tu misión es responder preguntas "
    "usando EXCLUSIVAMENTE la información del contexto proporcionado, con respuestas claras, precisas y bien estructuradas.\n\n"
    "🎯 **PRINCIPIOS FUNDAMENTALES:**\n"
    "1. **RESPONDE SIEMPRE EN ESPAÑOL** con lenguaje natural y accesible\n"
    "2. **FIDELIDAD AL CONTEXTO**: Usa solo información presente en el grafo\n"
    "3. **CERO INVENCIÓN**: Si no está en el contexto, indícalo claramente\n"
    "4. **CITACIÓN IMPLÍCITA**: Menciona las entidades específicas que usas\n\n"
    "� **TIPOS DE CONSULTA Y RESPUESTAS:**\n"
    "• **Consultas exploratorias** (\"qué es X\"): Definición + relaciones + contexto\n"
    "• **Consultas de navegación** (\"subclases de X\"): Jerarquías estructuradas\n"
    "• **Consultas relacionales** (\"cómo se relaciona X con Y\"): Conexiones específicas\n"
    "• **Consultas de instancia** (\"ejemplos de X\"): Instancias concretas + propiedades\n"
    "• **Consultas comparativas** (\"diferencias entre X e Y\"): Análisis contrastivo\n\n"
    
    "🏗️ **ESTRUCTURA DE RESPUESTA ADAPTATIVA:**\n"
    "**Para respuestas simples:**\n"
    "- Respuesta directa en 1-2 párrafos\n"
    "- Menciona la(s) entidad(es) clave usadas\n"
    "\n"
    "**Para respuestas complejas:**\n"
    "## Título Principal\n"
    "Introducción breve y contexto\n"
    "\n"
    "### Sección Específica\n"
    "Contenido detallado con:\n"
    "- **Entidades clave** en negritas\n"
    "- *Propiedades* en cursiva\n"
    "- `URIs o identificadores` en código\n"
    "\n"
    "### Relaciones y Jerarquías\n"
    "Para jerarquías tipo subClassOf:\n"
    "• **ClasePadre**\n"
    "  - Subclase1\n"
    "  - Subclase2\n"
    "    - Sub-subclase (si existe)\n"
    "\n"
    "### Ejemplos del Contexto\n"
    "(Solo si hay instancias específicas)\n\n"
    "🔍 **GESTIÓN DE CONTEXTO:**\n"
    "- **Prioriza entidades** mencionadas directamente en la pregunta\n"
    "- **Explora relaciones** relevantes para completar la respuesta\n"
    "- **Usa descripciones y etiquetas** para enriquecer el contexto semántico\n"
    "- **Mantén coherencia** entre diferentes partes de la respuesta\n\n"
    "⚠️ **MANEJO DE LIMITACIONES:**\n"
    "Si la información es insuficiente:\n"
    "1. Responde lo que SÍ puedes con el contexto disponible\n"
    "2. Indica claramente qué información falta\n"
    "3. Sugiere reformulaciones específicas: \"Podrías preguntar sobre **término_específico**\"\n"
    "4. Mantén el formato markdown para estructura clara\n\n"
    "✨ **CALIDAD DE RESPUESTA:**\n"
    "- **Precisión**: Información verificable en el contexto\n"
    "- **Claridad**: Lenguaje accesible sin jerga innecesaria\n"
    "- **Completitud**: Cubre todos los aspectos relevantes disponibles\n"
    "- **Utilidad**: Respuesta práctica que satisface la consulta del usuario\n"
    "- **Estructura**: Formato markdown consistente y legible"
)

# --- CONVERSATION HISTORY ---
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# --- VIRTUOSO CONFIGURATION ---
# Dynamic configuration - will be updated when user selects a graph
VIRTUOSO_CONFIG = {
    "endpoint": "http://192.168.216.102:8890/sparql",
    "database": None,  # Will be set when user selects a graph
    "username": "dba",
    "password": "password",  # Virtuoso password
}

# Current selected graph URI - updated via /select-graph endpoint
CURRENT_GRAPH_URI = None

def check_kg_ready():
    """
    Helper function to check if KG system is ready.
    Returns (is_ready: bool, error_response: dict)
    """
    if not KG_EMBEDDING_READY:
        return False, {
            "error": "Knowledge Graph system not initialized. Please select a graph first via /select-graph endpoint.",
            "status": "error",
            "kg_status": "not_ready"
        }
    
    if not CURRENT_GRAPH_URI:
        return False, {
            "error": "No graph selected. Please select a graph first via /select-graph endpoint.",
            "status": "error", 
            "kg_status": "no_graph_selected"
        }
    
    return True, None

# --- Knowledge Graph Embeddings Setup ---
# Global variables with proper initialization
KG_EMBEDDING_READY = False
ALL_TRIPLES = []
embedding_model = None
entity_embeddings = None

# Persistent cache file paths for storage between server restarts
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
ONTOLOGY_CACHE = os.path.join(CACHE_DIR, "ontology_structure.pkl")
TRIPLES_CACHE = os.path.join(CACHE_DIR, "all_triples.pkl")
MODEL_CACHE = os.path.join(CACHE_DIR, "embedding_model.pkl")
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "entity_embeddings.pkl")
STATUS_CACHE = os.path.join(CACHE_DIR, "kg_ready.txt")
CACHE_TIMESTAMP = os.path.join(CACHE_DIR, "timestamp.txt")
CACHE_EXPIRY = 43200  # Tiempo de expiración en segundos (12 horas)

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)


# Function to load cached data if available
def load_cached_data():
    global KG_EMBEDDING_READY, ALL_TRIPLES, embedding_model, entity_embeddings

    try:
        # NUEVA VALIDACIÓN: Verificar que el cache corresponde al grafo actual
        graph_info_cache = os.path.join(CACHE_DIR, "graph_info.txt")
        if os.path.exists(graph_info_cache):
            with open(graph_info_cache, "r") as f:
                cache_info = f.read()
            
            # Extraer la URI del grafo del cache
            cached_graph = None
            for line in cache_info.split('\n'):
                if line.startswith('graph_uri:'):
                    cached_graph = line.split(':', 1)[1].strip()
                    break
            
            # Validar que el grafo cached coincide con el actual
            current_graph = CURRENT_GRAPH_URI or VIRTUOSO_CONFIG.get('database')
            if cached_graph != current_graph:
                print(f"[Backend] Cache invalidated: cached graph '{cached_graph}' != current graph '{current_graph}'")
                return False
            else:
                print(f"[Backend] Cache validated: using data for graph '{current_graph}'")
        else:
            print("[Backend] No graph info in cache, proceeding with validation...")
        
        # Verificar si el caché ha expirado
        if os.path.exists(CACHE_TIMESTAMP):
            with open(CACHE_TIMESTAMP, "r") as f:
                timestamp = float(f.read().strip())
                current_time = time.time()
                if current_time - timestamp > CACHE_EXPIRY:
                    print(f"Cache ha expirado (creado hace {(current_time - timestamp)/3600:.1f} horas)")
                    return False

        # Check if the status file exists
        if os.path.exists(STATUS_CACHE):
            with open(STATUS_CACHE, "r") as f:
                status = f.read().strip()
                if status == "ready":
                    print("Found cached KG data, loading...")

                    # Load ontology structure
                    if os.path.exists(ONTOLOGY_CACHE):
                        with open(ONTOLOGY_CACHE, "rb") as f:
                            kg_embedding.ONTOLOGY_STRUCTURE = pickle.load(f)
                        print(f"Loaded {len(kg_embedding.ONTOLOGY_STRUCTURE)} relationships from cache")
                    
                    # NUEVAS LÍNEAS: Cargar las estructuras de clases si existen
                    class_hierarchy_cache = os.path.join(
                        CACHE_DIR, "class_hierarchy.pkl"
                    )
                    class_aliases_cache = os.path.join(CACHE_DIR, "class_aliases.pkl")

                    if os.path.exists(class_hierarchy_cache):
                        with open(class_hierarchy_cache, "rb") as f:
                            kg_embedding.CLASS_HIERARCHY = pickle.load(f)
                        print(f"Loaded class hierarchy with {len(kg_embedding.CLASS_HIERARCHY)} classes")
                        
                    if os.path.exists(class_aliases_cache):
                        with open(class_aliases_cache, "rb") as f:
                            kg_embedding.CLASS_ALIASES = pickle.load(f)
                        print(f"Loaded class aliases with {len(kg_embedding.CLASS_ALIASES)} entries")
                    
                    # Load triples
                    if os.path.exists(TRIPLES_CACHE):
                        with open(TRIPLES_CACHE, "rb") as f:
                            ALL_TRIPLES = pickle.load(f)
                        print(f"Loaded {len(ALL_TRIPLES)} triples from cache")

                    # Load embedding model
                    if os.path.exists(MODEL_CACHE):
                        with open(MODEL_CACHE, "rb") as f:
                            embedding_model = pickle.load(f)
                        print("Loaded embedding model from cache")

                    # Load entity embeddings
                    if os.path.exists(EMBEDDINGS_CACHE):
                        with open(EMBEDDINGS_CACHE, "rb") as f:
                            entity_embeddings = pickle.load(f)
                        print(f"Loaded {len(entity_embeddings)} entity embeddings from cache")
                    
                    # NUEVAS LÍNEAS: Cargar sistema de enrichment
                    enrichment_cache = os.path.join(
                        CACHE_DIR, "annotation_enricher.pkl"
                    )
                    enriched_embeddings_cache = os.path.join(
                        CACHE_DIR, "enriched_embeddings.pkl"
                    )

                    if os.path.exists(enrichment_cache):
                        try:
                            with open(enrichment_cache, "rb") as f:
                                kg_embedding.ANNOTATION_ENRICHER = pickle.load(f)
                            print("✅ Cargado ANNOTATION_ENRICHER desde cache")
                        except Exception as e:
                            print(f"⚠️ Error cargando ANNOTATION_ENRICHER: {e}")
                            kg_embedding.ANNOTATION_ENRICHER = None

                    if os.path.exists(enriched_embeddings_cache):
                        try:
                            with open(enriched_embeddings_cache, "rb") as f:
                                kg_embedding.ENRICHED_EMBEDDINGS = pickle.load(f)
                            print(f"✅ Cargados {len(kg_embedding.ENRICHED_EMBEDDINGS)} ENRICHED_EMBEDDINGS desde cache")
                        except Exception as e:
                            print(f"⚠️ Error cargando ENRICHED_EMBEDDINGS: {e}")
                            kg_embedding.ENRICHED_EMBEDDINGS = None

                    # Set ready flag if all components are loaded
                    if (
                        kg_embedding.ONTOLOGY_STRUCTURE
                        and ALL_TRIPLES
                        and embedding_model
                        and entity_embeddings
                    ):
                        KG_EMBEDDING_READY = True
                        print("Successfully restored KG state from cache")

                        # Verificar estado del sistema de enrichment
                        if (
                            kg_embedding.ANNOTATION_ENRICHER
                            and kg_embedding.ENRICHED_EMBEDDINGS
                        ):
                            print("✅ Sistema de enrichment también restaurado desde cache")
                        else:
                            print("⚠️ Sistema de enrichment no está completo en cache")

                        return True

    except Exception as e:
        print(f"Error loading cached data: {e}")
        traceback.print_exc()

    print("Cached data not available or incomplete")
    return False


# Function to save data to cache
def save_to_cache():
    try:
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Save ontology structure
        with open(ONTOLOGY_CACHE, "wb") as f:
            pickle.dump(kg_embedding.ONTOLOGY_STRUCTURE, f)

        # Save triples
        with open(TRIPLES_CACHE, "wb") as f:
            pickle.dump(ALL_TRIPLES, f)

        # Save embedding model
        with open(MODEL_CACHE, "wb") as f:
            pickle.dump(embedding_model, f)

        # Save entity embeddings
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump(entity_embeddings, f)

        # NUEVAS LÍNEAS: Guardar las estructuras de clases
        if hasattr(kg_embedding, 'CLASS_HIERARCHY'):
            with open(os.path.join(CACHE_DIR, "class_hierarchy.pkl"), "wb") as f:
                pickle.dump(kg_embedding.CLASS_HIERARCHY, f)

        if hasattr(kg_embedding, 'CLASS_ALIASES'):
            with open(os.path.join(CACHE_DIR, "class_aliases.pkl"), "wb") as f:
                pickle.dump(kg_embedding.CLASS_ALIASES, f)

        # NUEVAS LÍNEAS: Guardar sistema de enrichment
        enrichment_cache = os.path.join(CACHE_DIR, "annotation_enricher.pkl")
        enriched_embeddings_cache = os.path.join(CACHE_DIR, "enriched_embeddings.pkl")

        if hasattr(kg_embedding, 'ANNOTATION_ENRICHER') and kg_embedding.ANNOTATION_ENRICHER is not None:
            with open(enrichment_cache, "wb") as f:
                pickle.dump(kg_embedding.ANNOTATION_ENRICHER, f)
            print("✅ Guardado ANNOTATION_ENRICHER en cache")

        if hasattr(kg_embedding, 'ENRICHED_EMBEDDINGS') and kg_embedding.ENRICHED_EMBEDDINGS is not None:
            with open(enriched_embeddings_cache, "wb") as f:
                pickle.dump(kg_embedding.ENRICHED_EMBEDDINGS, f)
            print(f"✅ Guardados {len(kg_embedding.ENRICHED_EMBEDDINGS)} ENRICHED_EMBEDDINGS en cache")
        
        # Save ready status
        with open(STATUS_CACHE, "w") as f:
            f.write("ready")

        # Guardar timestamp actual
        with open(CACHE_TIMESTAMP, "w") as f:
            f.write(str(time.time()))
        
        # NUEVA LÍNEA: Guardar información del grafo usado
        graph_info_cache = os.path.join(CACHE_DIR, "graph_info.txt")
        with open(graph_info_cache, "w") as f:
            f.write(f"graph_uri: {CURRENT_GRAPH_URI or 'None'}\n")
            f.write(f"database: {VIRTUOSO_CONFIG.get('database', 'None')}\n")
            f.write(f"timestamp: {time.time()}\n")

        print("Successfully saved KG data to cache")
        update_progress(7, "Finalizing initialization...", "Cache saved successfully")
    except Exception as e:
        error_msg = f"Error saving to cache: {e}"
        print(error_msg)
        update_progress(7, "Finalizing initialization...", f"Warning: {error_msg}")
        traceback.print_exc()


# COMMENTED OUT: Automatic KG initialization moved to /select-graph endpoint
# This allows users to choose which ontology/graph to work with first
# 
# # Try to load from cache first
# if not load_cached_data():
#     # If not in cache, initialize from scratch
#     _initialize_kg_from_scratch()

print("🚀 Server started - waiting for graph selection...")
print("💡 KG system will initialize after user selects a graph via /select-graph endpoint")

# VERIFICACIÓN ADICIONAL: Asegurar que el sistema de enrichment esté funcional
# Esto maneja casos donde se carga desde cache pero el sistema de enrichment no está completo
if KG_EMBEDDING_READY and ALL_TRIPLES:
    print("\n🔍 Verificación post-inicialización del sistema de enrichment...")

    # Solo verificar, no re-ejecutar si ya está funcionando
    if (
        kg_embedding.ANNOTATION_ENRICHER is not None
        and kg_embedding.ENRICHED_EMBEDDINGS is not None
    ):
        print("✅ Sistema de enrichment ya está funcional")
        print(f"   - ENRICHED_EMBEDDINGS: {len(kg_embedding.ENRICHED_EMBEDDINGS)} entidades")
        print(f"   - ANNOTATION_ENRICHER: ✅")

        # Verificar si el ANNOTATION_ENRICHER tiene sistema dinámico
        if (
            hasattr(kg_embedding.ANNOTATION_ENRICHER, "dynamic_detector")
            and kg_embedding.ANNOTATION_ENRICHER.dynamic_detector is not None
        ):
            print("   - Sistema dinámico: ✅")
        else:
            print("   - Sistema dinámico: ❌ (usando legacy)")

    elif (
        kg_embedding.ANNOTATION_ENRICHER is None
        or kg_embedding.ENRICHED_EMBEDDINGS is None
    ):
        print("⚠️ Sistema de enrichment no está completo")
        print("🔄 Intentando reutilizar componentes ya inicializados...")

        # Solo crear embeddings enriquecidos si no existen, pero reutilizar el enricher
        if (
            kg_embedding.ANNOTATION_ENRICHER is not None
            and kg_embedding.ENRICHED_EMBEDDINGS is None
        ):
            print("🎯 Reutilizando ANNOTATION_ENRICHER existente para crear embeddings...")
            try:
                enriched_embeddings = kg_embedding.create_enriched_entity_embeddings(
                    ALL_TRIPLES, model_name=model_config.get_active_embedding_model()
                )

                if enriched_embeddings:
                    print(f"✅ Embeddings enriquecidos creados: {len(enriched_embeddings)} entidades")
                    # Guardar el cache actualizado
                    save_to_cache()
                else:
                    print("⚠️ No se pudieron crear embeddings enriquecidos")
            except Exception as e:
                print(f"⚠️ Error creando embeddings enriquecidos: {e}")

        elif kg_embedding.ANNOTATION_ENRICHER is None:
            print("⚠️ ANNOTATION_ENRICHER no disponible, inicialización completa necesaria")
            try:
                from annotation_enrichment import create_enhanced_embedding_system

                enriched_embeddings, enricher = create_enhanced_embedding_system(
                    ALL_TRIPLES,
                    existing_embeddings=entity_embeddings,
                    embedding_model_name=model_config.get_active_embedding_model(),
                    virtuoso_config=VIRTUOSO_CONFIG,
                )

                if enriched_embeddings and enricher:
                    kg_embedding.ENRICHED_EMBEDDINGS = enriched_embeddings
                    kg_embedding.ANNOTATION_ENRICHER = enricher
                    print(f"✅ Sistema de enrichment inicializado post-cache:")
                    print(f"   - ENRICHED_EMBEDDINGS: {len(kg_embedding.ENRICHED_EMBEDDINGS)} entidades")
                    print(f"   - ANNOTATION_ENRICHER: ✅")

                    # Guardar el cache actualizado
                    save_to_cache()
                else:
                    print("❌ No se pudo inicializar el sistema de enrichment")
            except Exception as e:
                print(f"⚠️ Error en inicialización completa: {e}")
        else:
            print("✅ Ambos componentes ya están disponibles")
    else:
        print("✅ Sistema de enrichment ya está funcional")
        print(f"   - ENRICHED_EMBEDDINGS: {len(kg_embedding.ENRICHED_EMBEDDINGS) if kg_embedding.ENRICHED_EMBEDDINGS else 0} entidades")
        print(f"   - ANNOTATION_ENRICHER: {'✅' if kg_embedding.ANNOTATION_ENRICHER else '❌'}")

# --- LLM INTERACTION FUNCTION ---

# Global function that can be imported by kg_embedding.py
def call_llm_api(
    prompt, model_name=MODEL_NAME, temperature=0.4, max_tokens=1200, max_retries=2
):
    """Función global que puede ser importada por otros módulos para llamadas LLM locales"""
    for attempt in range(max_retries + 1):
        try:
            print(f"\n[Global LLM] Sending request to local LLM API (attempt {attempt + 1}/{max_retries + 1})")
            print(f"[Global LLM] Model: {model_name}, temp: {temperature}, max_tokens: {max_tokens}")
            print(f"[Global LLM] Prompt length: {len(prompt)} characters")
            print(f"[Global LLM] Prompt preview: {repr(prompt[:200])}{'...' if len(prompt) > 200 else ''}")
            # print(f"[Global LLM] PROMPT PREVIEW: {prompt}")

            completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                presence_penalty=0.1,
                frequency_penalty=0.1,
                messages=[{"role": "user", "content": prompt}],
            )

            if completion.choices and len(completion.choices) > 0:
                content = completion.choices[0].message.content
                if content:
                    print(f"[Global LLM] SUCCESS: Received response with {len(content)} characters")
                    print(f"[Global LLM] Response preview: {repr(content[:200])}{'...' if len(content) > 200 else ''}")
                    # print(f"[Global LLM] RESPONSE PREVIEW: {content}")
                    return content.strip()
                else:
                    error_msg = f"[Global LLM] ERROR: Local LLM response missing content! (attempt {attempt + 1})"
                    print(error_msg)
                    print(f"[Global LLM] Full completion object: {repr(completion)}")
                    print(f"[Global LLM] First choice: {repr(completion.choices[0])}")
                    print(f"[Global LLM] Message: {repr(completion.choices[0].message)}")
                    
                    if attempt < max_retries:
                        print(f"[Global LLM] Retrying in 1 second...")
                        time.sleep(1)
                        continue
                    else:
                        return "Error: Local LLM response missing content."
            else:
                error_msg = f"[Global LLM] ERROR: Unexpected response structure! (attempt {attempt + 1})"
                print(error_msg)
                print(f"[Global LLM] Full completion object: {repr(completion)}")
                print(f"[Global LLM] Choices length: {len(completion.choices) if completion.choices else 'None'}")
                
                if attempt < max_retries:
                    print(f"[Global LLM] Retrying in 1 second...")
                    time.sleep(1)
                    continue
                else:
                    return "Error: Unexpected response structure from local LLM API."

        except Exception as e:
            error_msg = (
                f"[Global LLM] Error with local LLM API (attempt {attempt + 1}): {e}"
            )
            print(error_msg)
            print(traceback.format_exc())

            if attempt < max_retries:
                print(f"[Global LLM] Retrying in 1 second...")
                time.sleep(1)
                continue
            else:
                return f"Error processing local LLM API response: {e}"

    return "Error: All retry attempts failed."

# --- HELPER FUNCTIONS FOR @BROWSE TAG ---
def generate_browse_expansion_data(entities_with_scores, threshold):
    """
    Identifica las URIs de las entidades más relevantes para la expansión @Browse.
    """
    print(f"[Backend] Identifying target URIs for @Browse with threshold {threshold}")
    
    # Filtrar entidades que superan el umbral
    target_entities = [
        {"uri": entity, "score": score}
        for entity, score in entities_with_scores
        if score >= threshold
    ]

    # Si ninguna supera el umbral, usar la mejor como fallback
    if not target_entities and entities_with_scores:
        best_entity, best_score = entities_with_scores[0]
        target_entities.append({"uri": best_entity, "score": best_score})
        print(f"[Backend] No entities above threshold, falling back to top entity: {best_entity}")

    print(f"[Backend] Identified {len(target_entities)} target URIs for frontend expansion.")
    return target_entities

def extract_label_from_uri(uri):
    """
    Extract a human-readable label from a URI
    """
    if not uri or not isinstance(uri, str):
        return str(uri)

    # Try to extract from fragment (#) or last path segment (/)
    if "#" in uri:
        return uri.split("#")[-1]
    elif "/" in uri:
        return uri.split("/")[-1]
    else:
        return uri


# --- FLASK ROUTES ---
@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history
    try:
        # Check if KG system is ready
        kg_ready, error_response = check_kg_ready()
        if not kg_ready:
            return jsonify(error_response), 400
        
        data = request.json
        if not data or "message" not in data:
            return (
                jsonify(
                    {"error": "Invalid request: Missing 'message' in JSON payload"}
                ),
                400,
            )

        user_message = data.get("message")
        graph_data_from_request = data.get(
            "graph_data"
        )  # Get graph_data if provided (can be None)
        contains_browse_tag = data.get(
            "containsBrowseTag", False
        )  # Check for @Browse tag

        print(f"[Backend] === New /chat Request ===")
        print(f"[Backend] User message: {user_message}")
        print(f"[Backend] Contains @Browse tag: {contains_browse_tag}")

        # Iniciar contador de tiempo total del procesamiento
        request_start_time = time.time()

        # Prepare the content for the user's turn
        current_user_content = user_message

        # Check if we have graph data and embeddings initialized
        if not graph_data_from_request:
            graph_data_from_request = {"nodes": [], "links": []}

        if not KG_EMBEDDING_READY:
            print("[Backend] ERROR: Knowledge Graph embeddings not initialized")
            return (
                jsonify(
                    {
                        "error": "The knowledge graph system is not properly initialized. No relationships were found in the database. Please contact the system administrator."
                    }
                ),
                500,
            )

        node_count = len(graph_data_from_request.get("nodes", []))
        link_count = len(graph_data_from_request.get("links", []))
        print(
            f"[Backend] Received graph_data context: {node_count} nodes, {link_count} links."
        )

        # Add node information to the log (simplified)
        if node_count > 0:
            node_labels = []
            for node in graph_data_from_request.get("nodes", []):
                if isinstance(node, dict) and node.get("id"):
                    node_id = node.get("id")
                    node_name = str(node_id).split("/")[-1].split("#")[-1]
                    node_labels.append(node_name)
            print(f"[Backend] Graph contains nodes: {', '.join(node_labels)}")

        if node_count == 0:
            graph_triples = []
        else:
            try:
                # Convert graph data to triples for processing
                graph_triples = kg_embedding.graph_data_to_triples(
                    graph_data_from_request
                )
                print(
                    f"[Backend] Converted graph to {len(graph_triples)} triples for processing"
                )

                if not graph_triples:
                    print(
                        f"[Backend] WARNING: No triples extracted from {node_count} nodes and {link_count} links"
                    )
                    print(
                        "[Backend] This could happen if nodes are not in the ontology structure or have no relationships"
                    )
                    print("[Backend] Proceeding with minimal context...")
                    graph_triples = []
            except Exception as e:
                print(f"[Backend] Error converting graph data to triples: {e}")
                print("[Backend] Proceeding with empty graph context...")
                graph_triples = []

        try:
            # Use embeddings from the enriched system when available
            # Priorizar embeddings enriquecidos si están disponibles
            if (
                kg_embedding.ENRICHED_EMBEDDINGS is not None
                and kg_embedding.ANNOTATION_ENRICHER is not None
            ):
                print("✅ [Backend] Using enriched embedding system")
                use_entity_embeddings = kg_embedding.ENRICHED_EMBEDDINGS
                use_embedding_model = (
                    kg_embedding.ANNOTATION_ENRICHER._get_embedding_model()
                )
            else:
                use_entity_embeddings = (
                    entity_embeddings  # Fallback to basic embeddings
                )
                use_embedding_model = embedding_model

            # Combining visible graph triples with full knowledge base
            combined_triples = list(set(graph_triples + ALL_TRIPLES))
            print(f"[Backend] Using {len(graph_triples)} triples from the visible graph")
            print(f"[Backend] Combining with {len(ALL_TRIPLES)} triples from the full ontology")
            print(f"[Backend] Total of {len(combined_triples)} unique triples")

            # LOG: Embedding distribution analysis for threshold understanding
            print(f"[Backend] 📊 EMBEDDINGS ANALYSIS - THRESHOLD DIAGNOSIS:")
            print(f"[Backend] 📊 Total available embeddings: {len(use_entity_embeddings)}")
            print(f"[Backend] � Configured threshold: 0.05 (may be low for {len(combined_triples)} triples)")
            
            # Mostrar muestra de entidades en embeddings para ver diversidad
            sample_entities = list(use_entity_embeddings.keys())[:15]
            print(f"[Backend] � Sample of entities in embeddings:")
            for i, entity in enumerate(sample_entities):
                entity_name = str(entity).split("/")[-1].split("#")[-1]
                print(f"[Backend] �   {i+1}. {entity_name}")

            # LOG: Información sobre qué entidades van a ser evaluadas
            print(f"[Backend] 📊 The threshold of 0.05 will be applied to the list of embeddings, NOT to the triples")
            print(f"[Backend] � The {len(combined_triples)} triples are used for context, but the search is performed over {len(use_entity_embeddings)} embeddings")
            print(f"[Backend] � IMPORTANT: Only entities that have embeddings can be found")
            # NEW: Extract node IDs from the visible graph for prioritizing
            visible_node_ids = []
            for node in graph_data_from_request.get("nodes", []):
                if isinstance(node, dict) and node.get("id"):
                    node_id = node.get("id")
                    # Extract simplified node name from URI if needed
                    node_name = str(node_id).split("/")[-1].split("#")[-1]
                    visible_node_ids.append(node_name)
                    
            print(f"[Backend] Visible nodes that will be prioritized: {visible_node_ids}")
            
            # Check if any entity mentioned directly in the user's query is visible in the graph
            user_message_lower = user_message.lower()
            direct_matches = []

            for node_name in visible_node_ids:
                # Use word boundary matching to avoid partial matches like "Row" in "datos"
                pattern = r"\b" + re.escape(node_name.lower()) + r"\b"
                if re.search(pattern, user_message_lower):
                    print(f"[Backend] Direct mention of visible node '{node_name}' found in query")
                    # Look up the full URI in entity_embeddings keys
                    for entity_uri in use_entity_embeddings.keys():
                        if str(entity_uri).split("/")[-1].split("#")[-1] == node_name:
                            direct_matches.append(
                                (entity_uri, 0.9)
                            )  # High confidence for direct matches
                            break

            # NUEVA LÓGICA: Siempre usar búsqueda enriquecida cuando está disponible
            # incluso si hay coincidencias directas, para obtener resultados más completos
            if (
                kg_embedding.ENRICHED_EMBEDDINGS is not None
                and kg_embedding.ANNOTATION_ENRICHER is not None
            ):
                # LOG: Análisis detallado del proceso de threshold
                print(f"[Backend] 🔍 LOG THRESHOLD ANALYSIS:")
                print(f"[Backend] 🔍 Embeddings available for search: {len(use_entity_embeddings)}")
                print(f"[Backend] 🔍 Applied threshold: 0.05 (applied AFTER similarity calculation)")
                print(f"[Backend] 🔍 The process is: query → similarity with {len(use_entity_embeddings)} embeddings → filter by threshold")
                print(f"[Backend] 🔍 Only entities with score ≥ 0.05 are included in results")
                
                # Verificar si hay entidades específicas en embeddings (ejemplo genérico)
                specific_entities_found = 0
                for entity_uri in list(use_entity_embeddings.keys())[
                    :20
                ]:  # Revisar primeras 20
                    entity_name = str(entity_uri).split("/")[-1].split("#")[-1]
                    if any(
                        keyword in entity_name.lower()
                        for keyword in ["meteo", "dataset", "service", "endpoint"]
                    ):
                        specific_entities_found += 1
                
                print(f"[Backend] 🔍 Specific relevant entities in sample: {specific_entities_found}/20")
                
                # Obtener la configuración de búsqueda centralizada
                search_config = get_entity_search_config()
                
                enriched_results = kg_embedding.find_related_entities(
                    user_message,
                    embeddings_dict=use_entity_embeddings,
                    embedding_model=use_embedding_model,
                    top_k=search_config["top_k"],
                    similarity_threshold=search_config["similarity_threshold"],
                    visible_nodes=visible_node_ids,
                )

                if enriched_results:
                    print(f"[Backend] ✅ Enriched search returned {len(enriched_results)} results")
                    
                    # Log scores sin normalizar (el nuevo sistema debería mantener scores razonables)
                    if enriched_results:
                        scores = [score for _, score in enriched_results]
                        min_score = min(scores)
                        max_score = max(scores)
                        print(f"[Backend] Score range: min={min_score:.3f}, max={max_score:.3f}")
                        
                        # Solo normalizar si los scores siguen siendo demasiado altos (indicativo de error)
                        if max_score > 3.0:
                            print(f"[Backend] ⚠️ Scores still too high, applying corrective normalization")
                            normalized_enriched = []
                            for entity, score in enriched_results:
                                # Aplicar normalización suave para scores muy altos
                                normalized_score = min(1.0, score / max_score)
                                normalized_enriched.append((entity, normalized_score))
                            enriched_results = normalized_enriched

                    # Combinar con coincidencias directas si las hay
                    if direct_matches:
                        # Crear un diccionario para evitar duplicados y mantener mejores scores
                        combined_results = {}

                        # Agregar coincidencias directas con prioridad alta
                        for uri, score in direct_matches:
                            combined_results[uri] = max(
                                score, combined_results.get(uri, 0)
                            )

                        # Agregar resultados enriquecidos (ya normalizados)
                        for uri, score in enriched_results:
                            combined_results[uri] = max(
                                score, combined_results.get(uri, 0)
                            )

                        # Convertir de vuelta a lista y ordenar
                        top_entities_list = sorted(
                            combined_results.items(), key=lambda x: x[1], reverse=True
                        )[:15]
                        print(f"[Backend] 🎯 Total combined: {len(top_entities_list)} unique entities")
                    else:
                        top_entities_list = enriched_results
                        print(f"[Backend] 🎯 Using only enriched results: {len(top_entities_list)} entities")
                else:
                    top_entities_list = direct_matches if direct_matches else []
            else:
                # Fallback a lógica original si no hay sistema enriquecido
                if direct_matches:
                    print(f"[Backend] Found {len(direct_matches)} directly mentioned visible entities")
                    
                    # For @Browse tag, validate that direct matches are semantically relevant
                    if contains_browse_tag:
                        print("[Backend] @Browse tag detected - validating direct matches with semantic search")
                        # Get semantic validation config
                        from model_config import get_browse_config

                        browse_config = get_browse_config()
                        validation_config = browse_config["semantic_validation"]

                                # Obtener la configuración de búsqueda centralizada
                        search_config = get_entity_search_config()
                        
                        semantic_matches = kg_embedding.find_related_entities(
                            user_message,
                            embeddings_dict=use_entity_embeddings,
                            embedding_model=use_embedding_model,
                            top_k=search_config["top_k"],
                            similarity_threshold=search_config["similarity_threshold"],
                            visible_nodes=visible_node_ids,
                        )

                        # Convert semantic_matches to tuple format if needed
                        if semantic_matches and isinstance(semantic_matches[0], dict):
                            semantic_matches = [
                                (result["entity"], result["similarity"])
                                for result in semantic_matches
                            ]

                        # Check if any direct match entity appears in top semantic matches with decent score
                        direct_match_uris = [uri for uri, score in direct_matches]
                        semantic_match_uris = [
                            uri
                            for uri, score in semantic_matches[
                                : validation_config["top_k"]
                            ]
                        ]

                        valid_direct_matches = []
                        min_validation_score = validation_config["min_validation_score"]
                        for direct_uri, direct_score in direct_matches:
                            # Check if this direct match also appears in semantic results
                            semantic_match = next(
                                (
                                    s_score
                                    for s_uri, s_score in semantic_matches
                                    if s_uri == direct_uri
                                ),
                                None,
                            )
                            if (
                                semantic_match
                                and semantic_match >= min_validation_score
                            ):
                                valid_direct_matches.append((direct_uri, direct_score))
                                print(f"[Backend] Validated direct match: {direct_uri} (semantic score: {semantic_match:.3f})")
                            else:
                                node_name = (
                                    str(direct_uri).split("/")[-1].split("#")[-1]
                                )
                                print(f"[Backend] Rejecting direct match '{node_name}' - not semantically relevant (score: {semantic_match})")

                        if valid_direct_matches:
                            top_entities_list = valid_direct_matches
                            print(f"[Backend] Using {len(valid_direct_matches)} validated direct matches for @Browse")
                        else:
                            print("[Backend] No valid direct matches for @Browse - falling back to semantic search")
                            top_entities_list = semantic_matches
                    else:
                        # For non-@Browse queries, use direct matches as before
                        top_entities_list = direct_matches
                        print(f"[Backend] Using {len(direct_matches)} direct matches for regular query")
                else:
                    # Find related entities based on the user's question (lower threshold for better results)
                    # Usar parámetros correctos para find_related_entities
                    print(f"[Backend] Searching for entities related to: '{user_message}'")

                    # Obtener la configuración de búsqueda centralizada
                    search_config = get_entity_search_config()
                    
                    top_entities_list = kg_embedding.find_related_entities(
                        user_message,
                        embeddings_dict=use_entity_embeddings,
                        embedding_model=use_embedding_model,
                        top_k=search_config["top_k"],
                        similarity_threshold=search_config["similarity_threshold"],
                        visible_nodes=visible_node_ids,
                    )
                    
                    print(f"[Backend] Initial search returned {len(top_entities_list)} entities")
                
                # Si no encontramos nada, intentar con términos individuales
                if not top_entities_list and len(user_message.split()) > 1:
                    print("[Backend] No results with full query, trying individual terms...")
                    # Obtener la configuración de búsqueda centralizada
                    search_config = get_entity_search_config()
                    for term in user_message.split():
                        if len(term) > 3:  # Solo términos significativos        
                            term_results = kg_embedding.find_related_entities(
                                user_message,
                                embeddings_dict=use_entity_embeddings,
                                embedding_model=use_embedding_model,
                                top_k=search_config["top_k"],
                                similarity_threshold=search_config["similarity_threshold"],
                                visible_nodes=visible_node_ids,
                            )
                            if term_results:
                                print(f"[Backend] Found {len(term_results)} entities for term '{term}'")
                                top_entities_list.extend(term_results)

                    # Eliminar duplicados y mantener los mejores scores
                    if top_entities_list:
                        seen = set()
                        unique_results = []
                        for entity, score in sorted(
                            top_entities_list, key=lambda x: x[1], reverse=True
                        ):
                            if entity not in seen:
                                seen.add(entity)
                                unique_results.append((entity, score))
                        top_entities_list = unique_results[:10]  # Top 10 únicos

            # Format entities for logging (show only names, not full URIs)
            entities_for_log = []
            for entity, score in top_entities_list:
                entity_name = str(entity).split("/")[-1].split("#")[-1]
                entities_for_log.append((entity_name, round(score, 3)))

            print(f"[Backend] Top related entities: {entities_for_log}")

            if not top_entities_list:
                print("[Backend] ERROR: No related entities found in the knowledge base")
                return (
                    jsonify(
                        {
                            "error": "We cannot find entities related to your question. Please try rephrasing your question."
                        }
                    ),
                    400,
                )

            # Convert results to expected format (entity, score) if they're in dict format
            if top_entities_list and isinstance(top_entities_list[0], dict):
                top_entities_list = [
                    (result["entity"], result["similarity"])
                    for result in top_entities_list
                ]
                print(f"[Backend] Converted to tuple format: {len(top_entities_list)} entities")

            # Prepare entities for LLM context using threshold-based selection
            from model_config import get_llm_context_config, use_threshold_for_llm

            llm_config = get_llm_context_config()

            entities_for_context = []

            if use_threshold_for_llm():
                # Use threshold-based selection
                threshold = llm_config["threshold"]
                min_entities = llm_config["min_entities"]
                max_entities = llm_config["max_entities"]
                backup_count = llm_config["backup_fixed_count"]

                print(f"[Backend] Selecting entities with threshold >= {threshold}")

                # Filter entities by threshold
                threshold_entities = []
                for entity, score in top_entities_list:
                    if score >= threshold:
                        threshold_entities.append((entity, score))

                # Check if we have enough entities
                if len(threshold_entities) >= min_entities:
                    # Use threshold-based selection (limited by max_entities)
                    selected_entities = threshold_entities[:max_entities]
                    print(f"[Backend] Selected {len(selected_entities)} entities by threshold")
                else:
                    # Fallback to fixed count if threshold doesn't give enough results
                    selected_entities = top_entities_list[:backup_count]
                    print(f"[Backend] Threshold insufficient ({len(threshold_entities)} entities), using top {len(selected_entities)}")
                # Convert to context format
                for entity, score in selected_entities:
                    entities_for_context.append({"entity": entity, "similarity": score})
            else:
                # Use fixed count (legacy behavior)
                for entity, score in top_entities_list[:5]:
                    entities_for_context.append({"entity": entity, "similarity": score})
                print(f"[Backend] Using fixed selection: {len(entities_for_context)} entities")

            # Query LLM with deep thinking (múltiples pasadas) from BOTH the visible graph and full knowledge base
            print("[Backend] 🧠 Using deep thinking system with multiple steps...")

            kg_response = kg_embedding.query_llm_with_deep_thinking(
                user_message,
                entities_for_context,  # Pass the list of top entity names
                combined_triples,  # Use combined triples (visible graph + full ontology)
                None,  # model_url not needed for OpenAI API
                MODEL_NAME,
                thinking_steps=2,  # Número de pasadas de refinamiento
                use_openai_api=True,  # Flag to indicate we're using OpenAI API
            )

            # For the note, we can still refer to the single best entity for conciseness
            best_entity, best_score = top_entities_list[0]
            print(f"[Backend] KG-based response generated, focusing on entities like: {best_entity}")
            
            # Calcular y mostrar tiempo total de procesamiento
            total_time = time.time() - request_start_time
            print(f"[Backend] ⏱️ Total processing time: {total_time:.2f}s")

            # Check if the best entity is visible in the current graph
            entity_visible = any(
                best_entity == str(node.get("id", "")).split("/")[-1].split("#")[-1]
                for node in graph_data_from_request.get("nodes", [])
            )

            # Add context information based on visibility
            if entity_visible:
                visibility_note = "(Entidad principal visible en el grafo actual)"
            else:
                visibility_note = (
                    "(Entidad principal no visible actualmente en el grafo)"
                )

            # Add enhanced information about the approach used
            assistant_response_text = (
                f"{kg_response}"  # En caso de uso de la linea inferior deberia ser f"{kg_response}\n\n"
                #    f"(Nota: Esta respuesta se basa en el análisis de entidades relevantes como '{best_entity}' {visibility_note} (confianza: {best_score:.2f}))"
            )

            # Add to conversation history
            conversation_history.append(
                {"role": "user", "content": current_user_content}
            )
            conversation_history.append(
                {"role": "assistant", "content": assistant_response_text}
            )

            # Limit conversation history length
            MAX_HISTORY = 20
            if len(conversation_history) > MAX_HISTORY * 2 + 1:
                print(f"[Backend] Truncating conversation history from {len(conversation_history)} messages.")
                conversation_history = [conversation_history[0]] + conversation_history[
                    -(MAX_HISTORY * 2) :
                ]

            # Prepare response
            response_data = {"reply": assistant_response_text}

            # If @Browse tag is present, identify target URIs for the frontend to expand
            if contains_browse_tag:
                print("[Backend] Processing @Browse tag - identifying target URIs for expansion")
                
                # Get the expansion threshold from model_config.py
                from model_config import get_browse_threshold
                browse_threshold = get_browse_threshold()
                print(f"[Backend] Using @Browse expansion threshold: {browse_threshold}")

                # 1. Filter entities that meet the threshold
                target_entities = [
                    {"uri": entity, "score": score}
                    for entity, score in top_entities_list
                    if score >= browse_threshold
                ]

                # 2. If no entities meet the threshold, use the single best one as a fallback
                if not target_entities and top_entities_list:
                    best_entity, best_score = top_entities_list[0]
                    target_entities.append({"uri": best_entity, "score": best_score})
                    print(f"[Backend] No entities above threshold, falling back to top entity: {best_entity}")
                
                # 3. Add the identified target URIs to the response for the frontend
                if target_entities:
                    # The new key 'browse_targets' will be handled by index.js
                    response_data["browse_targets"] = target_entities
                    
                    # Log the targets being sent to the frontend
                    target_labels = [
                        f"{extract_label_from_uri(t['uri'])} ({t['score']:.2f})" 
                        for t in target_entities
                    ]
                    print(f"[Backend] Sending {len(target_entities)} target(s) to frontend for expansion: {', '.join(target_labels)}")
                else:
                    print("[Backend] No relevant entities found to expand for @Browse tag.")

            return jsonify(response_data)

        except Exception as e:
            print(f"[Backend] ERROR in knowledge graph processing: {e}")
            print(traceback.format_exc())
            jsonify(
                {
                    "error": f"Error processing your question using the knowledge graph: {str(e)}"
                }
            ), 500

    except Exception as e:
        print(f"[Backend] **** FATAL ERROR in /chat endpoint ****")
        print(traceback.format_exc())
        return (
            jsonify(
                {
                    "error": "An internal server error occurred. Please check backend logs."
                }
            ),
            500,
        )


@app.route("/reset", methods=["POST"])
def reset_chat():
    global conversation_history
    # Reset history using the defined SYSTEM_PROMPT constant
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("[Backend] Chat history has been reset.")
    return jsonify({"message": "Chat history reset successfully."})


@app.route("/clear_cache", methods=["POST"])
def clear_cache():
    """Endpoint para limpiar todos los archivos de cache y forzar la reinicialización del sistema"""
    global KG_EMBEDDING_READY, ALL_TRIPLES, embedding_model, entity_embeddings

    try:
        # Lista de archivos cache a eliminar
        cache_files = [
            ONTOLOGY_CACHE,
            TRIPLES_CACHE,
            MODEL_CACHE,
            EMBEDDINGS_CACHE,
            STATUS_CACHE,
            CACHE_TIMESTAMP,
        ]

        # Eliminar cada archivo si existe
        deleted_files = []
        for file_path in cache_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(os.path.basename(file_path))

        # Reiniciar variables globales
        KG_EMBEDDING_READY = False
        ALL_TRIPLES = []
        embedding_model = None
        entity_embeddings = None
        kg_embedding.ONTOLOGY_STRUCTURE = {}
        kg_embedding.CLASS_HIERARCHY = {}
        kg_embedding.CLASS_ALIASES = {}
        kg_embedding.ANNOTATION_ENRICHER = None
        kg_embedding.ENRICHED_EMBEDDINGS = None
        kg_embedding.DYNAMIC_TERMS_MAP = {}
        kg_embedding.DOMAIN_SPECIFIC_TERMS = {}

        # Respuesta de éxito
        if deleted_files:
            message = (
                f"Cache successfully cleared. Deleted files: {', '.join(deleted_files)}"
            )
            print(f"[Backend] {message}")
            return jsonify({"message": message, "status": "success"})
        else:
            message = "No cache files were found to delete."
            print(f"[Backend] {message}")
            return jsonify({"message": message, "status": "warning"})

    except Exception as e:
        error_message = f"Error clearing the cache: {str(e)}"
        print(f"[Backend] ERROR: {error_message}")
        print(traceback.format_exc())
        return jsonify({"error": error_message, "status": "error"}), 500


@app.route("/check-cache-status", methods=["POST"])
def check_cache_status():
    """Check if the cache is available for a specific graph URI"""
    global KG_EMBEDDING_READY, CURRENT_GRAPH_URI
    
    try:
        data = request.get_json()
        if not data or 'graph_uri' not in data:
            return jsonify({
                "error": "graph_uri is required",
                "cache_available": False
            }), 400
        
        requested_graph_uri = data['graph_uri']
        
        # Check if KG system is ready
        if not KG_EMBEDDING_READY:
            return jsonify({
                "cache_available": False,
                "reason": "KG embedding system not ready"
            })
        
        # Check if the current graph matches the requested one
        if CURRENT_GRAPH_URI != requested_graph_uri:
            return jsonify({
                "cache_available": False,
                "reason": "Different graph loaded",
                "current_graph": CURRENT_GRAPH_URI,
                "requested_graph": requested_graph_uri
            })
        
        # Check if essential cache files exist
        cache_files_to_check = [
            ONTOLOGY_CACHE,
            TRIPLES_CACHE,
            MODEL_CACHE,
            EMBEDDINGS_CACHE
        ]
        
        missing_files = []
        for file_path in cache_files_to_check:
            if not os.path.exists(file_path):
                missing_files.append(os.path.basename(file_path))
        
        if missing_files:
            return jsonify({
                "cache_available": False,
                "reason": "Missing cache files",
                "missing_files": missing_files
            })
        
        # All checks passed
        return jsonify({
            "cache_available": True,
            "graph_uri": CURRENT_GRAPH_URI,
            "status": "ready"
        })
        
    except Exception as e:
        print(f"[Backend] Error checking cache status: {str(e)}")
        return jsonify({
            "cache_available": False,
            "error": str(e)
        }), 500


@app.route("/available-graphs", methods=["GET"])
def get_available_graphs():
    """Endpoint to get list of available graphs from Virtuoso"""
    try:
        from virtuoso_client import VirtuosoClient
        
        # Initialize Virtuoso client
        client = VirtuosoClient({
            "endpoint": VIRTUOSO_CONFIG["endpoint"],
            "database": "default",  # For listing graphs, we don't need a specific graph
            "username": VIRTUOSO_CONFIG["username"],
            "password": VIRTUOSO_CONFIG["password"]
        })
        
        # Query to get all named graphs
        query = """
        SELECT DISTINCT ?g
        WHERE {
            GRAPH ?g {
                ?s ?p ?o .
            }
        }
        ORDER BY ?g
        """
        
        results = client.execute_query(query)
        graphs = []
        
        for result in results.get("results", {}).get("bindings", []):
            graph_uri = result["g"]["value"]
            
            # Skip system/internal graphs
            if any(skip in graph_uri.lower() for skip in ['system', 'virtrdf', 'http://www.openlinksw.com']):
                continue
                
            # Try to get some basic info about the graph
            info_query = f"""
            SELECT (COUNT(*) as ?triples)
            WHERE {{
                GRAPH <{graph_uri}> {{
                    ?s ?p ?o .
                }}
            }}
            """
            
            try:
                info_results = client.execute_query(info_query)
                triple_count = info_results.get("results", {}).get("bindings", [{}])[0].get("triples", {}).get("value", "0")
                
                # Extract a readable name from the URI
                name = graph_uri.split('/')[-1] or graph_uri.split('/')[-2] or graph_uri
                if '#' in name:
                    name = name.split('#')[-1]
                
                graphs.append({
                    "name": name.replace('_', ' ').replace('-', ' ').title(),
                    "uri": graph_uri,
                    "description": f"Grafo con {triple_count} tripletas",
                    "triples": int(triple_count) if triple_count.isdigit() else 0
                })
            except Exception as info_error:
                print(f"Error getting info for graph {graph_uri}: {info_error}")
                graphs.append({
                    "name": graph_uri.split('/')[-1] or "Unnamed Graph",
                    "uri": graph_uri,
                    "description": "Información no disponible",
                    "triples": 0
                })
        
        return jsonify({
            "graphs": graphs,
            "status": "success"
        })
        
    except Exception as e:
        error_message = f"Error fetching available graphs: {str(e)}"
        print(f"[Backend] ERROR: {error_message}")
        print(traceback.format_exc())
        return jsonify({"error": error_message, "status": "error"}), 500


@app.route("/query_virtuoso", methods=["POST"])
def query_virtuoso():
    """Endpoint to execute SPARQL queries against the selected Virtuoso graph"""
    try:
        # Check if KG system is ready
        kg_ready, error_response = check_kg_ready()
        if not kg_ready:
            return jsonify(error_response), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided", "status": "error"}), 400
        
        query = data.get('query')
        if not query:
            return jsonify({"error": "No SPARQL query provided", "status": "error"}), 400
        
        # Use the current selected graph URI
        graph_uri = data.get('virtuoso_database', CURRENT_GRAPH_URI)
        endpoint = data.get('virtuoso_endpoint', VIRTUOSO_CONFIG["endpoint"])
        username = data.get('virtuoso_username', VIRTUOSO_CONFIG["username"])
        password = data.get('virtuoso_password', VIRTUOSO_CONFIG["password"])
        
        print(f"[Backend] Executing SPARQL query on graph: {graph_uri}")
        print(f"[Backend] Query: {query[:200]}{'...' if len(query) > 200 else ''}")
        
        # Initialize Virtuoso client
        from virtuoso_client import VirtuosoClient
        client = VirtuosoClient({
            "endpoint": endpoint,
            "database": graph_uri,
            "username": username,
            "password": password
        })
        
        # Execute the query
        results = client.execute_query(query)
        
        # Process SPARQL results into the format expected by the frontend
        nodes = []
        edges = []
        literals = []
        
        if "results" in results and "bindings" in results["results"]:
            for binding in results["results"]["bindings"]:
                # Extract source, link, target from binding
                source = binding.get("source", {}).get("value", "")
                link = binding.get("link", {}).get("value", "")
                target = binding.get("target", {}).get("value", "")
                literal_type = binding.get("literal_type", {}).get("value", "")
                
                if source and link and target:
                    # Add source node if not already present
                    if source not in [n["uri"] for n in nodes]:
                        nodes.append({
                            "uri": source,
                            "label": source.split("/")[-1] or source.split("#")[-1] or source
                        })
                    
                    # Check if target is literal or URI
                    if literal_type or binding.get("target", {}).get("type") == "literal":
                        # It's a literal
                        literals.append({
                            "source": source,
                            "predicate": link,
                            "value": target,
                            "datatype": literal_type
                        })
                    else:
                        # It's a URI - add as node and edge
                        if target not in [n["uri"] for n in nodes]:
                            nodes.append({
                                "uri": target,
                                "label": target.split("/")[-1] or target.split("#")[-1] or target
                            })
                        
                        edges.append({
                            "source": source,
                            "target": target,
                            "predicate": link
                        })
        
        # Convert results to the expected format for the frontend
        return jsonify({
            "nodes": nodes,
            "edges": edges, 
            "literals": literals,
            "status": "success"
        })
        
    except Exception as e:
        error_message = f"Error executing SPARQL query: {str(e)}"
        print(f"[Backend] ERROR: {error_message}")
        print(traceback.format_exc())
        return jsonify({"error": error_message, "status": "error"}), 500


def load_ontology_to_virtuoso(file_path, graph_uri, rdf_format):
    """
    Load an ontology file into Virtuoso using SPARQL INSERT DATA with DIGEST authentication
    
    Args:
        file_path (str): Path to the RDF file
        graph_uri (str): URI of the named graph to load into
        rdf_format (str): RDF format ('turtle', 'xml', 'n3', etc.)
    
    Raises:
        Exception: If loading fails
    """
    try:
        from SPARQLWrapper import SPARQLWrapper, DIGEST
        import rdflib
        
        print(f"[Virtuoso] Loading ontology into graph: {graph_uri}")
        print(f"[Virtuoso] File: {file_path}, Format: {rdf_format}")
        
        # Parse the RDF file using rdflib
        g = rdflib.Graph()
        g.parse(file_path, format=rdf_format)
        triple_count = len(g)
        
        print(f"[Virtuoso] Parsed {triple_count} triples from ontology")
        
        # Configure Virtuoso connection using DIGEST auth and authenticated endpoint
        virtuoso_endpoint = "http://192.168.216.102:8890/sparql-auth"  # Use authenticated endpoint
        virtuoso_username = "dba"
        virtuoso_password = "password"
        
        sparql = SPARQLWrapper(virtuoso_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(virtuoso_username, virtuoso_password)
        
        # Convert graph to N-Triples for insertion
        ntriples_data = g.serialize(format='nt')
        
        # Insert data into named graph using INSERT DATA
        insert_query = f"""
        INSERT DATA {{
            GRAPH <{graph_uri}> {{
                {ntriples_data}
            }}
        }}
        """
        
        print(f"[Virtuoso] Executing INSERT DATA query...")
        sparql.setQuery(insert_query)
        sparql.method = 'POST'
        result = sparql.query()
        
        print(f"[Virtuoso] Successfully loaded ontology into graph: {graph_uri}")
        print(f"[Virtuoso] Inserted {triple_count} triples")
            
    except Exception as e:
        error_msg = f"Failed to load ontology into Virtuoso: {str(e)}"
        print(f"[Virtuoso] ERROR: {error_msg}")
        raise Exception(error_msg)


def remove_ontology_from_virtuoso(graph_uri):
    """
    Remove an ontology (named graph) from Virtuoso using SPARQL DROP GRAPH with DIGEST authentication
    
    Args:
        graph_uri (str): URI of the named graph to remove
    
    Raises:
        Exception: If removal fails
    """
    try:
        from SPARQLWrapper import SPARQLWrapper, DIGEST
        
        print(f"🗑️  [VIRTUOSO] Removing ontology graph: {graph_uri}")
        
        # Configure Virtuoso connection using DIGEST auth and authenticated endpoint
        virtuoso_endpoint = "http://192.168.216.102:8890/sparql-auth"  # Use authenticated endpoint
        virtuoso_username = "dba"
        virtuoso_password = "password"
        
        sparql = SPARQLWrapper(virtuoso_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(virtuoso_username, virtuoso_password)
        
        # Use CLEAR GRAPH instead of DROP GRAPH for better compatibility
        # CLEAR removes all triples from the graph but keeps the graph itself
        clear_query = f"CLEAR GRAPH <{graph_uri}>"
        
        print(f"📝 [VIRTUOSO] Executing CLEAR GRAPH query: {clear_query}")
        sparql.setQuery(clear_query)
        sparql.method = 'POST'
        result = sparql.query()
        
        print(f"✅ [VIRTUOSO] Successfully cleared ontology graph: {graph_uri}")
            
    except Exception as e:
        error_str = str(e)
        if "has not been explicitly created before" in error_str:
            print(f"ℹ️  [VIRTUOSO] Info: Graph {graph_uri} was already removed or never existed")
        elif "does not exist" in error_str.lower():
            print(f"ℹ️  [VIRTUOSO] Info: Graph {graph_uri} does not exist (already cleaned)")
        else:
            print(f"⚠️  [VIRTUOSO] Warning: Failed to remove ontology graph {graph_uri}: {str(e)}")
        # Don't raise exception - this is cleanup, not critical


@app.route("/upload-ontology", methods=["POST"])
def upload_ontology():
    """Endpoint to upload and process an ontology file, loading it directly into Virtuoso"""
    global CURRENT_GRAPH_URI
    
    try:
        if 'ontology' not in request.files:
            return jsonify({"error": "No file provided", "status": "error"}), 400
        
        file = request.files['ontology']
        if file.filename == '':
            return jsonify({"error": "No file selected", "status": "error"}), 400
        
        # Validate file extension
        allowed_extensions = {'.owl', '.ttl', '.rdf', '.n3', '.nt'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                "error": f"File type not supported. Use: {', '.join(allowed_extensions)}", 
                "status": "error"
            }), 400
        
        print(f"[Backend] Processing ontology upload: {file.filename}")
        print(f"[Backend] File extension detected: {file_ext}")
        
        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save file with timestamp to avoid conflicts
        timestamp = str(int(time.time()))
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(uploads_dir, safe_filename)
        file.save(file_path)
        
        print(f"[Backend] File saved to: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        print(f"[Backend] File size: {file_size} bytes")
        
        if file_size == 0:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                "error": "The uploaded file is empty", 
                "status": "error"
            }), 400
        
        # Skip rdflib validation - let Virtuoso handle parsing directly
        # This avoids issues with files that Virtuoso can parse but rdflib cannot
        
        # Generate unique graph URI for this uploaded ontology
        base_filename = os.path.splitext(file.filename)[0]
        uploaded_graph_uri = f"http://uploaded.ontology/{base_filename}_{timestamp}"
        
        # Determine format for Virtuoso based on extension
        format_mapping = {
            '.owl': 'xml',
            '.ttl': 'turtle', 
            '.rdf': 'xml',
            '.n3': 'n3',
            '.nt': 'nt'
        }
        format_used = format_mapping.get(file_ext, 'xml')
        
        print(f"[Backend] Using format {format_used} for direct Virtuoso upload")
        
        # Load directly into Virtuoso - let Virtuoso handle the parsing
        try:
            load_ontology_to_virtuoso(file_path, uploaded_graph_uri, format_used)
            
            print(f"[Backend] Successfully loaded ontology into Virtuoso graph: {uploaded_graph_uri}")
            
            # Validate success by querying Virtuoso to get triple count
            triple_count = 0
            try:
                # Query Virtuoso to count triples in the uploaded graph
                from SPARQLWrapper import SPARQLWrapper, JSON
                sparql = SPARQLWrapper("http://192.168.216.102:8890/sparql")
                sparql.setQuery(f"""
                    SELECT (COUNT(*) as ?count) 
                    WHERE {{ 
                        GRAPH <{uploaded_graph_uri}> {{ 
                            ?s ?p ?o 
                        }} 
                    }}
                """)
                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()
                
                if results["results"]["bindings"]:
                    triple_count = int(results["results"]["bindings"][0]["count"]["value"])
                    print(f"[Backend] Virtuoso reports {triple_count} triples in uploaded graph")
                
            except Exception as count_error:
                print(f"[Backend] Warning: Could not count triples: {count_error}")
                triple_count = "unknown"
            
            # Clean up local file after successful upload
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"[Backend] Cleaned up temporary file: {safe_filename}")
            
            return jsonify({
                "message": "Ontology uploaded and loaded into Virtuoso successfully",
                "graph_uri": uploaded_graph_uri,
                "filename": file.filename,
                "triples": triple_count,
                "status": "success"
            })
            
        except Exception as virtuoso_error:
            # Clean up the file if Virtuoso load failed
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                "error": f"Failed to load ontology into Virtuoso: {str(virtuoso_error)}", 
                "status": "error"
            }), 500
        
    except Exception as e:
        error_message = f"Error uploading ontology: {str(e)}"
        print(f"[Backend] ERROR: {error_message}")
        print(traceback.format_exc())
        return jsonify({"error": error_message, "status": "error"}), 500


@app.route("/cleanup-ontology", methods=["POST"])
def cleanup_ontology():
    """Endpoint to remove a temporary ontology from Virtuoso"""
    print(f"🔄 [CLEANUP] Received request - Method: {request.method}")
    print(f"🔄 [CLEANUP] Request headers: {dict(request.headers)}")
    print(f"🔄 [CLEANUP] Request origin: {request.headers.get('Origin', 'No origin header')}")
    
    try:
        data = request.get_json()
        if not data or 'graph_uri' not in data:
            return jsonify({"error": "graph_uri is required", "status": "error"}), 400
        
        graph_uri = data['graph_uri']
        
        print(f"🧹 [CLEANUP] Received cleanup request for: {graph_uri}")
        
        # Only cleanup graphs with the uploaded ontology pattern
        if not graph_uri.startswith("http://uploaded.ontology/"):
            print(f"❌ [CLEANUP] Rejected cleanup request - not an uploaded ontology: {graph_uri}")
            return jsonify({
                "error": "Can only cleanup uploaded ontologies", 
                "status": "error"
            }), 400
        
        print(f"✅ [CLEANUP] Valid uploaded ontology pattern, proceeding with cleanup: {graph_uri}")
        
        # Remove from Virtuoso
        remove_ontology_from_virtuoso(graph_uri)
        
        print(f"🎉 [CLEANUP] Successfully cleaned up ontology: {graph_uri}")
        
        return jsonify({
            "message": "Temporary ontology cleaned up successfully",
            "graph_uri": graph_uri,
            "status": "success"
        })
        
    except Exception as e:
        error_message = f"Error cleaning up ontology: {str(e)}"
        print(f"❌ [CLEANUP] ERROR: {error_message}")
        print(traceback.format_exc())
        return jsonify({"error": error_message, "status": "error"}), 500


@app.route("/initialize-progress", methods=["GET"])
def get_initialization_progress():
    """Get the current progress of KG initialization"""
    try:
        with PROGRESS_LOCK:
            progress_data = INITIALIZATION_PROGRESS.copy()
        
        return jsonify(progress_data)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"}), 500


@app.route("/start-initialization", methods=["POST"])
def start_kg_initialization():
    """Start KG initialization in background thread"""
    try:
        data = request.get_json()
        graph_uri = data.get('graph_uri')
        
        if not graph_uri:
            return jsonify({"error": "No graph URI provided", "status": "error"}), 400
        
        # Check if initialization is already running
        with PROGRESS_LOCK:
            if INITIALIZATION_PROGRESS['status'] == 'running':
                return jsonify({
                    "message": "Initialization already in progress",
                    "status": "running"
                })
        
        # Update global configuration
        global VIRTUOSO_CONFIG, CURRENT_GRAPH_URI, KG_EMBEDDING_READY
        VIRTUOSO_CONFIG["database"] = graph_uri
        CURRENT_GRAPH_URI = graph_uri
        
        print(f"[Backend] Starting KG initialization for graph: {graph_uri}")
        
        # Clear cache
        if os.path.exists(CACHE_DIR):
            import shutil
            try:
                shutil.rmtree(CACHE_DIR)
                print(f"[Backend] Cleared cache directory for graph switch")
            except Exception as cache_error:
                print(f"[Backend] Warning: Could not clear cache: {cache_error}")
        
        # Recreate cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Reset KG state and progress
        KG_EMBEDDING_READY = False
        reset_progress()
        
        # Start initialization in background thread
        def background_initialization():
            try:
                if not load_cached_data():
                    _initialize_kg_from_scratch()
            except Exception as e:
                set_progress_error(f"Background initialization failed: {str(e)}")
                traceback.print_exc()
        
        thread = threading.Thread(target=background_initialization)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": "KG initialization started in background",
            "selected_graph": graph_uri,
            "status": "started"
        })
        
    except Exception as e:
        error_message = f"Error starting initialization: {str(e)}"
        print(f"[Backend] ERROR: {error_message}")
        return jsonify({"error": error_message, "status": "error"}), 500


@app.route("/cleanup-uploaded-ontology", methods=["POST"])
def cleanup_uploaded_ontology():
    """Endpoint to remove an uploaded ontology from Virtuoso when user switches graphs"""
    try:
        data = request.get_json()
        graph_uri = data.get('graph_uri')
        
        if not graph_uri:
            return jsonify({"error": "graph_uri parameter required", "status": "error"}), 400
        
        # Only cleanup uploaded ontologies (those with our specific pattern)
        if not graph_uri.startswith("http://uploaded.ontology/"):
            return jsonify({
                "message": "Graph is not an uploaded ontology, skipping cleanup", 
                "status": "skipped"
            })
        
        print(f"[Backend] Cleaning up uploaded ontology: {graph_uri}")
        
        try:
            remove_ontology_from_virtuoso(graph_uri)
            return jsonify({
                "message": f"Uploaded ontology removed from Virtuoso",
                "graph_uri": graph_uri,
                "status": "success"
            })
        except Exception as removal_error:
            # Log warning but don't fail the request
            print(f"[Backend] Warning during ontology cleanup: {removal_error}")
            return jsonify({
                "message": f"Ontology cleanup completed with warnings",
                "warning": str(removal_error),
                "status": "success"
            })
        
    except Exception as e:
        error_message = f"Error during ontology cleanup: {str(e)}"
        print(f"[Backend] ERROR: {error_message}")
        return jsonify({"error": error_message, "status": "error"}), 500


@app.route("/select-graph", methods=["POST"])
def select_graph():
    """Endpoint to select a specific graph as the active one and start KG initialization"""
    try:
        data = request.get_json()
        graph_uri = data.get('graph_uri')
        is_temporary = data.get('is_temporary', False)
        
        if not graph_uri:
            return jsonify({"error": "No graph URI provided", "status": "error"}), 400
        
        # Check if initialization is already running
        with PROGRESS_LOCK:
            if INITIALIZATION_PROGRESS['status'] == 'running':
                return jsonify({
                    "message": "Initialization already in progress for another graph",
                    "status": "busy"
                }), 409
        
        # Update the global configuration to use the selected graph
        global VIRTUOSO_CONFIG, CURRENT_GRAPH_URI, KG_EMBEDDING_READY
        VIRTUOSO_CONFIG["database"] = graph_uri
        CURRENT_GRAPH_URI = graph_uri
        
        print(f"[Backend] Selected graph: {graph_uri}")
        print(f"[Backend] Is temporary: {is_temporary}")
        print(f"[Backend] Starting KG initialization in background...")
        
        # Clear any existing cache since we're switching graphs
        if os.path.exists(CACHE_DIR):
            import shutil
            try:
                shutil.rmtree(CACHE_DIR)
                print(f"[Backend] Cleared cache directory for graph switch")
            except Exception as cache_error:
                print(f"[Backend] Warning: Could not clear cache: {cache_error}")
        
        # Recreate cache directory
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # Reset KG state and progress
        KG_EMBEDDING_READY = False
        
        # NUEVO: Limpiar TODAS las variables globales de kg_embedding para evitar reutilización incorrecta
        global ALL_TRIPLES, embedding_model, entity_embeddings
        ALL_TRIPLES = []
        embedding_model = None
        entity_embeddings = None
        kg_embedding.ONTOLOGY_STRUCTURE = {}
        kg_embedding.CLASS_HIERARCHY = {}
        kg_embedding.CLASS_ALIASES = {}
        kg_embedding.ANNOTATION_ENRICHER = None
        kg_embedding.ENRICHED_EMBEDDINGS = None
        kg_embedding.DYNAMIC_TERMS_MAP = {}
        kg_embedding.DOMAIN_SPECIFIC_TERMS = {}
        print(f"[Backend] All KG global variables reset for graph switch")
        
        reset_progress()
        
        # Start initialization in background thread
        def background_initialization():
            try:
                # Always reset progress at the start of background initialization
                reset_progress()
                
                if not load_cached_data():
                    _initialize_kg_from_scratch()
                else:
                    # If cache exists, still show progress completion
                    with PROGRESS_LOCK:
                        INITIALIZATION_PROGRESS['status'] = 'completed'
                        INITIALIZATION_PROGRESS['progress'] = 100
                        INITIALIZATION_PROGRESS['current_step'] = 'Loaded from cache successfully'
                        INITIALIZATION_PROGRESS['logs'].append('[Sistema] Knowledge Graph loaded from cache')
            except Exception as e:
                set_progress_error(f"Background initialization failed: {str(e)}")
                traceback.print_exc()
        
        thread = threading.Thread(target=background_initialization)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": f"Graph selected and KG initialization started",
            "selected_graph": graph_uri,
            "kg_status": "initializing",
            "status": "success"
        })
        
    except Exception as e:
        error_message = f"Error selecting graph: {str(e)}"
        print(f"[Backend] ERROR: {error_message}")
        print(traceback.format_exc())
        return jsonify({"error": error_message, "status": "error"}), 500


# --- MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    import logging
    
    # Reduce Flask logging verbosity
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    # Ensure host='0.0.0.0' to be accessible on the network
    # Use debug=False in a production environment
    app.run(host="0.0.0.0", port=5000, debug=True)
