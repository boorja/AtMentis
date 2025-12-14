import numpy as np
import requests
import torch
import traceback

from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from rdflib import URIRef
from sentence_transformers import SentenceTransformer, util
from typing import Dict, Optional

# Importar funciones de configuración de modelos
from model_config import (
    get_active_embedding_model,
    get_active_kg_model,
    resolve_embedding_model as _resolve_embedding_model,
)

# Importar sistema de anotaciones enriquecido
from annotation_enrichment import (
    create_enhanced_embedding_system,
    enhanced_find_related_entities,
)

# Importar estrategia adaptativa
from adaptive_embedding_strategy import AdaptiveEmbeddingStrategy

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================

CLASS_HIERARCHY = {}  # Almacena jerarquía completa de clases
CLASS_ALIASES = {}  # Almacena alias y variantes de nombres de clases
ANNOTATION_ENRICHER = None  # Sistema de anotaciones enriquecido
ENRICHED_EMBEDDINGS = None  # Embeddings enriquecidos con anotaciones
ONTOLOGY_STRUCTURE = {}

# Variables para el modelo KGE entrenado
TRAINED_KGE_MODEL = None  # Modelo KGE entrenado con PyKEEN
KGE_ENTITY_FACTORY = None  # Factory de entidades para el modelo KGE

# Sistema dinámico de mapeo de términos - se popula automáticamente desde las anotaciones
DYNAMIC_TERMS_MAP = {}  # Se llena automáticamente desde las ontologías procesadas
DOMAIN_SPECIFIC_TERMS = {}  # Términos específicos detectados por dominio

# Mapeo de términos multilingües básicos para respaldo (solo términos genéricos universales)
BASE_MULTILINGUAL_MAP = {
    # Términos técnicos universales de ontologías
    'class': 'Class', 'clase': 'Class', 'classes': 'Class', 'clases': 'Class',
    'property': 'Property', 'propiedad': 'Property', 'properties': 'Property', 'propiedades': 'Property',
    'individual': 'Individual', 'individuo': 'Individual', 'instance': 'Individual',
    'concept': 'Concept', 'concepto': 'Concept', 'concepts': 'Concept', 'conceptos': 'Concept',
    'type': 'Type', 'tipo': 'Type', 'types': 'Type', 'tipos': 'Type',
    'category': 'Category', 'categoría': 'Category', 'categoria': 'Category',
    
    # Términos de relaciones comunes
    'relation': 'Relation', 'relación': 'Relation', 'relacion': 'Relation',
    'relationship': 'Relationship', 'part': 'Part', 'parte': 'Part',
    'component': 'Component', 'componente': 'Component',
    'element': 'Element', 'elemento': 'Element',
    
    # Términos descriptivos básicos
    'description': 'Description', 'descripción': 'Description', 'descripcion': 'Description',
    'definition': 'Definition', 'definición': 'Definition', 'definicion': 'Definition',
    'label': 'Label', 'etiqueta': 'Label', 'name': 'Name', 'nombre': 'Name',
    
    # Términos estructurales universales
    'hierarchy': 'Hierarchy', 'jerarquía': 'Hierarchy', 'jerarquia': 'Hierarchy',
    'subclass': 'Subclass', 'subclase': 'Subclass', 'parent': 'Parent',
    'child': 'Child', 'hijo': 'Child', 'children': 'Children',
    'domain': 'Domain', 'dominio': 'Domain', 'range': 'Range', 'rango': 'Range',
    'annotation': 'Annotation', 'anotación': 'Annotation', 'anotacion': 'Annotation',
}

# Stopwords multilingües expandidas para filtrado
MULTILINGUAL_STOPWORDS = {
    # Español
    'el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son',
    'con', 'para', 'al', 'del', 'los', 'las', 'una', 'sus', 'como', 'este', 'esta', 'esto', 'estos', 'estas',
    'pero', 'más', 'muy', 'todo', 'todos', 'toda', 'todas', 'otro', 'otra', 'otros', 'otras', 'cuando',
    'donde', 'quien', 'quién', 'cual', 'cuál', 'cuales', 'cuáles', 'como', 'cómo',
    
    # Inglés
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
    'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 
    'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    
    # Francés  
    'les', 'des', 'une', 'dans', 'pour', 'avec', 'par', 'sur', 'qui', 'que', 'est', 'ont',
    'son', 'ses', 'aux', 'ces', 'nos', 'vos',
    
    # Alemán
    'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'eines', 'und', 'oder', 'aber',
    'in', 'an', 'auf', 'für', 'von', 'zu', 'mit', 'bei', 'nach', 'vor', 'über', 'unter',
    
    # Términos técnicos genéricos que causan problemas
    'owl', 'rdf', 'rdfs', 'xml', 'uri', 'url', 'www', 'http', 'xsd', 'dom', 'api', 'src', 'ref'
}

# ============================================================================
# NUEVO: SYSTEM PROMPT PARA ACTIVAR EL RAZONAMIENTO DEL MODELO
# ============================================================================
DEEP_THINKING_SYSTEM_PROMPT = (
    "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem "
    "and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior "
    "to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, "
    "and then provide your solution or response to the problem."
)

# ============================================================================
# MODELO RESOLUTION Y ESTRATEGIA ADAPTATIVA
# ============================================================================


def resolve_embedding_model(model_name, content_type=None):
    """
    Resuelve el modelo de embedding apropiado, incluyendo estrategia adaptativa.

    Args:
        model_name: Nombre del modelo solicitado
        content_type: Tipo de contenido (short, medium, long, technical) para estrategia adaptativa

    Returns:
        Nombre del modelo real a usar
    """
    if (
        model_name == "adaptive_strategy"
        or model_name == "sentence-transformers/adaptive_strategy"
    ):
        try:
            # Usar estrategia adaptativa
            adaptive_strategy = AdaptiveEmbeddingStrategy()
            if content_type:
                # Si se especifica tipo de contenido, usar directamente
                if content_type in adaptive_strategy.model_configs:
                    return adaptive_strategy.model_configs[content_type]["model_name"]
                else:
                    # Fallback si el content_type no es válido
                    return adaptive_strategy.model_configs["medium"]["model_name"]
            else:
                # Fallback a modelo de alta calidad
                return adaptive_strategy.model_configs["medium"]["model_name"]
        except Exception as e:
            print(f"⚠️ Error en estrategia adaptativa: {e}")
            return "sentence-transformers/all-mpnet-base-v2"
    else:
        # Usar función base de model_config
        return _resolve_embedding_model(model_name)


def load_embedding_model(model_name, content_type=None):
    """
    Carga un modelo de embedding resolviendo primero el nombre correcto.

    Args:
        model_name: Nombre del modelo a cargar
        content_type: Tipo de contenido para estrategia adaptativa

    Returns:
        SentenceTransformer model cargado
    """
    resolved_model_name = resolve_embedding_model(model_name, content_type)
    print(f"🔄 Cargando modelo: {model_name} → {resolved_model_name}")

    try:
        return SentenceTransformer(
            resolved_model_name, device="cpu"
        )  # USO FORZADO DE CPU
    except Exception as e:
        print(f"❌ Error cargando {resolved_model_name}: {e}")
        # Fallback al modelo más básico
        fallback_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        print(f"🔄 Usando modelo fallback: {fallback_model}")
        return SentenceTransformer(fallback_model, device="cpu")  # USO FORZADO DE CPU


# ============================================================================
# FUNCIONES UTILITARIAS
# ============================================================================


def triple_element_to_string(element):
    """
    Convierte un elemento de triple que puede ser una lista o string a un formato de string consistente.

    Args:
        element: Elemento que puede ser una lista ['namespace', 'name'] o un string

    Returns:
        String normalizado para el elemento
    """
    if isinstance(element, list):
        if len(element) >= 2:
            return element[
                -1
            ]  # Usamos el último elemento que suele ser el nombre local
        elif len(element) == 1:
            return element[0]
        else:
            return ""
    else:
        return str(element)


def get_global_state():
    """Obtiene el estado actual de las variables globales del KG"""
    global CLASS_HIERARCHY, CLASS_ALIASES, ANNOTATION_ENRICHER, ENRICHED_EMBEDDINGS
    return {
        "hierarchy": dict(CLASS_HIERARCHY),
        "aliases": dict(CLASS_ALIASES),
        "enricher": ANNOTATION_ENRICHER,
        "annotation_enricher": ANNOTATION_ENRICHER,  # Key expected by test
        "embeddings": ENRICHED_EMBEDDINGS,
        "enriched_embeddings": ENRICHED_EMBEDDINGS,  # Alternative key
    }


# ============================================================================
# FUNCIONES DE OBTENCIÓN DE DATOS
# ============================================================================


def fetch_triples_from_virtuoso(virtuoso_config=None, limit=10000):
    """Fetch all triples from Virtuoso SPARQL endpoint and return them as a list of (subject, predicate, object) tuples"""
    print("Fetching all triples from Virtuoso...")

    if virtuoso_config is None:
        virtuoso_config = {
            "endpoint": "http://localhost:8890/sparql",
            "username": "dba",
            "password": "dba",
        }

    endpoint = virtuoso_config["endpoint"]
    username = virtuoso_config.get("username")
    password = virtuoso_config.get("password")
    database = virtuoso_config.get("database")

    # SPARQL query para obtener todos los triples
    # Usar FROM si se especifica una base de datos/grafo específico
    if database:
        query = f"""
        SELECT ?s ?p ?o
        FROM <{database}>
        WHERE {{
            ?s ?p ?o .
        }}
        LIMIT {limit}
        """
        print(f"Querying graph: {database}")
    else:
        query = f"""
        SELECT ?s ?p ?o
        WHERE {{
            ?s ?p ?o .
        }}
        LIMIT {limit}
        """
        print("Querying default graph")

    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/sparql-query",
    }

    auth = None
    if username and password:
        auth = (username, password)

    try:
        response = requests.post(endpoint, data=query, headers=headers, auth=auth)
        response.raise_for_status()

        data = response.json()

        triples = []
        for binding in data.get("results", {}).get("bindings", []):
            subject = binding.get("s", {}).get("value", "")
            predicate = binding.get("p", {}).get("value", "")
            object_val = binding.get("o", {}).get("value", "")

            if subject and predicate and object_val:
                triples.append((subject, predicate, object_val))

        print(f"Fetched {len(triples)} triples from Virtuoso")
        return triples

    except Exception as e:
        print(f"Error fetching triples from Virtuoso: {e}")
        traceback.print_exc()
        return []


def fetch_ontology_structure(virtuoso_config=None):
    """Fetch ontology structure from Virtuoso including classes, properties, and hierarchy"""
    print("Fetching ontology structure from Virtuoso...")

    if virtuoso_config is None:
        virtuoso_config = {
            "endpoint": "http://localhost:8890/sparql",
            "username": "dba",
            "password": "dba",
        }

    endpoint = virtuoso_config["endpoint"]
    username = virtuoso_config.get("username")
    password = virtuoso_config.get("password")
    database = virtuoso_config.get("database")

    # Preparar cláusula FROM si se especifica una base de datos
    from_clause = f"FROM <{database}>" if database else ""
    if database:
        print(f"Querying ontology structure from graph: {database}")

    # Query para obtener clases
    classes_query = f"""
    SELECT DISTINCT ?class ?label
    {from_clause}
    WHERE {{
        ?class a owl:Class .
        OPTIONAL {{ ?class rdfs:label ?label }}
    }}
    """

    # Query para obtener propiedades
    properties_query = f"""
    SELECT DISTINCT ?property ?label ?type
    {from_clause}
    WHERE {{
        {{
            ?property a owl:ObjectProperty .
            BIND("ObjectProperty" as ?type)
        }} UNION {{
            ?property a owl:DatatypeProperty .
            BIND("DatatypeProperty" as ?type)
        }}
        OPTIONAL {{ ?property rdfs:label ?label }}
    }}
    """

    # Query para obtener jerarquía
    hierarchy_query = f"""
    SELECT DISTINCT ?subclass ?superclass
    {from_clause}
    WHERE {{
        ?subclass rdfs:subClassOf ?superclass .
        ?subclass a owl:Class .
        ?superclass a owl:Class .
    }}
    """

    headers = {
        "Accept": "application/sparql-results+json",
        "Content-Type": "application/sparql-query",
    }

    auth = None
    if username and password:
        auth = (username, password)

    structure = {"classes": {}, "properties": {}, "hierarchy": {}}

    try:
        # Obtener clases
        response = requests.post(
            endpoint, data=classes_query, headers=headers, auth=auth
        )
        if response.status_code == 200:
            data = response.json()
            for binding in data.get("results", {}).get("bindings", []):
                class_uri = binding.get("class", {}).get("value", "")
                label = binding.get("label", {}).get("value", "")
                if class_uri:
                    structure["classes"][class_uri] = {
                        "label": label or class_uri.split("/")[-1]
                    }

        # Obtener propiedades
        response = requests.post(
            endpoint, data=properties_query, headers=headers, auth=auth
        )
        if response.status_code == 200:
            data = response.json()
            for binding in data.get("results", {}).get("bindings", []):
                prop_uri = binding.get("property", {}).get("value", "")
                label = binding.get("label", {}).get("value", "")
                prop_type = binding.get("type", {}).get("value", "")
                if prop_uri:
                    structure["properties"][prop_uri] = {
                        "label": label or prop_uri.split("/")[-1],
                        "type": prop_type,
                    }

        # Obtener jerarquía
        response = requests.post(
            endpoint, data=hierarchy_query, headers=headers, auth=auth
        )
        if response.status_code == 200:
            data = response.json()
            for binding in data.get("results", {}).get("bindings", []):
                subclass = binding.get("subclass", {}).get("value", "")
                superclass = binding.get("superclass", {}).get("value", "")
                if subclass and superclass:
                    if subclass not in structure["hierarchy"]:
                        structure["hierarchy"][subclass] = []
                    structure["hierarchy"][subclass].append(superclass)

        print(f"Fetched ontology structure: {len(structure['classes'])} classes, {len(structure['properties'])} properties")
        return structure

    except Exception as e:
        print(f"Error fetching ontology structure: {e}")
        return structure


def graph_data_to_triples(graph_data):
    """Convert graph data structure to list of triples"""
    triples = []

    if isinstance(graph_data, dict):
        nodes = graph_data.get("nodes", [])
        links = graph_data.get("links", [])

        # Convert nodes to triples (entity, type, class)
        for node in nodes:
            if "id" in node and "type" in node:
                entity_id = triple_element_to_string(node["id"])
                entity_type = triple_element_to_string(node["type"])
                triples.append((entity_id, "rdf:type", entity_type))

        # Convert links to triples
        for link in links:
            if "source" in link and "target" in link and "type" in link:
                source = triple_element_to_string(link["source"])
                target = triple_element_to_string(link["target"])
                predicate = triple_element_to_string(link["type"])
                triples.append((source, predicate, target))

    elif isinstance(graph_data, list):
        # Assume it's already a list of triples
        for item in graph_data:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                subject = triple_element_to_string(item[0])
                predicate = triple_element_to_string(item[1])
                obj = triple_element_to_string(item[2])
                triples.append((subject, predicate, obj))

    return triples


# ============================================================================
# FUNCIONES DE EMBEDDINGS ENRIQUECIDOS
# ============================================================================


def create_entity_embeddings(
    triples, model_name=None, use_enriched=True, virtuoso_config: Optional[Dict] = None
):
    """
    Crea embeddings para entidades. Por defecto usa el sistema enriquecido con anotaciones SKOS,
    pero puede fallar al sistema básico si hay problemas.

    Args:
        triples: Lista de triples RDF
        model_name: Nombre del modelo de embeddings (opcional)
        use_enriched: Si usar el sistema enriquecido con anotaciones (por defecto True)

    Returns:
        Tupla (modelo_embeddings, diccionario_embeddings)
    """
    global ANNOTATION_ENRICHER, ENRICHED_EMBEDDINGS

    if use_enriched:
        # Verificar si ya tenemos un sistema enriquecido funcional
        if (
            ANNOTATION_ENRICHER is not None
            and ENRICHED_EMBEDDINGS is not None
            and len(ENRICHED_EMBEDDINGS) > 0
        ):

            print("🎯 Reutilizando sistema de embeddings enriquecidos ya existente...")
            print(f"   - ENRICHED_EMBEDDINGS: {len(ENRICHED_EMBEDDINGS)} entidades")
            print(f"   - ANNOTATION_ENRICHER: ✅")

            # Reutilizar el modelo existente
            if hasattr(ANNOTATION_ENRICHER, "_get_embedding_model"):
                try:
                    embedding_model = ANNOTATION_ENRICHER._get_embedding_model()
                    return embedding_model, ENRICHED_EMBEDDINGS
                except:
                    pass

            # Si no se puede obtener el modelo, crear uno básico pero usar embeddings existentes
            basic_model = get_embedding_model(model_name)  # LIMPIAR CHECK
            return basic_model, ENRICHED_EMBEDDINGS

        try:
            print("🔍 Intentando crear embeddings enriquecidos con anotaciones SKOS...")
            return create_enriched_entity_embeddings(
                triples, model_name, virtuoso_config=virtuoso_config
            )
        except Exception as e:
            print(f"⚠️ Error con embeddings enriquecidos, usando sistema básico: {e}")
            return create_basic_entity_embeddings(triples, model_name)
    else:
        return create_basic_entity_embeddings(triples, model_name)


def create_basic_entity_embeddings(triples, model_name=None):
    """
    Crea embeddings básicos para entidades sin anotaciones enriquecidas.
    """
    if model_name is None:
        model_name = get_active_embedding_model()

    print(f"Creating basic entity embeddings using {model_name}...")

    # Usar nueva función de carga de modelos que maneja adaptive_strategy
    model = load_embedding_model(model_name)

    # Extraer entidades únicas
    entities = set()
    for triple in triples:
        subject = triple_element_to_string(triple[0])
        obj = triple_element_to_string(triple[2])
        entities.add(subject)
        entities.add(obj)

    entities = list(entities)

    # Crear embeddings
    embeddings_dict = {}
    batch_size = 32

    for i in range(0, len(entities), batch_size):
        batch_entities = entities[i : i + batch_size]
        batch_embeddings = model.encode(batch_entities, convert_to_tensor=True)

        for j, entity in enumerate(batch_entities):
            embeddings_dict[entity] = batch_embeddings[j]

    print(f"Created basic embeddings for {len(embeddings_dict)} entities")
    return model, embeddings_dict


def create_enriched_entity_embeddings(
    triples, model_name=None, virtuoso_config: Optional[Dict] = None
):
    """
    Crea embeddings enriquecidos llamando al sistema de anotaciones.
    Esta función ahora actúa como un orquestador limpio.
    """
    global ANNOTATION_ENRICHER, ENRICHED_EMBEDDINGS

    if model_name is None:
        model_name = get_active_embedding_model()

    print(f"🔍 Creating enriched embeddings with annotations using {model_name}...")

    try:
        # La función create_enhanced_embedding_system ahora se encarga de toda la lógica,
        # incluyendo la inicialización del AnnotationEnricher (dinámico o legacy).
        enriched_embeddings, enricher = create_enhanced_embedding_system(
            triples,
            existing_embeddings=None,
            embedding_model_name=model_name,
            virtuoso_config=virtuoso_config,
        )

        if enriched_embeddings and enricher:
            # Establecer las variables globales correctamente
            ENRICHED_EMBEDDINGS = enriched_embeddings
            ANNOTATION_ENRICHER = enricher

            print(f"✅ Created {len(ENRICHED_EMBEDDINGS)} enriched embeddings")
            print(f"✅ Enriched annotation system initialized")
            print(f"🎯 ANNOTATION_ENRICHER set: {ANNOTATION_ENRICHER is not None}")
            print(f"🎯 ENRICHED_EMBEDDINGS set: {ENRICHED_EMBEDDINGS is not None}")

            # Cargar el modelo de embedding para devolverlo, como se espera en el flujo
            basic_model = load_embedding_model(model_name)

            return basic_model, ENRICHED_EMBEDDINGS
        else:
            # Si la función principal de enriquecimiento falla, se lanza una excepción
            raise Exception("Could not create enriched embeddings.")

    except Exception as e:
        print(f"❌ Final error in enriched embeddings creation process: {e}")
        # Propagar la excepción para que el bloque de fallback en server.py pueda manejarla
        raise e


def get_entity_context(entity, triples, max_context=10):
    """
    Obtiene el contexto de una entidad basado en sus relaciones DIRECTAS.
    Usa una comparación exacta para evitar contaminación de contexto.
    """
    context_triples = []
    entity_str = str(entity)

    for s, p, o in triples:
        # Usar comparación EXACTA para sujeto y objeto
        if str(s) == entity_str or str(o) == entity_str:
            context_triples.append((s, p, o))
            if len(context_triples) >= max_context:
                break

    return context_triples


# ============================================================================
# FUNCIONES DE ENTRENAMIENTO KG
# ============================================================================


def train_kge(triples, kge_config=None, progress_callback=None):
    """Entrena un modelo KGE usando PyKEEN"""
    if kge_config is None:
        kge_config = {
            "model": get_active_kg_model(),
            "epochs": 100,
            "batch_size": 256,
            "learning_rate": 0.01,
        }

    # Extraer el nombre del modelo desde la configuración
    if isinstance(kge_config, dict) and "name" in kge_config:
        model_name = kge_config["name"]
        print(f"Training KGE model with {model_name}...")
    elif isinstance(kge_config, dict) and "model" in kge_config:
        model_name = kge_config["model"]
        print(f"Training KGE model with {model_name}...")
    else:
        model_name = "ComplEx"  # Fallback
        print(f"Training KGE model with fallback {model_name}...")

    try:
        # Filtrar blank nodes ANTES del entrenamiento para mejor eficiencia
        print("🔍 Filtering blank nodes from KGE training...")
        formatted_triples = []  # Inicializar la lista aquí
        blank_node_count = 0

        for triple in triples:
            subject = triple_element_to_string(triple[0])
            predicate = triple_element_to_string(triple[1])
            obj = triple_element_to_string(triple[2])

            # Filtrar blank nodes y entidades problemáticas
            if (
                is_blank_node(subject)
                or is_blank_node(obj)
                or "NamedIndividual" in subject
                or "NamedIndividual" in obj
            ):
                blank_node_count += 1
                continue

            formatted_triples.append([subject, predicate, obj])

        print(f"📊 Training filtering:")
        print(f"   ✅ {len(formatted_triples)} valid triples for training")
        print(f"   🚫 {blank_node_count} blank nodes/NamedIndividuals excluded")

        if not formatted_triples:
            print("❌ No valid triples to train after filtering")
            return None, None

        # Crear TriplesFactory
        tf = TriplesFactory.from_labeled_triples(np.array(formatted_triples))

        # Dividir en train/test para PyKEEN
        training_tf, testing_tf = tf.split([0.8, 0.2])

        # Extraer parámetros de entrenamiento desde la configuración
        epochs = kge_config.get("training", {}).get(
            "num_epochs", kge_config.get("epochs", 100)
        )
        batch_size = kge_config.get("training", {}).get(
            "batch_size", kge_config.get("batch_size", 256)
        )
        learning_rate = kge_config.get("optimizer", {}).get(
            "lr", kge_config.get("learning_rate", 0.01)
        )

        # Extraer parámetros del modelo
        model_params = kge_config.get('params', {})
        
        print(f"Training with: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        print(f"Model parameters: {model_params}")
        print(f"Training set: {training_tf.num_triples} triples, Testing set: {testing_tf.num_triples} triples")
        
        # Implementar callback de progreso si se proporciona
        training_callbacks = []
        if progress_callback:
            from pykeen.training.callbacks import TrainingCallback
            
            class ProgressCallback(TrainingCallback):
                def __init__(self, callback_fn, total_epochs):
                    self.callback_fn = callback_fn
                    self.total_epochs = total_epochs
                
                def post_epoch(self, epoch: int, epoch_loss: float, **kwargs):
                    # Llamar al callback con época actual y total
                    self.callback_fn(epoch + 1, self.total_epochs)  # +1 porque epoch es 0-indexed
            
            training_callbacks.append(ProgressCallback(progress_callback, epochs))

        # Configurar y ejecutar pipeline
        result = pipeline(
            model=model_name,
            model_kwargs=model_params,
            training=training_tf,
            testing=testing_tf,
            training_kwargs=dict(
                num_epochs=epochs,
                batch_size=batch_size,
                callbacks=training_callbacks,
            ),
            optimizer_kwargs=dict(lr=learning_rate),
        )

        print("KGE model training completed successfully")

        # Almacenar el modelo entrenado globalmente
        global TRAINED_KGE_MODEL, KGE_ENTITY_FACTORY
        TRAINED_KGE_MODEL = result.model
        KGE_ENTITY_FACTORY = training_tf
        print("✅ KGE model stored globally for use in hybrid scoring")

        return result.model, training_tf

    except Exception as e:
        print(f"Error training KGE model: {e}")
        traceback.print_exc()
        return None, None


# ============================================================================
# FUNCIONES DE ESTRUCTURA DE CONOCIMIENTO
# ============================================================================


def extract_class_hierarchy(triples):
    """Extrae la jerarquía de clases de los triples"""
    global CLASS_HIERARCHY

    hierarchy = {}

    for triple in triples:
        subject = triple_element_to_string(triple[0])
        predicate = triple_element_to_string(triple[1])
        obj = triple_element_to_string(triple[2])

        # Buscar relaciones de subclase
        if "subClassOf" in predicate or "rdfs:subClassOf" in predicate:
            # subject es subclase de obj
            if subject not in hierarchy:
                hierarchy[subject] = []
            hierarchy[subject].append(obj)

        # Buscar declaraciones de tipo clase
        elif "type" in predicate and ("Class" in obj or "owl:Class" in obj):
            if subject not in hierarchy:
                hierarchy[subject] = []

    CLASS_HIERARCHY = hierarchy
    print(f"Extracted class hierarchy with {len(hierarchy)} classes")
    return hierarchy


def build_class_aliases(triples):
    """Construye un diccionario de aliases para las clases"""
    global CLASS_ALIASES

    aliases = {}

    for triple in triples:
        subject = triple_element_to_string(triple[0])
        predicate = triple_element_to_string(triple[1])
        obj = triple_element_to_string(triple[2])

        # Buscar etiquetas
        if "label" in predicate or "rdfs:label" in predicate:
            # obj es una etiqueta para subject
            label = obj.strip('"').lower()
            aliases[label] = subject

            # Agregar variantes
            if label.endswith("s") and len(label) > 3:
                aliases[label[:-1]] = subject  # singular
            else:
                aliases[label + "s"] = subject  # plural

    CLASS_ALIASES = aliases
    print(f"Built {len(aliases)} class aliases")
    return aliases


def build_enhanced_aliases_from_annotations():
    """
    Construye aliases enriquecidos usando las anotaciones extraídas.
    Aprovecha etiquetas SKOS, comentarios, etc.
    """
    global ANNOTATION_ENRICHER

    if ANNOTATION_ENRICHER is None:
        return {}

    enhanced_aliases = {}

    # Obtener anotaciones de entidades
    entity_annotations = ANNOTATION_ENRICHER.entity_annotations

    for entity, annotations in entity_annotations.items():
        entity_name = entity.split("/")[-1].split("#")[-1]

        # Agregar todas las etiquetas como aliases
        for label in annotations.get("labels", []):
            if label and len(label.strip()) > 2:
                label_clean = label.strip().lower()
                enhanced_aliases[label_clean] = entity_name

                # Variantes plural/singular
                if label_clean.endswith("s") and len(label_clean) > 3:
                    enhanced_aliases[label_clean[:-1]] = entity_name
                else:
                    enhanced_aliases[label_clean + "s"] = entity_name

        # Extraer términos clave de descripciones
        for desc in annotations.get("descriptions", []):
            if desc and len(desc) > 10:  # Solo descripciones significativas
                key_terms = ANNOTATION_ENRICHER._extract_key_terms(desc, max_terms=3)
                for term in key_terms:
                    if len(term) > 3:  # Solo términos significativos
                        enhanced_aliases[term.lower()] = entity_name

        # Agregar ejemplos si están disponibles
        for example in annotations.get("examples", []):
            if example and len(example.strip()) > 2:
                example_clean = example.strip().lower()
                # Solo si el ejemplo es una palabra o frase corta
                if len(example_clean.split()) <= 3:
                    enhanced_aliases[example_clean] = entity_name

    print(f"🏷️ Built {len(enhanced_aliases)} enriched aliases from annotations")
    return enhanced_aliases


def normalize_query_terms(query, class_names=None, embedding_model=None, threshold=0.6):
    """
    Normaliza términos en la consulta usando similitud semántica y anotaciones enriquecidas.
    Prioriza las anotaciones SKOS/Dublin Core extraídas automáticamente.
    """
    global ANNOTATION_ENRICHER

    if class_names is None:
        # Priorizar términos extraídos de anotaciones enriquecidas
        class_names = []

        # 1. Intentar usar aliases enriquecidos con anotaciones
        if ANNOTATION_ENRICHER is not None:
            enhanced_aliases = build_enhanced_aliases_from_annotations()
            if enhanced_aliases:
                class_names = list(set(enhanced_aliases.values()))
                print(f"📝 Using {len(class_names)} classes from enriched annotations")

        # 2. Fallback a aliases básicos
        if not class_names and CLASS_ALIASES:
            class_names = list(set(CLASS_ALIASES.values()))
            print(f"📝 Using {len(class_names)} classes from basic aliases")

        # 3. Fallback a jerarquía
        if not class_names and CLASS_HIERARCHY:
            class_names = list(CLASS_HIERARCHY.keys())
            print(f"📝 Using {len(class_names)} classes from hierarchy")

        # 4. Fallback final al diccionario base multilingüe
        if not class_names:
            # Combinar diccionario base con términos dinámicos
            all_base_terms = {**BASE_MULTILINGUAL_MAP, **DYNAMIC_TERMS_MAP}
            class_names = list(set(all_base_terms.values()))
            print(f"📝 Using {len(class_names)} classes from base and dynamic dictionary")
    
    if embedding_model is None:
        try:
            model_name = get_active_embedding_model()
            embedding_model = load_embedding_model(model_name)
        except Exception as e:
            print(f"Error loading embedding model: {e}. Using dictionary fallback.")
            embedding_model = None

    normalized = query
    mapping = {}

    # 1. Intentar mapeo usando anotaciones enriquecidas primero
    if ANNOTATION_ENRICHER is not None:
        enhanced_aliases = build_enhanced_aliases_from_annotations()
        if enhanced_aliases:
            query_lower = query.lower()
            for alias_term, entity_name in enhanced_aliases.items():
                if alias_term in query_lower and entity_name not in normalized:
                    # Buscar todas las variantes del término en el query
                    original_term = None
                    for word in query.split():
                        if word.lower() == alias_term:
                            original_term = word
                            break

                    if original_term:
                        normalized = normalized.replace(original_term, entity_name)
                        mapping[original_term] = entity_name
                        print(f"📝 Mapped '{original_term}' → '{entity_name}' (from annotations)")
    
    # 2. Mapeo usando diccionarios combinados (base + dinámico + dominio específico)
    combined_terms_map = {**BASE_MULTILINGUAL_MAP, **DYNAMIC_TERMS_MAP}
    if DOMAIN_SPECIFIC_TERMS:
        combined_terms_map.update(DOMAIN_SPECIFIC_TERMS)

    for term, replacement in combined_terms_map.items():
        if term in query.lower() and replacement not in normalized:
            # Buscar la forma original del término en el query
            original_term = None
            for word in query.split():
                if word.lower() == term:
                    original_term = word
                    break

            if original_term:
                normalized = normalized.replace(original_term, replacement)
                mapping[original_term] = replacement
                print(f"📝 Mapped '{original_term}' → '{replacement}' (from combined dictionary)")
    
    # 3. Usar similitud semántica si está disponible
    if embedding_model is not None and class_names:
        try:
            mapped_terms = map_terms_with_labse(
                query, class_names, embedding_model, threshold
            )
            for original, mapped in mapped_terms.items():
                if original not in mapping and mapped not in normalized:
                    normalized = normalized.replace(original, mapped)
                    mapping[original] = mapped
                    print(f"📝 Mapped '{original}' → '{mapped}' (semantic similarity)")
        except Exception as e:
            print(f"Error in semantic mapping: {e}")

    return normalized, mapping


def map_terms_with_labse(query, class_names, embedding_model, threshold=0.6):
    """
    Mapea automáticamente los términos de la consulta a las clases de la ontología usando similitud semántica con LaBSE.
    """
    # Filtrar palabras relevantes de la consulta
    query_words = [
        w
        for w in query.lower().split()
        if len(w) > 2 and w not in MULTILINGUAL_STOPWORDS
    ]

    if not query_words or not class_names:
        return {}

    try:
        # Crear embeddings para palabras de la consulta y nombres de clases
        query_embeddings = embedding_model.encode(query_words)
        class_embeddings = embedding_model.encode(class_names)

        mapping = {}

        for i, word in enumerate(query_words):
            # Calcular similitudes con todas las clases
            similarities = util.cos_sim(query_embeddings[i], class_embeddings)[0]

            # Encontrar la clase más similar que supere el umbral
            max_sim_idx = similarities.argmax()
            max_similarity = similarities[max_sim_idx].item()

            if max_similarity >= threshold:
                mapped_class = class_names[max_sim_idx]
                mapping[word] = mapped_class
                print(f"📊 Semantic similarity: '{word}' → '{mapped_class}' (score: {max_similarity:.3f})")
        
        return mapping

    except Exception as e:
        print(f"Error in semantic mapping: {e}")
        return {}


# ============================================================================
# FUNCIONES DE BÚSQUEDA DE ENTIDADES
# ============================================================================


# En kg_embedding.py

# Importar la función necesaria al principio del archivo
from annotation_enrichment import enhanced_find_related_entities

def find_related_entities(
    query: str,
    embeddings_dict=None,
    embedding_model=None,
    top_k: int = 5,
    similarity_threshold: float = 0.3,
    visible_nodes=None,
) -> list:
    """
    Punto de entrada para la búsqueda de entidades.
    Delega la búsqueda al sistema de enriquecimiento de anotaciones.
    """
    global ANNOTATION_ENRICHER, ENRICHED_EMBEDDINGS
    from model_config import should_log

    if ANNOTATION_ENRICHER is None or ENRICHED_EMBEDDINGS is None:
        if should_log("MINIMAL"):
            print("⚠️ ADVERTENCIA: Sistema de enriquecimiento no inicializado. No se pueden buscar entidades.")
        return []

    if should_log("DEBUG"):
        print(">>> Delegando la búsqueda a `enhanced_find_related_entities`...")
    
    return enhanced_find_related_entities(
        query=query,
        enriched_embeddings=ENRICHED_EMBEDDINGS,
        enricher=ANNOTATION_ENRICHER,
        threshold=similarity_threshold,
        top_n=top_k,
        visible_nodes=visible_nodes,
    )

# ============================================================================
# FUNCIONES DE FILTRADO Y LIMPIEZA
# ============================================================================


def is_blank_node(entity_uri):
    """
    Verifica si una URI corresponde a un blank node (nodo anónimo) de RDF.
    Los blank nodes no deben aparecer como resultados principales.
    """
    if not entity_uri or not isinstance(entity_uri, str):
        return False

    # Patrones comunes de blank nodes
    blank_patterns = [
        "nodeID://",  # Virtuoso blank nodes
        "_:",  # Estándar RDF blank nodes
        "genid:",  # Generated IDs
        "bnode:",  # Blank nodes
    ]

    entity_str = str(entity_uri).lower()
    return any(pattern.lower() in entity_str for pattern in blank_patterns)


def is_unwanted_system_entity(entity_uri):
    """
    Verifica si una entidad es un concepto del sistema que no debe aparecer en resultados.
    """
    if not entity_uri or not isinstance(entity_uri, str):
        return False

    entity_str = str(entity_uri).lower()
    entity_name = entity_str.split("/")[-1].split("#")[-1]

    # Patrones de entidades del sistema que deben ser filtradas
    system_patterns = [
        "namedindividual",
        "domainConcept",
        "valuepartition",
        "owl:thing",
        "owl:nothing",
        "rdfs:resource",
        "rdfs:class",
        "http://www.w3.org/1999/",  # RDF schema
        "http://www.w3.org/2000/",  # RDFS
        "http://www.w3.org/2001/",  # XML Schema
        "http://www.w3.org/2002/07/owl#",  # OWL
    ]

    # Entidades de metadatos Dublin Core y similares (NUEVO)
    metadata_entities = [
        "contributor",
        "creator",
        "publisher",
        "subject",
        "coverage",
        "format",
        "identifier",
        "source",
        "relation",
        "rights",
        "type",  # Dublin Core type
        "date",
        "language",
        "abstract",
        "modified",
        "created",
        "issued",
        "valid",
        "available",
        "license",
        "bibliographiccitation",
        "conformsto",
        "hasformat",
        "haspart",
        "hasversion",
        "isformatof",
        "ispartof",
        "isversionof",
        "references",
        "isreferencedby",
        "requires",
        "isrequiredby",
        "replaces",
        "isreplacedby",
    ]

    # Verificar patrones del sistema
    if any(pattern in entity_str for pattern in system_patterns):
        return True

    # Verificar entidades de metadatos por nombre exacto
    if entity_name in metadata_entities:
        return True

    # Verificar URIs específicas de metadatos
    if any(
        meta_uri in entity_str
        for meta_uri in [
            "purl.org/dc/",  # Dublin Core
            "xmlns.com/foaf/",  # FOAF
            "www.w3.org/2003/01/geo/",  # Geo
            "creativecommons.org/",  # Creative Commons
            "/metadata/",  # Generic metadata paths
            "/vocab/",  # Vocabulary paths
        ]
    ):
        return True

    return False


def filter_unwanted_entities(entities_with_scores):
    """
    Filtra entidades no deseadas de una lista de entidades con puntuaciones.
    """
    from model_config import should_log

    filtered = []
    filtered_count = 0

    for entity_info in entities_with_scores:
        entity = entity_info.get("entity", "")

        if is_blank_node(entity):
            if should_log("DEBUG"):
                print(f"🚫 Filtrando blank node: {entity}")
            filtered_count += 1
            continue

        if is_unwanted_system_entity(entity):
            if should_log("DEBUG"):
                print(f"🚫 Filtrando entidad del sistema: {entity}")
            filtered_count += 1
            continue

        # Solo incluir entidades que parezcan ser contenido real de la ontología
        if entity.startswith("http") and len(entity) > 10:
            filtered.append(entity_info)

    # Solo log en modo DEBUG
    if filtered_count > 0 and should_log("DEBUG"):
        print(f"🧹 Filtradas {filtered_count} entidades no deseadas")

    return filtered


# ============================================================================
# FUNCIONES DE INICIALIZACIÓN
# ============================================================================


def initialize_kg_structure(all_triples):
    """
    Inicializa la estructura BÁSICA del grafo de conocimiento.
    La inicialización del sistema de enriquecimiento se maneja por separado.
    """
    global CLASS_HIERARCHY, CLASS_ALIASES

    print("🏗️ Inicializando estructura básica del KG...")

    # Análisis básico de la ontología
    entities = set(s for s, p, o in all_triples) | set(o for s, p, o in all_triples)
    predicates = set(p for s, p, o in all_triples)
    classes = set(s for s, p, o in all_triples if "type" in p and "Class" in o)

    # Construir jerarquía y aliases básicos
    CLASS_HIERARCHY = extract_class_hierarchy(all_triples)
    CLASS_ALIASES = build_class_aliases(all_triples)

    # Detección de términos de dominio
    update_domain_specific_terms_from_triples(all_triples)
    
    print(f"📊 Estructura básica del KG inicializada: {len(entities)} entidades, {len(predicates)} predicados, {len(classes)} clases.")
    return {
        "entities": list(entities),
        "predicates": list(predicates),
        "classes": list(classes),
    }

# =================================================================================================
# ===== NUEVAS FUNCIONES DE AYUDA PARA DEEP THINKING ADAPTATIVO =====
# =================================================================================================


def _classify_query_intent(query: str, entities_in_context: list) -> str:
    """
    Clasifica la intención de la consulta para elegir la estrategia de razonamiento adecuada.
    Versión mejorada con detección de intenciones más precisa.
    """
    query_lower = query.lower()

    # Detectar si la consulta pide información sobre tipos de datos, clases o categorías
    # Esta detección es crítica para consultas sobre taxonomías o categorizaciones
    types_keywords = [
        'tipo', 'tipos', 'clases', 'categoría', 'categorías', 'clasificación', 
        'taxonomía', 'hay', 'existen', 'cuáles son', 'cuántos', 'listado'
    ]

    # Consultas sobre qué tipos de X hay son un caso especial que debe usar análisis estructurado
    if any(k in query_lower for k in types_keywords):
        entity_count = len(entities_in_context)
        # Si tenemos suficientes entidades y hay palabras clave de tipos/categorización,
        # esto sugiere que el usuario quiere una lista o categorización completa
        if entity_count >= 1:
            print("🧠 Intent detected: TAXONOMY/TYPES QUERY")
            # Usamos análisis estructurado para consultas de tipos, es más adecuado para organizar y categorizar
            return "structured_analysis"

    # Estrategia Comparativa (la más compleja)
    compare_keywords = [
        'diferencia', 'compara', 'vs', 'versus', 'semejanzas', 
        'similitudes', 'distingue', 'contrasta', 'mejor que',
        'ventajas', 'desventajas', 'pros', 'contras'
    ]
    if (
        any(keyword in query_lower for keyword in compare_keywords)
        and len(entities_in_context) > 1
    ):
        print("🧠 Intent detected: COMPARATIVE")
        return "comparative"

    # Estrategia de Análisis Estructurado (complejidad media)
    relation_keywords = [
        'relaciona', 'conecta', 'impacta', 'causa', 'efecto', 
        'jerarquía', 'estructura', 'subclases', 'ejemplos',
        'cómo se relaciona', 'propiedades', 'características',
        'atributos', 'componentes', 'partes de'
    ]
    if any(keyword in query_lower for keyword in relation_keywords):
        print("🧠 Intent detected: RELATIONSHIP ANALYSIS")
        return "structured_analysis"

    # Estrategia de Definición Simple (la más rápida)
    definition_keywords = [
        'qué es', 'define', 'describ', 'significa', 'es un', 
        'explica', 'concepto de', 'definición', 'significado'
    ]
    if any(keyword in query_lower for keyword in definition_keywords):
        print("🧠 Intent detected: SIMPLE DEFINITION")
        return "direct_answer"

    # Consultas muy cortas (1-3 palabras) suelen ser búsquedas de definición
    word_count = len(query_lower.split())
    if word_count <= 3 and not query_lower.endswith("?"):
        print("🧠 Intent detected: SHORT QUERY (likely definition)")
        return "direct_answer"

    # Por defecto, usar el análisis estructurado si no hay una clasificación clara
    print("🧠 Default intent: STRUCTURED ANALYSIS")
    return "structured_analysis"


# Asumo que estas funciones auxiliares existen
# def get_readable_entity_name(uri, triples): ...
# def get_entity_context(uri, triples, max_context): ...

# Asegúrate de tener estas importaciones donde definas la función
from rdflib import URIRef

# Asumo que estas funciones auxiliares existen y funcionan como antes
# def get_readable_entity_name(uri, triples): ...
# def get_entity_context(uri, triples, max_context): ...


def _build_rich_context(entities_for_context, all_triples):
    """
    Construye un contexto AVANZADO y CONCISO. Omite las secciones que no tienen propiedades,
    expande completamente los nodos vinculados y diferencia correctamente entre URIs y Literales.
    """
    final_context_lines = []
    processed_uris = set()
    all_subjects = {str(s) for s, p, o in all_triples}

    primary_uris = {
        info["entity"] for info in entities_for_context if info.get("entity")
    }

    for entity_uri in primary_uris:
        if entity_uri in processed_uris:
            continue

        current_entity_block = []
        entity_name = get_readable_entity_name(entity_uri, all_triples)
        current_entity_block.append(
            f"### Entidad de Contexto: {entity_name} (URI: `{entity_uri}`)"
        )

        direct_triples = get_entity_context(entity_uri, all_triples, max_context=50)
        linked_nodes_to_expand = set()

        # 1. Relaciones Salientes (La entidad es el SUJETO)
        outgoing_relations = []
        for s, p, o in direct_triples:
            if str(s) == entity_uri:
                p_name = get_readable_entity_name(p, all_triples)
                if isinstance(o, URIRef) and str(o) in all_subjects:
                    o_name = get_readable_entity_name(o, all_triples)
                    outgoing_relations.append(
                        f"    - `{entity_name}` -> `{p_name}` -> (Nodo) `{o_name}`"
                    )
                    linked_nodes_to_expand.add(str(o))
                else:
                    outgoing_relations.append(
                        f'    - `{entity_name}` -> `{p_name}` -> (Literal) "{o}"'
                    )

        if outgoing_relations:
            current_entity_block.append("  - Propiedades y Relaciones:")
            current_entity_block.extend(outgoing_relations)

        # 2. Relaciones Entrantes (La entidad es el OBJETO)
        incoming_relations = []
        for s, p, o in direct_triples:
            if str(o) == entity_uri:
                s_name = get_readable_entity_name(s, all_triples)
                p_name = get_readable_entity_name(p, all_triples)
                incoming_relations.append(
                    f"    - `{s_name}` -> `{p_name}` -> `{entity_name}`"
                )
                linked_nodes_to_expand.add(str(s))

        if incoming_relations:
            current_entity_block.append("\n  - Usado Por (Relaciones Entrantes):")
            current_entity_block.extend(incoming_relations)

        # 3. Expansión de Nodos Vinculados
        if linked_nodes_to_expand:
            linked_details_lines = []
            for linked_uri in sorted(list(linked_nodes_to_expand)):
                if linked_uri in processed_uris:
                    continue

                node_specific_props = []
                linked_details = get_entity_context(
                    linked_uri, all_triples, max_context=25
                )

                for ls, lp, lo in linked_details:
                    if str(ls) == linked_uri:
                        p_name = get_readable_entity_name(lp, all_triples)

                        # ### CAMBIO CLAVE: LÓGICA CORREGIDA ###
                        # Usar isinstance para diferenciar URIs de Literales de forma segura.
                        if isinstance(lo, URIRef):
                            # Es una propiedad de objeto (un enlace a otro nodo).
                            o_name = get_readable_entity_name(lo, all_triples)
                            node_specific_props.append(
                                f"      - `{p_name}` -> (Nodo) `{o_name}`"
                            )
                        else:
                            # Es una propiedad de datos (un valor literal), sin importar su contenido.
                            # Se imprime tal cual, entre comillas.
                            node_specific_props.append(f'      - `{p_name}`: "{lo}"')

                if node_specific_props:
                    processed_uris.add(linked_uri)
                    linked_name = get_readable_entity_name(linked_uri, all_triples)
                    linked_details_lines.append(f"    - **Para `{linked_name}`:**")
                    linked_details_lines.extend(sorted(node_specific_props))

            if linked_details_lines:
                current_entity_block.append("\n  - Detalles de Nodos Vinculados:")
                current_entity_block.extend(linked_details_lines)

        if len(current_entity_block) > 1:
            final_context_lines.extend(current_entity_block)
            final_context_lines.append("-" * 30)

    return "\n".join(final_context_lines)


def _call_llm_step(
    prompt: str,
    model_url: str,
    model_name: str,
    temperature: float = 0.2,
    max_tokens: int = 4000,
    use_openai_api: bool = False,
) -> str:
    """
    Función de ayuda para hacer una llamada al LLM en un paso de razonamiento.
    Versión mejorada con soporte para OpenAI API, logging detallado y extracción del pensamiento del modelo.
    """
    import time
    import requests
    import traceback
    import json
    import re
    from model_config import should_log

    start_time = time.time()
    try:
        # Imprimir información del prompt para debugging (solo en modo debug)
        if should_log("DEBUG"):
            prompt_preview = prompt[:150] + "..." if len(prompt) > 150 else prompt
            print(f"[KG] Enviando prompt ({len(prompt)} caracteres): {prompt_preview}")

        # Se declara la variable que contendrá la respuesta completa del modelo
        full_content = ""

        # Usar OpenAI API si está habilitado
        if use_openai_api:
            try:
                # Intentar importar la función desde server.py
                import sys
                import os

                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from server import call_llm_api
                
                print(f"[KG] Usando OpenAI API ({model_name}, temp={temperature}, max_tokens={max_tokens})")
                full_content = call_llm_api(prompt, model_name, temperature, max_tokens)

            except ImportError as e:
                print(f"[KG] Error importando call_llm_api: {e}")
                print("[KG] Fallback a método requests local")
                use_openai_api = False

        # Usar método requests local (fallback o cuando use_openai_api=False)
        if not use_openai_api:
            params = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
            }

            print(f"[KG] Llamando al LLM local ({model_name}, temp={temperature}, max_tokens={max_tokens})")
            
            response = requests.post(
                model_url,
                headers={"Content-Type": "application/json"},
                json=params,
                timeout=90,
            )
            response.raise_for_status()

            response_data = response.json()
            full_content = response_data["choices"][0]["message"]["content"].strip()

        # === PROCESAMIENTO UNIFICADO DE LA RESPUESTA ===

        duration = time.time() - start_time
        print(f"⏱️  Paso LLM completado en {duration:.2f}s | Longitud de respuesta total: {len(full_content)} caracteres")

        # ### CAMBIO ###: Lógica para extraer, mostrar y limpiar el pensamiento
        cleaned_content = full_content
        think_match = re.search(r"<think>(.*?)</think>", full_content, re.DOTALL)

        if think_match:
            thinking_process = think_match.group(1).strip()

            preview_length = 500
            thinking_preview = thinking_process[:preview_length]
            if len(thinking_process) > preview_length:
                thinking_preview += "..."

            print("\n" + "="*25 + " VISTA PREVIA DEL PENSAMIENTO DEL MODELO " + "="*25)
            print(thinking_preview)
            print("="*24 + " (Pensamiento completo oculto para brevedad) " + "="*24 + "\n")
            
            cleaned_content = re.sub(
                r"<think>.*?</think>", "", full_content, flags=re.DOTALL
            ).strip()
            print(f"✔️  Pensamiento extraído. Longitud de respuesta final: {len(cleaned_content)} caracteres.")
        else:
            print("⚠️  No se encontraron etiquetas <think> en la respuesta del modelo.")

        # ### CAMBIO ###: Toda la lógica de logging y validación ahora usa `full_content` para analizar
        # y `cleaned_content` para el valor de retorno.

        # Logging adicional para respuestas sospechosamente cortas (se analiza la respuesta completa)
        if len(full_content) < 100:
            print(f"⚠️ [KG] RESPUESTA CORTA DETECTADA:")
            print(f"--- INICIO DE RESPUESTA CORTA ---")
            print(repr(full_content))
            print(f"--- FIN DE RESPUESTA CORTA ---")

        # Validar JSON si es necesario (se analiza la respuesta completa, ya que el JSON podría estar dentro del <think>)
        json_indicators_in_prompt = [
            "```json" in prompt.lower(),
            "`json`" in prompt.lower(),
            "formato json" in prompt.lower(),
            '"analisis_taxonomico"' in prompt.lower(),
        ]

        if any(json_indicators_in_prompt):
            print(f"🔍 [KG] Validando JSON porque se detectaron indicadores en el prompt...")
            try:
                # Buscamos el JSON tanto en la respuesta completa como en la limpia
                json_block_match = re.search(
                    r"```json\s*([\s\S]*?)\s*```", full_content
                )
                if json_block_match:
                    json_str = json_block_match.group(1).strip()
                    json.loads(json_str)
                    print(f"✅ [KG] JSON válido encontrado en bloque de código.")
                else:
                    # Intenta cargar directamente, puede que esté sin el bloque de código
                    try:
                        json.loads(cleaned_content)
                        print(f"✅ [KG] JSON válido encontrado en la respuesta limpia.")
                    except json.JSONDecodeError:
                        print(f"⚠️ [KG] No se encontró JSON válido en la respuesta.")
                        print(f"📄 [KG] CONTENIDO COMPLETO PARA DEBUG:\n{full_content}")

            except Exception as json_err:
                print(f"⚠️ [KG] Error procesando JSON: {json_err}")
                print(f"📄 [KG] CONTENIDO COMPLETO PARA DEBUG:\n{full_content}")

        return cleaned_content  # ### CAMBIO ###: Devolvemos la respuesta ya limpia

    except requests.exceptions.RequestException as e:
        error_msg = f"❌ Error en la llamada al LLM: {e}"
        print(error_msg)
        traceback.print_exc()
        return f"Error de comunicación con el modelo: {str(e)[:150]}..."
    except Exception as e:
        error_msg = f"❌ Error inesperado en el paso LLM: {e}"
        print(error_msg)
        traceback.print_exc()
        return "Error inesperado durante el procesamiento del LLM. Por favor, intenta reformular tu pregunta."


def query_llm_with_deep_thinking(
    prompt, entities_for_context, all_triples, model_url, model_name, **kwargs
):
    """
    Consulta al LLM con un proceso de "pensamiento profundo" ADAPTATIVO.
    Elige la estrategia de razonamiento más eficiente según la intención de la pregunta.
    """
    from model_config import should_log

    # from . import DEEP_THINKING_SYSTEM_PROMPT

    # Extraer el parámetro use_openai_api de kwargs
    use_openai_api = kwargs.get("use_openai_api", False)

    if should_log("INFO"):
        print(f"🧠 Iniciando consulta con deep thinking ADAPTATIVO... (OpenAI API: {use_openai_api})")

    # 1. Clasificar la intención de la pregunta para elegir una estrategia
    intent = _classify_query_intent(prompt, entities_for_context)

    # 2. Construir el contexto enriquecido una sola vez
    rich_context_str = _build_rich_context(entities_for_context, all_triples)
    if not rich_context_str:
        return (
            "No he encontrado información específica en la base de conocimiento para tu consulta. "
            "Por favor, intenta reformular la pregunta con términos más relacionados con la ontología."
        )

    # 3. Ejecutar la estrategia seleccionada
    # --- ESTRATEGIA 1: RESPUESTA DIRECTA (1 llamada al LLM) ---
    if intent == "direct_answer":
        print("🚀 Ejecutando estrategia: RESPUESTA DIRECTA")
        final_prompt = f""" {DEEP_THINKING_SYSTEM_PROMPT}
Eres un experto en la ontología proporcionada. Tu misión es responder de forma clara y concisa.

**Contexto del Grafo (ÚNICA FUENTE DE VERDAD):**
---
{rich_context_str}
---

**Pregunta del usuario:** "{prompt}"

**Instrucciones:**
1. Responde directamente a la pregunta en español basándote EXCLUSIVAMENTE en la información del grafo proporcionado.
2. Si la información del grafo es insuficiente para responder completamente, indícalo claramente en tu respuesta.
3. Usa **negritas** para destacar la entidad principal y sus propiedades clave.
4. Si hay URIs, tipos de entidades o relaciones específicas en el grafo, menciónalas explícitamente.
5. IMPORTANTE: Si el grafo menciona tipos de datos o clasificaciones, debes utilizar EXACTAMENTE esas clasificaciones.
6. No menciones términos como "contexto", "grafo" o "ontología" en tu respuesta. Actúa como si conocieras la información directamente.
7. PROHIBIDO inventar información que no esté en el grafo. Si hay poca información, limítate a ella.
"""
        return _call_llm_step(
            final_prompt,
            model_url,
            model_name,
            temperature=0.2,
            max_tokens=3000,
            use_openai_api=use_openai_api,
        )

    # --- ESTRATEGIA 2: ANÁLISIS ESTRUCTURADO (2 llamadas al LLM) ---
    elif intent == "structured_analysis":
        print("🚀 Ejecutando estrategia: ANÁLISIS ESTRUCTURADO")

        # Verificar si es una consulta de taxonomía/tipos para adaptar el prompt
        is_taxonomy_query = any(
            k in prompt.lower()
            for k in [
                "tipo",
                "tipos",
                "clases",
                "categoría",
                "categorías",
                "clasificación",
            ]
        )

        # Paso 1: Análisis interno estructurado
        if is_taxonomy_query:
            # Prompt especializado para consultas de taxonomía y tipos
            analysis_prompt = f""" 
{DEEP_THINKING_SYSTEM_PROMPT}

**Rol:** Analista de Taxonomías y Clasificaciones en Grafos de Conocimiento.
**Tarea:** Analiza el siguiente contexto del grafo para identificar tipos, categorías y clasificaciones relacionados con la pregunta del usuario.

**Contexto del Grafo (ÚNICA FUENTE DE VERDAD):**
---
{rich_context_str}
---

**Pregunta del usuario:** "{prompt}"

**INSTRUCCIÓN CRÍTICA:** Debes basar tu análisis EXCLUSIVAMENTE en la información proporcionada en el contexto del grafo.
Si hay poca información o no es clara, reporta fielmente lo que encuentres sin inventar datos.

**FORMATO DE ANÁLISIS (JSON estricto):**
```json
{{
  "pregunta_taxonomica": "Resume la pregunta del usuario en una frase concisa enfocada en la taxonomía/tipos.",
  "entidad_taxonomica_principal": "La entidad que representa la clase/tipo principal",
  "uri_principal": "URI de la entidad principal en el grafo",
  "subtipos_identificados": [
    "Lista de subtipos/subclases EXPLÍCITAMENTE mencionados en el grafo",
    "..."
  ],
  "relaciones_taxonomicas": [
    "Relación explícita 1 (ej: SubtipoA es subclase de TipoPrincipal)",
    "Relación explícita 2 (ej: SubtipoB es instancia de TipoPrincipal)",
    "..."
  ],
  "propiedades_de_clasificacion": [
    "Propiedades que determinan la clasificación (ej: hasProperty, hasCategory)",
    "..."
  ],
  "info_taxonomica_faltante": [
    "Información que sería relevante pero no está en el grafo (ej: ejemplos específicos, instancias)"
  ],
  "analisis_taxonomico": "Análisis técnico de la estructura taxonómica/jerárquica encontrada en el grafo."
}}
```
"""
        else:
            # Prompt estándar para análisis estructurado normal
            analysis_prompt = f""" 
{DEEP_THINKING_SYSTEM_PROMPT}

**Rol:** Analista de Grafos de Conocimiento.
**Tarea:** Analiza el siguiente contexto del grafo para responder a la pregunta del usuario. Genera un resumen estructurado en formato JSON.

**Contexto del Grafo (ÚNICA FUENTE DE VERDAD):**
---
{rich_context_str}
---

**Pregunta del usuario:** "{prompt}"

**INSTRUCCIÓN CRÍTICA:** Debes basar tu análisis EXCLUSIVAMENTE en la información proporcionada en el contexto del grafo. 
Si no hay suficiente información en el grafo para responder, indica los vacíos de información en tu análisis.

**Formato de Salida (JSON estricto):**
```json
{{
  "pregunta_clave": "Resume la pregunta del usuario en una frase concisa.",
  "entidades_principales": ["Lista de nombres EXACTOS de las entidades del grafo más relevantes para la pregunta."],
  "hechos_explícitos_del_grafo": [
    "Hecho 1 extraído LITERALMENTE del contexto sobre la primera entidad.",
    "Hecho 2 sobre una relación importante entre entidades.",
    "Hecho 3 sobre una jerarquía, propiedad o descripción crucial."
  ],
  "URIs_relevantes": ["Lista de URIs importantes mencionadas en el grafo"],
  "info_faltante": ["Información que sería relevante pero NO está presente en el grafo"],
  "conclusion_analitica": "Una conclusión técnica sobre cómo los hechos explícitos del grafo responden a la pregunta."
}}
```
"""
        # Usar temperature más baja para consultas de taxonomía para obtener respuestas más precisas
        taxonomy_temp = 0.1 if is_taxonomy_query else 0.1
        structured_analysis = _call_llm_step(
            analysis_prompt,
            model_url,
            model_name,
            temperature=taxonomy_temp,
            max_tokens=4000,
            use_openai_api=use_openai_api,
        )

        # Paso 2: Generar respuesta final a partir del análisis
        if is_taxonomy_query:
            # Prompt especializado para respuestas de taxonomía
            final_prompt = f"""
{DEEP_THINKING_SYSTEM_PROMPT}

**Rol:** Especialista en Taxonomías y Clasificaciones de Datos.
**Tarea:** Transforma el siguiente análisis técnico en una respuesta natural y educativa sobre tipos de datos.

**Análisis Técnico Interno (NO mostrar al usuario):**
{structured_analysis}

**Contexto Original del Grafo (para detalles adicionales - ÚNICA FUENTE DE VERDAD):**
{rich_context_str}

**Pregunta Original:** "{prompt}"

**INSTRUCCIONES CRÍTICAS:**
1. Tu respuesta debe basarse EXCLUSIVAMENTE en la información disponible en el grafo.
2. Presenta la información taxonómica de forma CONVERSACIONAL y EDUCATIVA.
3. Si la información es limitada, sé honesto sobre esto pero no te disculpes excesivamente.
4. NO INVENTES subcategorías, ejemplos, o clasificaciones que no estén en el grafo.

**ESTRUCTURA RECOMENDADA PARA RESPUESTA:**
1. Comienza con una introducción clara sobre los tipos principales encontrados.
2. Usa un estilo educativo como si estuvieras explicando a un estudiante.
3. Organiza los tipos en una estructura jerárquica clara.
4. Describe cada tipo principal de forma breve pero informativa.
5. Si hay subtipos, preséntalos como listas con viñetas bajo su tipo principal.
6. Concluye con una nota sobre la utilidad de estos tipos sin disculparte por las limitaciones.

**EJEMPLO DE FORMATO:**
En esta ontología, encontramos varios **tipos** organizados jerárquicamente:

El tipo principal es **[TipoPrincipal]**, que se divide en algunas categorías:

- **[Subtipo1]**: Estos representan [breve descripción genérica]
- **[Subtipo2]**: Estos son [breve descripción genérica]

Cada uno de estos tipos tiene propiedades específicas que definen su comportamiento...

**IMPORTANTE:**
- Usa **negritas** para destacar los nombres de tipos y categorías usando markdown.
- Evita mencionar términos técnicos como "grafo", "ontología", "análisis" o "URI".
- Si la información es limitada, menciona este hecho de manera BREVE y POSITIVA al final.
- El tono debe ser educativo, confiado y natural - como un profesor explicando conceptos.

**RECUERDA:** Usar **negritas** para nombres de tipos/clases y evitar referencias a "grafo", "contexto", o "análisis".
"""
        else:
            # Prompt estándar para respuesta estructurada normal
            final_prompt = f"""
{DEEP_THINKING_SYSTEM_PROMPT}

**Rol:** Comunicador de Conocimiento Especializado en Ontologías.
**Tarea:** Transforma el siguiente análisis técnico en una respuesta natural basada ESTRICTAMENTE en el grafo.

**Análisis Técnico Interno (NO mostrar al usuario):**
{structured_analysis}

**Contexto Original del Grafo (para detalles adicionales - ÚNICA FUENTE DE VERDAD):**
{rich_context_str}

**Pregunta Original:** "{prompt}"

**INSTRUCCIONES CRÍTICAS:**
1. Tu respuesta debe basarse EXCLUSIVAMENTE en el contexto del grafo proporcionado.
2. NO INVENTES información que no esté presente en el grafo.
3. Si el grafo tiene información limitada, admítelo claramente - es mejor una respuesta precisa pero incompleta.
4. Menciona las entidades, clases, tipos de datos o categorías EXACTAMENTE como aparecen en el grafo.
5. Si hay URIs o identificadores técnicos relevantes, incorpóralos en tu respuesta.
6. Cualquier clasificación o listado debe derivarse directamente de la estructura del grafo.

**Formato de Respuesta:**
1. Escribe una respuesta fluida y en español.
2. Resalta nombres de entidades, clases y propiedades clave en **negritas** usando markdown.
3. Organiza la información de forma lógica (párrafos, listas con viñetas) usando markdown.
4. NO menciones "análisis", "contexto", "grafo" o tu proceso interno.
5. NO incluyas frases como "según el grafo" o "según la información proporcionada".
"""
        # Usar temperatura diferente según si es taxonomía (más natural para taxonomía) o análisis general (más natural)
        final_temp = 0.3 if is_taxonomy_query else 0.3
        # print(final_prompt)
        return _call_llm_step(
            final_prompt,
            model_url,
            model_name,
            temperature=final_temp,
            max_tokens=3000,
            use_openai_api=use_openai_api,
        )

    # --- ESTRATEGIA 3: ANÁLISIS COMPARATIVO (3 llamadas al LLM) ---
    elif intent == "comparative":
        print("🚀 Ejecutando estrategia: ANÁLISIS COMPARATIVO")
        # Asegurarse de que hay al menos dos entidades para comparar
        if len(entities_for_context) < 2:
            return "Para realizar una comparación, necesito que la pregunta se refiera al menos a dos entidades claras del grafo."

        entity1_name = get_readable_entity_name(
            entities_for_context[0]["entity"], all_triples
        )
        entity2_name = get_readable_entity_name(
            entities_for_context[1]["entity"], all_triples
        )
        entity1_uri = entities_for_context[0]["entity"]
        entity2_uri = entities_for_context[1]["entity"]

        # Paso 1: Analizar la primera entidad
        analysis1_prompt = f"""
{DEEP_THINKING_SYSTEM_PROMPT}

**Tarea:** Analiza en detalle la entidad **{entity1_name}** (URI: {entity1_uri}) basándote EXCLUSIVAMENTE en el contexto del grafo proporcionado.

**Contexto del Grafo (ÚNICA FUENTE DE VERDAD):**
{rich_context_str}

**Instrucciones de Análisis:**
1. Extrae SOLO la información explícita sobre {entity1_name} que aparece en el contexto.
2. Identifica su tipo, clase, propiedades, relaciones y posición jerárquica.
3. Documenta TODAS las relaciones directas que tenga con otras entidades.
4. Si hay descripciones o anotaciones explícitas, inclúyelas.
5. NO INVENTES información que no esté presente en el contexto del grafo.
6. Si falta información importante, señala explícitamente estos vacíos.

**Formato de Salida:**
- **Tipo/Clase:** [El tipo o clase exacta de la entidad según el grafo]
- **Propiedades:** [Lista de propiedades con sus valores]
- **Relaciones:** [Lista de relaciones con otras entidades]
- **Jerarquía:** [Superclases, subclases, instancias, etc.]
- **Descripción:** [Descripciones explícitas del grafo]
- **Vacíos de Información:** [Información que sería relevante pero no está presente]
"""
        analysis1 = _call_llm_step(
            analysis1_prompt,
            model_url,
            model_name,
            temperature=0.1,
            max_tokens=3000,
            use_openai_api=use_openai_api,
        )

        # Paso 2: Analizar la segunda entidad
        analysis2_prompt = f"""
{DEEP_THINKING_SYSTEM_PROMPT}

**Tarea:** Analiza en detalle la entidad **{entity2_name}** (URI: {entity2_uri}) basándote EXCLUSIVAMENTE en el contexto del grafo proporcionado.

**Contexto del Grafo (ÚNICA FUENTE DE VERDAD):**
{rich_context_str}

**Instrucciones de Análisis:**
1. Extrae SOLO la información explícita sobre {entity2_name} que aparece en el contexto.
2. Identifica su tipo, clase, propiedades, relaciones y posición jerárquica.
3. Documenta TODAS las relaciones directas que tenga con otras entidades.
4. Si hay descripciones o anotaciones explícitas, inclúyelas.
5. NO INVENTES información que no esté presente en el contexto del grafo.
6. Si falta información importante, señala explícitamente estos vacíos.

**Formato de Salida:**
- **Tipo/Clase:** [El tipo o clase exacta de la entidad según el grafo]
- **Propiedades:** [Lista de propiedades con sus valores]
- **Relaciones:** [Lista de relaciones con otras entidades]
- **Jerarquía:** [Superclases, subclases, instancias, etc.]
- **Descripción:** [Descripciones explícitas del grafo]
- **Vacíos de Información:** [Información que sería relevante pero no está presente]
"""
        analysis2 = _call_llm_step(
            analysis2_prompt,
            model_url,
            model_name,
            temperature=0.1,
            max_tokens=3000,
            use_openai_api=use_openai_api,
        )

        # Paso 3: Comparar y sintetizar
        final_prompt = f"""
{DEEP_THINKING_SYSTEM_PROMPT}

**Rol:** Especialista en Comparación de Entidades de Ontologías.
**Tarea:** Compara las dos entidades basándote EXCLUSIVAMENTE en los análisis proporcionados del grafo.

**Análisis de {entity1_name} (URI: {entity1_uri}):**
{analysis1}

**Análisis de {entity2_name} (URI: {entity2_uri}):**
{analysis2}

**Pregunta Original:** "{prompt}"

**INSTRUCCIONES CRÍTICAS:**
1. Tu comparación debe basarse ÚNICAMENTE en la información extraída del grafo.
2. NO INVENTES similitudes o diferencias que no estén respaldadas por los análisis.
3. Si hay información limitada sobre algún aspecto, indícalo claramente.
4. Usa la terminología EXACTA que aparece en los análisis (nombres de clases, propiedades, etc.).
5. Si ambas entidades comparten un tipo, clase o tienen una relación directa entre ellas, destácalo.
6. Estructura tu respuesta de forma que muestre claramente las similitudes y diferencias.

**Formato de Respuesta:**
1. Escribe una comparación detallada en español.
2. Estructura con secciones claras: ### Similitudes, ### Diferencias, ### Relaciones Entre Ambas (si existen).
3. Usa **negritas** para resaltar conceptos clave y nombres de entidades.
4. Utiliza listas con viñetas para facilitar la lectura de los puntos de comparación.
5. NO menciones frases como "según el grafo" o "basado en el análisis".
"""
        return _call_llm_step(
            final_prompt,
            model_url,
            model_name,
            temperature=0.3,
            max_tokens=3000,
            use_openai_api=use_openai_api,
        )

    # Fallback si ninguna estrategia coincide
    return (
        "No se pudo determinar una estrategia de respuesta adecuada para esta consulta."
    )


def get_readable_entity_name(entity_uri, all_triples=None):
    """
    Obtiene un nombre legible para una entidad URI, utilizando labels cuando están disponibles.

    Args:
        entity_uri: URI de la entidad
        all_triples: Lista de triples para buscar labels (opcional)

    Returns:
        String con el nombre más legible disponible
    """
    if not entity_uri or not isinstance(entity_uri, str):
        return str(entity_uri)

    # Intentar buscar un label en los triples
    if all_triples:
        entity_str = str(entity_uri)
        label_predicates = [
            "http://www.w3.org/2000/01/rdf-schema#label",
            "rdfs:label",
            "label",
            "http://www.w3.org/2004/02/skos/core#prefLabel",
            "skos:prefLabel",
            "prefLabel",
        ]

        for triple in all_triples:
            subject = triple_element_to_string(triple[0])
            predicate = triple_element_to_string(triple[1])
            obj = triple_element_to_string(triple[2])

            if subject == entity_str:
                # Verificar si es un predicado de label
                for label_pred in label_predicates:
                    if label_pred in predicate:
                        # Limpiar el label (remover comillas si las tiene)
                        clean_label = obj.strip('"').strip("'")
                        if (
                            clean_label and len(clean_label) < 100
                        ):  # Validar que es un label razonable
                            return clean_label

    # Fallback: extraer el nombre de la URI
    return extract_label_from_uri(entity_uri)


def extract_label_from_uri(uri):
    """
    Extrae una etiqueta legible de una URI (versión mejorada)
    """
    if not uri or not isinstance(uri, str):
        return str(uri)

    # Manejar URIs especiales del sistema
    if "nodeID://" in uri or "_:" in uri or "genid:" in uri:
        # Para blank nodes, intentar extraer algo más legible
        parts = uri.split("/")
        if len(parts) > 1:
            return f"AnonymousNode_{parts[-1]}"
        return "AnonymousNode"

    # Extraer de fragmento (#) o última parte de la ruta (/)
    if "#" in uri:
        label = uri.split("#")[-1]
    elif "/" in uri:
        label = uri.split("/")[-1]
    else:
        label = uri

    # Limpiar caracteres especiales comunes
    label = label.replace("%20", " ").replace("_", " ")

    # Si el label está vacío, usar una parte anterior de la URI
    if not label or label in ["", "#", "/"]:
        parts = uri.replace("#", "/").split("/")
        for part in reversed(parts):
            if part and len(part) > 1:
                label = part.replace("%20", " ").replace("_", " ")
                break

    return label if label else uri





# ============================================================================
# DETECCIÓN AUTOMÁTICA DE TÉRMINOS DE DOMINIO
# ============================================================================


def auto_detect_domain_terms_from_triples(
    triples: list, min_frequency: int = 3
) -> dict:
    """
    Detecta automáticamente términos específicos del dominio a partir de los triples.

    Args:
        triples: Lista de triples (subject, predicate, object)
        min_frequency: Frecuencia mínima para considerar un término como específico del dominio

    Returns:
        Diccionario con términos específicos del dominio detectados
    """
    from collections import Counter
    import re

    # Extraer nombres de entidades de los triples
    entity_names = []

    for s, p, o in triples:
        # Extraer nombres de sujetos y objetos que parecen ser entidades
        for element in [s, o]:
            if isinstance(element, str):
                # Extraer nombre de la URI/IRI
                if "#" in element:
                    name = element.split("#")[-1]
                elif "/" in element:
                    name = element.split("/")[-1]
                else:
                    name = element

                # Solo considerar nombres que parecen ser clases/conceptos (empiezan con mayúscula)
                if name and len(name) > 2 and name[0].isupper():
                    entity_names.append(name)

    # Extraer palabras de los nombres de entidades usando CamelCase splitting
    domain_words = []
    for name in entity_names:
        # Dividir CamelCase en palabras
        words = re.findall(r"[A-Z][a-z]*", name)
        domain_words.extend([word.lower() for word in words if len(word) > 2])

    # Contar frecuencias
    word_counts = Counter(domain_words)

    # Filtrar stopwords y términos técnicos genéricos
    technical_stopwords = {
        'class', 'property', 'type', 'thing', 'resource', 'individual', 'concept',
        'owl', 'rdf', 'rdfs', 'xml', 'namespace', 'prefix'
    }

    # Detectar términos específicos del dominio
    domain_terms = {}
    for word, count in word_counts.items():
        if (
            count >= min_frequency
            and word not in technical_stopwords
            and word not in MULTILINGUAL_STOPWORDS
            and len(word) > 3
        ):
            # Capitalizar para ser consistente con el formato de clases
            domain_terms[word] = word.capitalize()
            # También agregar la versión plural si es común
            if word.endswith("s"):
                singular = word[:-1]
                domain_terms[singular] = word.capitalize()

    return domain_terms


def auto_populate_dynamic_terms_from_annotations():
    """
    Popula automáticamente DYNAMIC_TERMS_MAP usando las anotaciones descubiertas.
    """
    global DYNAMIC_TERMS_MAP

    if ANNOTATION_ENRICHER is None:
        return

    print("🔍 Detectando términos dinámicos desde anotaciones...")

    # Extraer términos desde las anotaciones
    new_terms = {}

    for entity, annotations in ANNOTATION_ENRICHER.entity_annotations.items():
        entity_name = entity.split("#")[-1].split("/")[-1]

        # Procesar labels
        if "labels" in annotations:
            for label in annotations["labels"]:
                if isinstance(label, str) and len(label) > 2:
                    # Normalizar el label
                    normalized_label = label.lower().strip()
                    if normalized_label not in MULTILINGUAL_STOPWORDS:
                        new_terms[normalized_label] = entity_name

                        # También agregar variantes comunes
                        words = normalized_label.split()
                        if len(words) == 1:
                            # Agregar plurales/singulares
                            if words[0].endswith("s") and len(words[0]) > 3:
                                singular = words[0][:-1]
                                new_terms[singular] = entity_name
                            elif not words[0].endswith("s"):
                                plural = words[0] + "s"
                                new_terms[plural] = entity_name

        # Procesar descripciones para extraer términos clave
        if "descriptions" in annotations:
            for desc in annotations["descriptions"]:
                if isinstance(desc, str) and len(desc) > 10:
                    # Extraer palabras clave de las descripciones
                    key_terms = extract_key_terms_from_description(desc, entity_name)
                    new_terms.update(key_terms)

    # Actualizar DYNAMIC_TERMS_MAP
    DYNAMIC_TERMS_MAP.update(new_terms)

    print(f"✅ Detectados {len(new_terms)} términos dinámicos desde anotaciones")
    return new_terms


def extract_key_terms_from_description(description: str, entity_name: str) -> dict:
    """
    Extrae términos clave de una descripción que pueden ser útiles para mapeo.

    Args:
        description: Texto de la descripción
        entity_name: Nombre de la entidad asociada

    Returns:
        Diccionario con términos clave mapeados a la entidad
    """
    import re

    # Limpiar y normalizar descripción
    clean_desc = re.sub(r"[^\w\s]", " ", description.lower())
    words = [w for w in clean_desc.split() if len(w) > 3]

    # Filtrar stopwords
    key_words = [w for w in words if w not in MULTILINGUAL_STOPWORDS]

    # Buscar patrones comunes que indican términos clave
    key_terms = {}

    # Patrones como "is a type of", "contains", "has", etc.
    patterns = [
        r"is a (\w+)",
        r"type of (\w+)",
        r"contains (\w+)",
        r"has (\w+)",
        r"includes (\w+)",
        r"consists of (\w+)",
        r"made of (\w+)",
        r"characterized by (\w+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, description.lower())
        for match in matches:
            if len(match) > 3 and match not in MULTILINGUAL_STOPWORDS:
                key_terms[match] = entity_name

    # También buscar sustantivos importantes (heurística simple)
    # Palabras que aparecen cerca del nombre de la entidad
    entity_base = entity_name.lower()
    for i, word in enumerate(key_words):
        if word in entity_base or entity_base in word:
            # Tomar palabras cercanas como potenciales términos clave
            for j in range(max(0, i - 2), min(len(key_words), i + 3)):
                if j != i and key_words[j] not in {"the", "and", "or", "with"}:
                    key_terms[key_words[j]] = entity_name

    return key_terms


def update_domain_specific_terms_from_triples(triples: list):
    """
    Actualiza DOMAIN_SPECIFIC_TERMS detectando automáticamente el dominio de la ontología.

    Args:
        triples: Lista de triples de la ontología
    """
    global DOMAIN_SPECIFIC_TERMS

    print("🎯 Detectando términos específicos del dominio...")

    # Detectar términos del dominio desde los triples
    domain_terms = auto_detect_domain_terms_from_triples(triples)

    # Combinar con términos desde anotaciones si están disponibles
    if ANNOTATION_ENRICHER:
        annotation_terms = auto_populate_dynamic_terms_from_annotations()
        domain_terms.update(annotation_terms)

    # Actualizar el diccionario global
    DOMAIN_SPECIFIC_TERMS.update(domain_terms)

    print(f"✅ Detectados {len(domain_terms)} términos específicos del dominio")

    # Mostrar algunos ejemplos
    if domain_terms:
        examples = list(domain_terms.items())[:10]
        print("📝 Ejemplos de términos detectados:")
        for term, mapped in examples:
            print(f"   '{term}' → '{mapped}'")

    return domain_terms
