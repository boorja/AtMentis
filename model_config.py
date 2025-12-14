"""
Configuración centralizada de modelos para embeddings y procesamiento de la ontología.
Este archivo permite ajustar fácilmente los modelos y parámetros sin modificar el código principal.
"""

# Configuración de modelos de embeddings
EMBEDDING_MODELS = {
    # Modelos de Sentence Transformers
    "default": "sentence-transformers/LaBSE",  # CAMBIADO: Usar LaBSE para mayor consistencia y calidad
    "fast": "distiluse-base-multilingual-cased-v2",  # Más rápido, menor calidad
    "high_quality": "sentence-transformers/LaBSE",  # Alta calidad, más lento
    # Configuraciones específicas por idioma
    "spanish_focused": "hiiamsid/sentence_similarity_spanish_es",  # Optimizado para español
    "english_focused": "sentence-transformers/all-mpnet-base-v2",  # Optimizado para inglés
    # Modelo para dominios técnicos/científicos
    "technical": "allenai/scibert_scivocab_uncased",  # Mejor para vocabulario científico
    # Estrategia adaptativa (NUEVO)
    "adaptive": "adaptive_strategy",  # Usa múltiples modelos según el contenido
}

# Configuración para la estrategia adaptativa
ADAPTIVE_STRATEGY_CONFIG = {
    "enabled": False,  # TEMPORALMENTE DESHABILITADA para evitar problemas de compatibilidad
    "auto_select": True,  # Selección automática basada en contenido
    "models": {
        "short": "sentence-transformers/LaBSE",  # Labels y textos cortos
        "medium": "sentence-transformers/all-mpnet-base-v2",  # Párrafos medianos
        "long": "sentence-transformers/all-MiniLM-L12-v2",  # Textos largos
        "technical": "sentence-transformers/msmarco-distilbert-base-v4",  # Contenido técnico
    },
    "thresholds": {
        "short_max_chars": 100,
        "medium_max_chars": 300,
        "technical_keywords_min": 2,
    },
}


# Configuración de modelos de grafos de conocimiento
KG_MODELS = {
    "default": {
        "name": "RotatE",  
        "params": {
            "embedding_dim": 128,
            "entity_initializer": "xavier_uniform_",
            "relation_initializer": "xavier_uniform_",
        },
        "training": {"num_epochs": 1000, "batch_size": 512},
        "optimizer": {"lr": 0.0005},
    },
    "complex_relations": {
        "name": "ComplEx",  # <--- Y aquí ComplEx
        "params": {
            "embedding_dim": 256,
            "entity_initializer": "xavier_normal_",
            "relation_initializer": "xavier_normal_",
        },
        "training": {"num_epochs": 15, "batch_size": 2048},
        "optimizer": {"lr": 0.0003},
    },
    "simple": {
        "name": "TransE",
        "params": {"embedding_dim": 50},
        "training": {"num_epochs": 1000, "batch_size": 512},
        "optimizer": {"lr": 0.001},
    },
}

# Parámetros para la normalización de términos
TERM_NORMALIZATION = {
    "use_lemmatization": True,  # Usar lematización para normalizar términos
    "use_stemming": False,  # Usar stemming (puede ser demasiado agresivo)
    "match_threshold": 0.85,  # Umbral de coincidencia para términos similares
}


# Obtener modelo de embeddings activo
def get_active_embedding_model():
    """Retorna el modelo de embeddings que se debe usar actualmente"""
    # Si la estrategia adaptativa está habilitada, la usamos
    if ADAPTIVE_STRATEGY_CONFIG.get("enabled", False):
        return EMBEDDING_MODELS["adaptive"]
    else:
        return EMBEDDING_MODELS["high_quality"]


# Obtener configuración de la estrategia adaptativa
def get_adaptive_strategy_config():
    """Retorna la configuración de la estrategia adaptativa"""
    return ADAPTIVE_STRATEGY_CONFIG


# Verificar si la estrategia adaptativa está habilitada
def is_adaptive_strategy_enabled():
    """Verifica si la estrategia adaptativa está habilitada"""
    return ADAPTIVE_STRATEGY_CONFIG.get("enabled", False)


# NUEVA FUNCIÓN: Resolver modelo real desde configuración
def resolve_embedding_model(model_name):
    """
    Resuelve el nombre real del modelo desde la configuración.
    Si el modelo es 'adaptive_strategy', devuelve un modelo fallback.
    """
    if (
        model_name == "adaptive_strategy"
        or model_name == "sentence-transformers/adaptive_strategy"
    ):
        # Si se pide estrategia adaptativa pero se necesita un modelo específico,
        # devolver el modelo de alta calidad como fallback
        print("⚠️ Estrategia adaptativa solicitada en contexto de modelo único, usando fallback")
        return EMBEDDING_MODELS["high_quality"]
    elif model_name in EMBEDDING_MODELS:
        return EMBEDDING_MODELS[model_name]
    else:
        return model_name


# Obtener configuración del modelo KG activo
def get_active_kg_model():
    """Retorna la configuración del modelo KG que se debe usar actualmente"""
    return KG_MODELS["complex_relations"] 


# Configuración para scoring híbrido KGE + Text Embeddings
HYBRID_SCORING_CONFIG = {
    "enabled": True,
    "weights": {
        "kge_similarity": 0.4,  # Peso para similitud de KGE (relaciones conceptuales)
        "text_similarity": 0.6,  # Peso para similitud de texto (matching directo)
    },
    "boosts": {
        "visible_nodes": 2.0,  # Multiplicador para nodos visibles
        "conceptual_match": 1.5,  # Multiplicador para matches conceptuales
        "exact_match": 1.8,  # Multiplicador para matches exactos
    },
    "normalization": {
        "method": "min_max",  # "min_max", "z_score", o "sigmoid"
        "smooth_factor": 0.1,  # Factor de suavizado para evitar divisiones por cero
    },
    "fallback": {
        "use_text_only": True,  # Si KGE no está disponible, usar solo text
        "use_kg_only": False,  # Si text no está disponible, usar solo KGE
        "min_score_threshold": 0.1,  # Umbral mínimo para considerar un resultado
    },
}


# Función para obtener configuración de scoring híbrido
def get_hybrid_scoring_config():
    """Retorna la configuración del sistema de scoring híbrido"""
    return HYBRID_SCORING_CONFIG


# Función para verificar si el scoring híbrido está habilitado
def is_hybrid_scoring_enabled():
    """Verifica si el sistema de scoring híbrido está habilitado"""
    return HYBRID_SCORING_CONFIG.get("enabled", False)


# Configuración de logging para el sistema
LOGGING_CONFIG = {
    "level": "MINIMAL",  # "DEBUG", "INFO", "VERBOSE", "MINIMAL"
    "show_query_expansion": False,
    "show_boost_details": False,  # Solo mostrar resumen
    "show_scoring_details": False,  # Solo para debugging
    "show_pre_filtering": False,
    "max_boost_logs": 0,  # Máximo número de boosts a mostrar
    "emoji_style": "minimal",  # "full", "minimal", "none"
}


# Función para obtener configuración de logging
def get_logging_config():
    """Retorna la configuración de logging"""
    return LOGGING_CONFIG


# Función para verificar nivel de logging
def should_log(level):
    """Verifica si se debe hacer log según el nivel configurado"""
    levels = {"DEBUG": 0, "INFO": 1, "VERBOSE": 2, "MINIMAL": 3}
    current_level = levels.get(LOGGING_CONFIG.get("level", "MINIMAL"), 3)
    requested_level = levels.get(level, 1)
    return requested_level >= current_level


# Configuración para el contexto del LLM
LLM_CONTEXT_CONFIG = {
    "entity_selection_mode": "threshold",  # "fixed_count" o "threshold"
    "threshold": 0.5,  # Solo entidades con score >= 0.5 van al LLM (reducido de 0.6 para incluir METEO)
    "min_entities": 1,  # Mínimo número de entidades (incluso si están por debajo del threshold)
    "max_entities": 10,  # Máximo número de entidades para evitar contextos muy largos
    "backup_fixed_count": 5,  # Si threshold no da suficientes resultados, usar top N
}


# Función para obtener configuración del contexto LLM
def get_llm_context_config():
    """Retorna la configuración del contexto del LLM"""
    return LLM_CONTEXT_CONFIG


# Función para verificar si se usa threshold o conteo fijo
def use_threshold_for_llm():
    """Verifica si se debe usar threshold para seleccionar entidades para el LLM"""
    return LLM_CONTEXT_CONFIG.get("entity_selection_mode", "threshold") == "threshold"


# Configuración para la funcionalidad @Browse
BROWSE_CONFIG = {
    "entity_threshold": 0.6,  # Solo entidades con score >= 0.6 se expanden con @Browse
    "max_total_nodes": 50,  # Máximo número de nodos en la expansión
    "semantic_validation": {
        "enabled": True,
        "top_k": 10,  # Entidades para validación semántica
        "threshold": 0.4,  # Threshold para validación semántica
        "min_validation_score": 0.3,  # Score mínimo para validar coincidencia directa
    },
}


# Función para obtener configuración de Browse
def get_browse_config():
    """Retorna la configuración de la funcionalidad @Browse"""
    return BROWSE_CONFIG


# Función para obtener el threshold de Browse
def get_browse_threshold():
    """Retorna el threshold para expandir entidades con @Browse"""
    return BROWSE_CONFIG.get("entity_threshold", 0.6)


# Función para obtener el límite máximo de nodos para Browse
def get_browse_max_nodes():
    """Retorna el máximo número de nodos para expansión @Browse"""
    return BROWSE_CONFIG.get("max_total_nodes", 15)

# ============================================================================
# CONFIGURACIÓN PARA LA BÚSQUEDA DE ENTIDADES
# ============================================================================

ENTITY_SEARCH_CONFIG = {
    # Número máximo de entidades a considerar en la búsqueda inicial
    "top_k": 20,
    
    # Umbral de similitud mínimo para que una entidad sea considerada relevante
    # Un valor más bajo permite más resultados, un valor más alto es más estricto.
    "similarity_threshold": 0.05,
    
    # Configuración de fallback si la búsqueda principal no da resultados
    "fallback_top_k": 5,
    "fallback_threshold": 0.15,
}

def get_entity_search_config():
    """Retorna la configuración para la búsqueda de entidades."""
    return ENTITY_SEARCH_CONFIG