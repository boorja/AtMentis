"""
Sistema de enriquecimiento de anotaciones para ontologías.
Detecta automáticamente anotaciones semánticas (SKOS, Dublin Core, RDFS, etc.)
y las usa para mejorar los embeddings y la búsqueda contextual.
"""

import numpy as np
import re
import torch

from collections import defaultdict
from rdflib import Namespace
from rdflib.namespace import DC, DCTERMS, RDFS, SKOS
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, List, Optional, Tuple


# Importar funciones de configuración de modelos
from model_config import (
    get_active_embedding_model,
    is_adaptive_strategy_enabled,
    get_hybrid_scoring_config,
    is_hybrid_scoring_enabled,
    get_logging_config,
    should_log,
)

# ============================================================================
# MANEJO DE DISPOSITIVOS GPU/CPU
# ============================================================================


def get_optimal_device():
    """
    Determina el dispositivo óptimo para los cálculos de embeddings.

    Returns:
        torch.device: El dispositivo a usar ('cuda' si está disponible, sino 'cpu')
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🎯 GPU detectada: {torch.cuda.get_device_name(0)}")
        return device
    else:
        device = torch.device("cpu")
        print("🎯 Usando CPU para cálculos")
        return device


def ensure_tensor_device(tensor, target_device):
    """
    Asegura que un tensor esté en el dispositivo correcto.

    Args:
        tensor: Tensor a mover
        target_device: Dispositivo destino

    Returns:
        Tensor en el dispositivo correcto
    """
    try:
        # Normalizar target_device a un objeto torch.device
        if isinstance(target_device, str):
            target_device = torch.device(target_device)
        elif not isinstance(target_device, torch.device):
            target_device = torch.device(target_device)

        # Función auxiliar para comparar dispositivos
        def devices_match(device1, device2):
            dev1_str = str(device1)
            dev2_str = str(device2)
            # Si uno es 'cuda' y el otro es 'cuda:0', son equivalentes
            if (dev1_str == "cuda" and dev2_str.startswith("cuda:")) or (
                dev2_str == "cuda" and dev1_str.startswith("cuda:")
            ):
                return True
            return dev1_str == dev2_str

        # Si ya es un tensor de PyTorch
        if hasattr(tensor, "device") and hasattr(tensor, "to"):
            # Comparar usando la función de comparación mejorada
            if not devices_match(tensor.device, target_device):
                return tensor.to(target_device)
            return tensor

        # Si es numpy array, convertir a tensor en el dispositivo correcto
        elif isinstance(tensor, np.ndarray):
            return torch.tensor(tensor, dtype=torch.float32, device=target_device)

        # Si es lista o tupla, convertir a tensor
        elif isinstance(tensor, (list, tuple)):
            return torch.tensor(tensor, dtype=torch.float32, device=target_device)

        # Si ya es tensor pero sin atributos device/to (tensor más antiguo)
        else:
            try:
                tensor_obj = torch.tensor(
                    tensor, dtype=torch.float32, device=target_device
                )
                return tensor_obj
            except Exception as e:
                print(f"⚠️ Error convirtiendo tensor: {e}")
                return tensor
    except Exception as e:
        print(f"⚠️ Error en ensure_tensor_device: {e}")
        return tensor


def normalize_embeddings_device(embeddings_dict, target_device=None):
    """
    Normaliza todos los embeddings de un diccionario al mismo dispositivo.

    Args:
        embeddings_dict: Diccionario de embeddings
        target_device: Dispositivo destino (si es None, se usa el óptimo)

    Returns:
        Diccionario con embeddings normalizados al mismo dispositivo
    """
    if target_device is None:
        target_device = get_optimal_device()

    normalized_dict = {}
    converted_count = 0

    for entity, embedding in embeddings_dict.items():
        try:
            # Convertir numpy a tensor si es necesario
            if isinstance(embedding, np.ndarray):
                embedding = torch.tensor(embedding, dtype=torch.float32)
                converted_count += 1
            
            # NUEVA VERIFICACIÓN: Asegurar que el embedding sea de tipo flotante
            if hasattr(embedding, 'dtype') and embedding.dtype.is_complex:
                # Si es complejo, tomar solo la parte real
                embedding = embedding.real.to(torch.float32)
                print(f"⚠️ Convertido embedding complejo a real para {entity}")
            elif hasattr(embedding, 'dtype') and embedding.dtype != torch.float32:
                # Forzar conversión a float32
                embedding = embedding.to(torch.float32)

            # Asegurar que está en el dispositivo correcto
            embedding = ensure_tensor_device(embedding, target_device)
            normalized_dict[entity] = embedding

        except Exception as e:
            print(f"⚠️ Error normalizando embedding para {entity}: {e}")
            # Como fallback, crear un embedding cero en el dispositivo correcto
            normalized_dict[entity] = torch.zeros(
                384, device=target_device, dtype=torch.float32
            )

    if converted_count > 0:
        print(f"🔄 Convertidos {converted_count} embeddings numpy → tensor en {target_device}")
    
    return normalized_dict


def safe_cosine_similarity(tensor1, tensor2, target_device=None):
    """
    Calcula similitud coseno asegurando que ambos tensores estén en el mismo dispositivo.

    Args:
        tensor1: Primer tensor
        tensor2: Segundo tensor
        target_device: Dispositivo destino

    Returns:
        float: Similitud coseno
    """
    try:
        if target_device is None:
            target_device = get_optimal_device()

        # Normalizar target_device a objeto torch.device
        if isinstance(target_device, str):
            target_device = torch.device(target_device)

        # Convertir primero a tensor si es numpy, luego mover al dispositivo
        if isinstance(tensor1, np.ndarray):
            tensor1 = torch.tensor(tensor1, dtype=torch.float32)
        if isinstance(tensor2, np.ndarray):
            tensor2 = torch.tensor(tensor2, dtype=torch.float32)

        # NUEVA VERIFICACIÓN: Asegurar que los tensores sean de tipo flotante
        if hasattr(tensor1, 'dtype') and tensor1.dtype.is_complex:
            # Si es complejo, tomar solo la parte real
            tensor1 = tensor1.real.to(torch.float32)
        elif hasattr(tensor1, 'dtype') and tensor1.dtype != torch.float32:
            # Forzar conversión a float32
            tensor1 = tensor1.to(torch.float32)
            
        if hasattr(tensor2, 'dtype') and tensor2.dtype.is_complex:
            # Si es complejo, tomar solo la parte real
            tensor2 = tensor2.real.to(torch.float32)
        elif hasattr(tensor2, 'dtype') and tensor2.dtype != torch.float32:
            # Forzar conversión a float32
            tensor2 = tensor2.to(torch.float32)

        # Asegurar que ambos tensores estén en el dispositivo correcto
        if hasattr(tensor1, "to"):
            tensor1 = tensor1.to(target_device)
        if hasattr(tensor2, "to"):
            tensor2 = tensor2.to(target_device)

        # Verificar dimensiones antes de calcular similitud
        if tensor1.dim() == 1:
            tensor1 = tensor1.unsqueeze(0)
        if tensor2.dim() == 1:
            tensor2 = tensor2.unsqueeze(0)

        # Calcular similitud coseno usando PyTorch directamente para mayor control
        similarity = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=-1)

        # Extraer el valor escalar
        if similarity.dim() > 0:
            return similarity.item()
        else:
            return float(similarity)

    except Exception as e:
        print(f"⚠️ Error en cálculo de similitud coseno: {e}")
        tensor1_device = getattr(tensor1, "device", "no device attr")
        tensor2_device = getattr(tensor2, "device", "no device attr")
        tensor1_dtype = getattr(tensor1, "dtype", "no dtype attr")
        tensor2_dtype = getattr(tensor2, "dtype", "no dtype attr")
        tensor1_shape = getattr(tensor1, "shape", "no shape attr")
        tensor2_shape = getattr(tensor2, "shape", "no shape attr")
        print(f"   Tensor1 device: {tensor1_device}, dtype: {tensor1_dtype}, shape: {tensor1_shape}")
        print(f"   Tensor2 device: {tensor2_device}, dtype: {tensor2_dtype}, shape: {tensor2_shape}")
        print(f"   Target device: {target_device}")
        return 0.0


# Variable global para el dispositivo actual
# CURRENT_DEVICE = get_optimal_device()
CURRENT_DEVICE = "cpu"

# ============================================================================
# SISTEMA DE DETECCIÓN DINÁMICA DE PROPIEDADES VIA SPARQL
# ============================================================================


class DynamicPropertyDetector:
    """
    Detecta automáticamente propiedades de anotación usando consultas SPARQL.
    No usa términos hardcodeados - se adapta a cualquier ontología.
    """

    def __init__(self, virtuoso_client: Optional["VirtuosoClient"] = None):
        """
        Inicializa el detector dinámico.

        Args:
            virtuoso_client: Cliente de Virtuoso para ejecutar consultas SPARQL
        """
        self.virtuoso_client = virtuoso_client
        self.discovered_properties = {
            "annotation_properties": {},  # URI -> metadata
            "datatype_properties": {},  # URI -> metadata
            "object_properties": {},  # URI -> metadata
            "literal_properties": {},  # URI -> metadata
            "usage_stats": {},  # URI -> usage_count
        }
        self.classification_weights = {
            "annotation_properties": 1.0,  # Peso máximo para annotation properties
            "datatype_properties": 0.8,  # Alto peso para datatype properties con literales
            "literal_properties": 0.6,  # Peso moderado para propiedades con literales
            "object_properties": 0.2,  # Peso bajo para object properties
        }

    def discover_properties_from_virtuoso(self) -> Dict[str, Dict]:
        """
        Descubre propiedades automáticamente usando consultas SPARQL a Virtuoso.

        Returns:
            Diccionario con propiedades clasificadas y sus metadatos
        """
        if not self.virtuoso_client:
            print("⚠️ Cliente de Virtuoso no disponible, usando sistema estático")
            return self.discovered_properties

        print("🔍 Descubriendo propiedades dinámicamente desde Virtuoso...")

        try:
            # 1. Obtener Annotation Properties
            ann_props = self.virtuoso_client.get_annotation_properties()
            for prop in ann_props:
                self.discovered_properties["annotation_properties"][prop["uri"]] = {
                    "label": prop.get("label", ""),
                    "type": "annotation",
                    "weight": self.classification_weights["annotation_properties"],
                }
            print(f"📝 Encontradas {len(ann_props)} annotation properties")

            # 2. Obtener Datatype Properties
            data_props = self.virtuoso_client.get_datatype_properties()
            for prop in data_props:
                self.discovered_properties["datatype_properties"][prop["uri"]] = {
                    "label": prop.get("label", ""),
                    "range": prop.get("range", ""),
                    "type": "datatype",
                    "weight": self.classification_weights["datatype_properties"],
                }
            print(f"📊 Encontradas {len(data_props)} datatype properties")

            # 3. Obtener Object Properties
            obj_props = self.virtuoso_client.get_object_properties()
            for prop in obj_props:
                self.discovered_properties["object_properties"][prop["uri"]] = {
                    "label": prop.get("label", ""),
                    "domain": prop.get("domain", ""),
                    "range": prop.get("range", ""),
                    "type": "object",
                    "weight": self.classification_weights["object_properties"],
                }
            print(f"🔗 Encontradas {len(obj_props)} object properties")

            # 4. Obtener propiedades con valores literales
            literal_props = self.virtuoso_client.get_literal_properties()
            for prop in literal_props:
                if (
                    prop["uri"]
                    not in self.discovered_properties["annotation_properties"]
                    and prop["uri"]
                    not in self.discovered_properties["datatype_properties"]
                ):
                    self.discovered_properties["literal_properties"][prop["uri"]] = {
                        "label": prop.get("label", ""),
                        "example_literal": prop.get("example_literal", ""),
                        "type": "literal",
                        "weight": self.classification_weights["literal_properties"],
                    }
            print(f"📄 Encontradas {len(self.discovered_properties['literal_properties'])} propiedades con literales")
            
            # 5. Obtener estadísticas de uso
            usage_stats = self.virtuoso_client.get_property_usage_stats()
            for stat in usage_stats:
                self.discovered_properties["usage_stats"][stat["predicate"]] = {
                    "usage_count": stat.get("usage_count", 0),
                    "example_value": stat.get("example_value", ""),
                }
            print(f"📈 Obtenidas estadísticas para {len(usage_stats)} propiedades")

        except Exception as e:
            print(f"⚠️ Error en descubrimiento dinámico: {e}")

        return self.discovered_properties

    def classify_property_for_annotations(self, property_uri: str) -> Dict[str, Any]:
        """
        Clasifica una propiedad para determinar su relevancia para anotaciones.

        Args:
            property_uri: URI de la propiedad a clasificar

        Returns:
            Diccionario con clasificación y metadatos
        """
        classification = {
            "uri": property_uri,
            "category": "unknown",
            "weight": 0.0,
            "annotation_type": "unknown",
            "metadata": {},
        }

        # Buscar en annotation properties (prioridad máxima)
        if property_uri in self.discovered_properties["annotation_properties"]:
            prop_data = self.discovered_properties["annotation_properties"][
                property_uri
            ]
            classification.update(
                {
                    "category": "annotation",
                    "weight": prop_data["weight"],
                    "annotation_type": self._determine_annotation_type(
                        property_uri, prop_data
                    ),
                    "metadata": prop_data,
                }
            )

        # Buscar en datatype properties (prioridad alta)
        elif property_uri in self.discovered_properties["datatype_properties"]:
            prop_data = self.discovered_properties["datatype_properties"][property_uri]
            classification.update(
                {
                    "category": "datatype",
                    "weight": prop_data["weight"],
                    "annotation_type": self._determine_annotation_type(
                        property_uri, prop_data
                    ),
                    "metadata": prop_data,
                }
            )

        # Buscar en literal properties (prioridad moderada)
        elif property_uri in self.discovered_properties["literal_properties"]:
            prop_data = self.discovered_properties["literal_properties"][property_uri]
            classification.update(
                {
                    "category": "literal",
                    "weight": prop_data["weight"],
                    "annotation_type": self._determine_annotation_type(
                        property_uri, prop_data
                    ),
                    "metadata": prop_data,
                }
            )

        # Buscar en object properties (prioridad baja)
        elif property_uri in self.discovered_properties["object_properties"]:
            prop_data = self.discovered_properties["object_properties"][property_uri]
            classification.update(
                {
                    "category": "object",
                    "weight": prop_data["weight"],
                    "annotation_type": "semantic_relation",
                    "metadata": prop_data,
                }
            )

        # Ajustar peso basado en estadísticas de uso
        if property_uri in self.discovered_properties["usage_stats"]:
            usage_data = self.discovered_properties["usage_stats"][property_uri]
            usage_count = usage_data.get("usage_count", 0)

            # Boost por uso frecuente
            if usage_count > 10:
                classification["weight"] *= 1.3
            elif usage_count > 5:
                classification["weight"] *= 1.1

            classification["metadata"]["usage_count"] = usage_count
            classification["metadata"]["example_value"] = usage_data.get(
                "example_value", ""
            )

        return classification

    def _determine_annotation_type(self, property_uri: str, prop_data: Dict) -> str:
        """
        Determina el tipo de anotación basado en la URI y metadatos.

        Args:
            property_uri: URI de la propiedad
            prop_data: Metadatos de la propiedad

        Returns:
            Tipo de anotación clasificado
        """
        uri_lower = property_uri.lower()
        label_lower = prop_data.get("label", "").lower()

        # Clasificación semántica basada en patrones universales
        if any(term in uri_lower for term in ["title", "name", "label"]):
            return "labels"
        elif any(
            term in uri_lower for term in ["description", "comment", "abstract", "note"]
        ):
            return "descriptions"
        elif any(term in uri_lower for term in ["example", "sample", "demo"]):
            return "examples"
        elif any(
            term in uri_lower for term in ["keyword", "tag", "category", "subject"]
        ):
            return "keywords"
        elif any(term in uri_lower for term in ["version", "revision", "release"]):
            return "version_info"
        elif any(term in uri_lower for term in ["license", "rights", "copyright"]):
            return "rights_info"
        elif any(term in uri_lower for term in ["language", "locale", "lang"]):
            return "language_info"
        elif any(term in uri_lower for term in ["format", "type", "media"]):
            return "format_info"
        elif any(term in uri_lower for term in ["creator", "author", "contributor"]):
            return "provenance_info"
        elif any(term in uri_lower for term in ["date", "time", "created", "modified"]):
            return "temporal_info"
        else:
            return "other_annotation"

    def get_annotation_properties_by_type(self) -> Dict[str, List[str]]:
        """
        Agrupa las propiedades descubiertas por tipo de anotación.

        Returns:
            Diccionario con URIs agrupadas por tipo de anotación
        """
        grouped_properties = {
            "labels": [],
            "descriptions": [],
            "examples": [],
            "keywords": [],
            "version_info": [],
            "rights_info": [],
            "language_info": [],
            "format_info": [],
            "provenance_info": [],
            "temporal_info": [],
            "semantic_relations": [],
            "other_annotation": [],
        }

        # Clasificar todas las propiedades descubiertas
        all_properties = set()
        for category in [
            "annotation_properties",
            "datatype_properties",
            "literal_properties",
            "object_properties",
        ]:
            all_properties.update(self.discovered_properties[category].keys())

        for prop_uri in all_properties:
            classification = self.classify_property_for_annotations(prop_uri)
            annotation_type = classification["annotation_type"]

            if annotation_type in grouped_properties:
                grouped_properties[annotation_type].append(prop_uri)
            else:
                grouped_properties["other_annotation"].append(prop_uri)

        return grouped_properties

    def get_summary(self) -> Dict[str, Any]:
        """Devuelve un resumen del descubrimiento dinámico."""
        total_properties = sum(
            len(props)
            for props in self.discovered_properties.values()
            if isinstance(props, dict)
        )

        return {
            "total_properties_discovered": total_properties,
            "annotation_properties": len(
                self.discovered_properties["annotation_properties"]
            ),
            "datatype_properties": len(
                self.discovered_properties["datatype_properties"]
            ),
            "object_properties": len(self.discovered_properties["object_properties"]),
            "literal_properties": len(self.discovered_properties["literal_properties"]),
            "properties_with_usage_stats": len(
                self.discovered_properties["usage_stats"]
            ),
            "grouped_by_annotation_type": {
                k: len(v) for k, v in self.get_annotation_properties_by_type().items()
            },
        }


# Importar estrategia adaptativa
try:
    from adaptive_embedding_strategy import (
        AdaptiveEmbeddingStrategy,
        create_adaptive_embeddings_for_annotations,
    )

    ADAPTIVE_STRATEGY_AVAILABLE = True
except ImportError:
    print("⚠️ Estrategia adaptativa no disponible, usando modelo único")
    ADAPTIVE_STRATEGY_AVAILABLE = False

# Importar cliente de Virtuoso
try:
    from virtuoso_client import VirtuosoClient

    VIRTUOSO_CLIENT_AVAILABLE = True
except ImportError:
    print("⚠️ Cliente de Virtuoso no disponible, usando sistema estático")
    VIRTUOSO_CLIENT_AVAILABLE = False

# ============================================================================
# SISTEMA DE LOGGING ORGANIZADO
# ============================================================================


class SearchLogger:
    """Sistema de logging organizado para búsquedas enriquecidas"""

    def __init__(self):
        self.config = get_logging_config()
        self.boost_count = {"conceptual": 0, "visible": 0, "exact": 0}
        self.boost_examples = []

    def log_search_start(self, query: str, total_entities: int):
        """Inicia el log de búsqueda"""
        if should_log("MINIMAL"):
            print(f"🔍 Búsqueda: '{query}' ({total_entities} entidades)")

    def log_query_expansion(self, original: str, expanded: str):
        """Log de expansión de query"""
        # Solo log en modo DEBUG
        if (
            self.config["show_query_expansion"]
            and original != expanded
            and should_log("DEBUG")
        ):
            print(f"├── Query expandida: '{expanded}'")

    def log_pre_filtering(self, relevant: int, filtered: int):
        """Log de pre-filtrado"""
        # Solo log en modo INFO o menos
        if self.config["show_pre_filtering"] and should_log("INFO"):
            print(f"├── Pre-filtrado: {relevant} relevantes, {filtered} descartadas")

    def log_scoring_mode(self, is_hybrid: bool):
        """Log del modo de scoring"""
        # Solo log en modo INFO o menos
        if should_log("INFO"):
            mode = "Híbrido (KGE + Text)" if is_hybrid else "Solo Text Embeddings"
            print(f"├── Scoring: {mode}")

    def log_boost(
        self, entity_name: str, boost_type: str, value: float, mapping: str = ""
    ):
        """Log de boost aplicado"""
        self.boost_count[boost_type] = self.boost_count.get(boost_type, 0) + 1

        if (
            self.config["show_boost_details"]
            and len(self.boost_examples) < self.config["max_boost_logs"]
        ):
            if mapping:
                self.boost_examples.append(f"{entity_name} ({mapping}): +{value:.1f}")
            else:
                self.boost_examples.append(f"{entity_name}: +{value:.1f}")

    def log_boost_summary(self):
        """Log resumen de boosts aplicados"""
        # Solo log en modo INFO o menos
        if should_log("INFO"):
            total_boosts = sum(self.boost_count.values())
            if total_boosts > 0:
                boost_details = ", ".join(
                    [f"{k}: {v}" for k, v in self.boost_count.items() if v > 0]
                )
                print(f"├── Boosts aplicados: {total_boosts} total ({boost_details})")

                if self.config["show_boost_details"] and self.boost_examples:
                    print(f"    Ejemplos: {'; '.join(self.boost_examples[:3])}")

    def log_hybrid_scoring(self, entity: str, details: dict):
        """Log de scoring híbrido detallado"""
        if self.config["show_scoring_details"] and should_log("DEBUG"):
            print(
                f"    {entity}: Text:{details['text_similarity']:.3f} + "
                f"KGE:{details['kge_similarity']:.3f} = {details['final_score']:.3f}"
            )

    def log_results(self, results: list, filtered_count: int = 0):
        """Log de resultados finales"""
        if should_log("MINIMAL"):
            print(f"✅ Encontradas {len(results)} entidades relevantes")

            # Solo mostrar top 3 en modo MINIMAL
            if results and should_log("MINIMAL"):
                # Normalizar scores para mostrar en el log
                scores = [score for _, score in results]
                if scores:
                    min_score = min(scores)
                    max_score = max(scores)

                    # Mostrar resultados con scores normalizados si es necesario
                    for i, (entity, score) in enumerate(results[:3], 1):
                        entity_name = entity.split("/")[-1].split("#")[-1]

                        # Normalizar score para display si supera 1
                        if max_score > 1.0 and max_score != min_score:
                            normalized_score = (score - min_score) / (
                                max_score - min_score
                            )
                            print(f"   {i}. {entity_name}: {normalized_score:.3f}")
                        else:
                            print(f"   {i}. {entity_name}: {score:.3f}")

                    if len(results) > 3:
                        print(f"   ... y {len(results) - 3} más")


# ============================================================================
# NAMESPACES Y CONFIGURACIÓN
# ============================================================================

# Namespaces estándar para anotaciones
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
DC = Namespace("http://purl.org/dc/elements/1.1/")
DCTERMS = Namespace("http://purl.org/dc/terms/")

# SISTEMA DINÁMICO DE DETECCIÓN DE PROPIEDADES
# Estas son categorías base que se expandirán automáticamente con consultas SPARQL

# Predicados base para clasificación automática (fallback si SPARQL no está disponible)
BASE_ANNOTATION_PREDICATES = {
    # Etiquetas y nombres
    "labels": {
        str(RDFS.label),
        str(SKOS.prefLabel),
        str(SKOS.altLabel),
        "http://xmlns.com/foaf/0.1/name",
        "https://w3id.org/idsa/core/title",
    },
    # Descripciones y definiciones
    "descriptions": {
        str(RDFS.comment),
        str(SKOS.definition),
        str(DC.description),
        str(DCTERMS.description),
        "https://w3id.org/idsa/core/description",
    },
    # Palabras clave y metadatos
    "keywords": {
        "https://w3id.org/idsa/core/keyword",
        "http://purl.org/dc/elements/1.1/subject",
        "http://purl.org/dc/terms/subject",
    },
    # Información técnica
    "technical": {
        "https://w3id.org/idsa/core/version",
        "https://w3id.org/idsa/core/license",
        "https://w3id.org/idsa/core/language",
        "http://www.w3.org/ns/dcat#version",
        "http://www.w3.org/ns/dcat#mediaType",
    },
    # Endpoints y URLs
    "endpoints": {
        "http://www.w3.org/ns/dcat#endpointURL",
        "http://www.w3.org/ns/dcat#endpointUrl",
        "http://www.w3.org/ns/dcat#endpointDescription",
    },
    # Relaciones jerárquicas SKOS
    "hierarchical": {
        str(SKOS.broader),
        str(SKOS.narrower),
        str(RDFS.subClassOf),
    },
    # Relaciones semánticas
    "semantic": {
        str(SKOS.related),
        str(SKOS.exactMatch),
        str(SKOS.closeMatch),
    },
}

# Cache global para propiedades detectadas dinámicamente
DYNAMIC_ANNOTATION_PREDICATES = None
SPARQL_DETECTION_ENABLED = True

# Alias para compatibilidad
ANNOTATION_PREDICATES = BASE_ANNOTATION_PREDICATES

# Stopwords multilingües para filtrar en descripciones
MULTILINGUAL_STOPWORDS = {
    'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
           'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
           'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
           'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
           'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'},
    'es': {'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'e', 'o', 'u',
           'pero', 'sino', 'que', 'de', 'del', 'al', 'en', 'con', 'por', 'para',
           'sin', 'sobre', 'bajo', 'entre', 'desde', 'hasta', 'durante', 'mediante',
           'es', 'son', 'era', 'eran', 'ser', 'estar', 'está', 'están', 'estaba',
           'estaban', 'fue', 'fueron', 'ha', 'han', 'había', 'habían', 'he', 'hemos',
           'habéis', 'esto', 'esta', 'este', 'estos', 'estas', 'eso', 'esa', 'ese'},
    'de': {'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'eines',
           'und', 'oder', 'aber', 'in', 'an', 'auf', 'für', 'von', 'zu', 'mit',
           'bei', 'nach', 'vor', 'über', 'unter', 'zwischen', 'durch', 'ohne',
           'ist', 'sind', 'war', 'waren', 'sein', 'haben', 'hat', 'hatte', 'hatten'},
    'fr': {'le', 'la', 'les', 'un', 'une', 'des', 'du', 'de', 'et', 'ou', 'mais',
           'dans', 'sur', 'avec', 'par', 'pour', 'sans', 'sous', 'entre', 'depuis',
           'est', 'sont', 'était', 'étaient', 'être', 'avoir', 'a', 'ont', 'avait',
           'avaient', 'ce', 'cette', 'ces', 'cet', 'il', 'elle', 'ils', 'elles'}
}

# ============================================================================
# SISTEMA GENÉRICO DE NORMALIZACIÓN DE NOMBRES DE ENTIDADES
# ============================================================================


def normalize_entity_name_for_embedding(entity_name: str) -> str:
    """
    Normaliza nombres de entidades para mejorar el embedding similarity.
    Funciona genéricamente con cualquier ontología.

    Args:
        entity_name: Nombre de la entidad (ej: "VegetarianPizza", "NonSmoker")

    Returns:
        Nombre normalizado (ej: "vegetarian pizza", "non smoker")
    """
    if not entity_name:
        return entity_name

    # 1. Separar CamelCase
    normalized = _split_camel_case(entity_name)

    # 2. Expandir prefijos comunes
    normalized = _expand_common_prefixes(normalized)

    # 3. Expandir sufijos comunes
    normalized = _expand_common_suffixes(normalized)

    # 4. Normalizar espacios y caracteres
    normalized = re.sub(r"[_\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.lower().strip()

    return normalized


def _split_camel_case(text: str) -> str:
    """
    Separa palabras en CamelCase de forma inteligente.
    Ej: "VegetarianPizza" → "Vegetarian Pizza"
    """
    # Insertar espacio antes de mayúsculas que siguen a minúsculas
    # o antes de mayúsculas seguidas de minúsculas
    result = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    result = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", result)
    return result


def _expand_common_prefixes(text: str) -> str:
    """
    Expande prefijos comunes encontrados en ontologías.
    Detecta automáticamente patrones sin hardcodear dominios específicos.
    """
    # Diccionario de prefijos comunes en ontologías
    prefix_expansions = {
        "non": "non ",
        "anti": "anti ",
        "multi": "multi ",
        "sub": "sub ",
        "super": "super ",
        "pre": "pre ",
        "post": "post ",
        "inter": "inter ",
        "intra": "intra ",
        "ultra": "ultra ",
        "semi": "semi ",
        "pseudo": "pseudo ",
        "quasi": "quasi ",
        "micro": "micro ",
        "macro": "macro ",
        "meta": "meta ",
        "proto": "proto ",
        "neo": "neo ",
        "auto": "auto ",
        "co": "co ",
        "bi": "bi ",
        "tri": "tri ",
        "uni": "uni ",
        "mono": "mono ",
        "poly": "poly ",
        "omni": "omni ",
    }

    text_lower = text.lower()
    for prefix, expansion in prefix_expansions.items():
        if text_lower.startswith(prefix) and len(text) > len(prefix):
            # Verificar que no sea parte de una palabra más larga
            next_char_pos = len(prefix)
            if next_char_pos < len(text):
                next_char = text[next_char_pos]
                if next_char.isupper() or next_char in ["_", "-"]:
                    # Es realmente un prefijo
                    rest = text[len(prefix) :]
                    return expansion + rest

    return text


def _expand_common_suffixes(text: str) -> str:
    """
    Expande sufijos comunes encontrados en ontologías.
    """
    # Diccionario de sufijos comunes en ontologías
    suffix_expansions = {
        "ness": " ness",
        "ment": " ment",
        "tion": " tion",
        "sion": " sion",
        "able": " able",
        "ible": " ible",
        "ful": " ful",
        "less": " less",
        "ish": " ish",
        "like": " like",
        "wise": " wise",
        "ward": " ward",
        "ship": " ship",
        "hood": " hood",
        "dom": " dom",
        "age": " age",
        "ery": " ery",
        "ary": " ary",
        "ory": " ory",
        "ive": " ive",
        "ous": " ous",
        "ious": " ious",
        "eous": " eous",
    }

    text_lower = text.lower()
    for suffix, expansion in suffix_expansions.items():
        if text_lower.endswith(suffix) and len(text) > len(suffix):
            # Verificar que no sea parte de una palabra más larga
            prefix_end = len(text) - len(suffix)
            if prefix_end > 0:
                prev_char = text[prefix_end - 1]
                if prev_char.islower():
                    # Es realmente un sufijo
                    prefix = text[:prefix_end]
                    return prefix + expansion

    return text


def create_alternative_entity_representations(entity_name: str) -> List[str]:
    """
    Crea representaciones alternativas de una entidad para mejorar matching.

    Args:
        entity_name: Nombre original de la entidad

    Returns:
        Lista de representaciones alternativas
    """
    if not entity_name:
        return [entity_name]

    alternatives = [entity_name]  # Incluir el original

    # 1. Versión normalizada
    normalized = normalize_entity_name_for_embedding(entity_name)
    if normalized != entity_name.lower():
        alternatives.append(normalized)

    # 2. Versión con guiones
    with_hyphens = re.sub(r"([a-z])([A-Z])", r"\1-\2", entity_name).lower()
    if with_hyphens not in alternatives:
        alternatives.append(with_hyphens)

    # 3. Versión con guiones bajos
    with_underscores = re.sub(r"([a-z])([A-Z])", r"\1_\2", entity_name).lower()
    if with_underscores not in alternatives:
        alternatives.append(with_underscores)

    # 4. Acrónimos si el nombre es muy largo
    if len(entity_name) > 15:
        acronym = _create_acronym(entity_name)
        if acronym and len(acronym) >= 2:
            alternatives.append(acronym)

    return list(set(alternatives))  # Eliminar duplicados


def _create_acronym(text: str) -> str:
    """
    Crea un acrónimo a partir de un texto.
    Ej: "VegetarianPizza" → "VP"
    """
    # Separar en palabras
    words = re.findall(r"[A-Z][a-z]*", text)
    if len(words) < 2:
        return ""

    # Tomar primera letra de cada palabra
    acronym = "".join(word[0].upper() for word in words)
    return acronym.lower()


# ============================================================================
# SISTEMA DINÁMICO DE DETECCIÓN DE PROPIEDADES VÍA SPARQL
# ============================================================================


async def discover_properties_via_sparql(sparql_endpoint_config=None):
    """
    Descubre dinámicamente las propiedades de anotación, datos y objetos
    usando consultas SPARQL contra Virtuoso.

    Args:
        sparql_endpoint_config: Configuración del endpoint SPARQL

    Returns:
        Dict con propiedades clasificadas por tipo y frecuencia de uso
    """
    try:
        # Importar la función de consulta SPARQL
        import sys
        import os

        sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
        from sparql import queryVirtuoso

        print("🔍 Descubriendo propiedades dinámicamente vía SPARQL...")

        discovered_properties = {
            "annotation_properties": {},
            "data_properties": {},
            "object_properties": {},
            "literal_properties": {},
            "usage_stats": {},
        }

        # 1. Detectar Annotation Properties
        annotation_query = """
        SELECT DISTINCT ?annotationProperty ?label
        WHERE {
            ?annotationProperty a owl:AnnotationProperty .
            OPTIONAL { ?annotationProperty rdfs:label ?label }
        }
        ORDER BY ?annotationProperty
        """

        try:
            result = await queryVirtuoso(annotation_query)
            for binding in result.get("results", {}).get("bindings", []):
                prop_uri = binding["annotationProperty"]["value"]
                label = binding.get("label", {}).get("value", "")
                discovered_properties["annotation_properties"][prop_uri] = {
                    "label": label,
                    "type": "annotation",
                }
        except Exception as e:
            print(f"⚠️ Error consultando annotation properties: {e}")

        # 2. Detectar Data Properties
        data_query = """
        SELECT DISTINCT ?dataProperty ?label ?range
        WHERE {
            ?dataProperty a owl:DatatypeProperty .
            OPTIONAL { ?dataProperty rdfs:label ?label }
            OPTIONAL { ?dataProperty rdfs:range ?range }
        }
        ORDER BY ?dataProperty
        """

        try:
            result = await queryVirtuoso(data_query)
            for binding in result.get("results", {}).get("bindings", []):
                prop_uri = binding["dataProperty"]["value"]
                label = binding.get("label", {}).get("value", "")
                range_uri = binding.get("range", {}).get("value", "")
                discovered_properties["data_properties"][prop_uri] = {
                    "label": label,
                    "range": range_uri,
                    "type": "datatype",
                }
        except Exception as e:
            print(f"⚠️ Error consultando data properties: {e}")

        # 3. Detectar Object Properties
        object_query = """
        SELECT DISTINCT ?objectProperty ?label ?domain ?range
        WHERE {
            ?objectProperty a owl:ObjectProperty .
            OPTIONAL { ?objectProperty rdfs:label ?label }
            OPTIONAL { ?objectProperty rdfs:domain ?domain }
            OPTIONAL { ?objectProperty rdfs:range ?range }
        }
        ORDER BY ?objectProperty
        """

        try:
            result = await queryVirtuoso(object_query)
            for binding in result.get("results", {}).get("bindings", []):
                prop_uri = binding["objectProperty"]["value"]
                label = binding.get("label", {}).get("value", "")
                domain_uri = binding.get("domain", {}).get("value", "")
                range_uri = binding.get("range", {}).get("value", "")
                discovered_properties["object_properties"][prop_uri] = {
                    "label": label,
                    "domain": domain_uri,
                    "range": range_uri,
                    "type": "object",
                }
        except Exception as e:
            print(f"⚠️ Error consultando object properties: {e}")

        # 4. Detectar propiedades con valores literales (uso real)
        literal_query = """
        SELECT DISTINCT ?property (COUNT(*) as ?usage_count) (SAMPLE(?literal) as ?example_literal)
        WHERE {
            ?subject ?property ?literal .
            FILTER(isLiteral(?literal))
            FILTER(STRLEN(STR(?literal)) > 3)
            FILTER(STRLEN(STR(?literal)) < 500)
        }
        GROUP BY ?property
        ORDER BY DESC(?usage_count)
        """

        try:
            result = await queryVirtuoso(literal_query)
            for binding in result.get("results", {}).get("bindings", []):
                prop_uri = binding["property"]["value"]
                usage_count = int(binding["usage_count"]["value"])
                example_literal = binding.get("example_literal", {}).get("value", "")
                discovered_properties["literal_properties"][prop_uri] = {
                    "usage_count": usage_count,
                    "example": example_literal,
                    "type": "literal",
                }
        except Exception as e:
            print(f"⚠️ Error consultando literal properties: {e}")

        # 5. Estadísticas generales de uso
        usage_query = """
        SELECT ?predicate (COUNT(*) as ?usage_count) (SAMPLE(?object) as ?example_value)
        WHERE {
            ?subject ?predicate ?object .
            FILTER(!isBlank(?subject))
            FILTER(STRSTARTS(STR(?predicate), "http"))
        }
        GROUP BY ?predicate
        ORDER BY DESC(?usage_count)
        LIMIT 50
        """

        try:
            result = await queryVirtuoso(usage_query)
            for binding in result.get("results", {}).get("bindings", []):
                prop_uri = binding["predicate"]["value"]
                usage_count = int(binding["usage_count"]["value"])
                example_value = binding.get("example_value", {}).get("value", "")
                discovered_properties["usage_stats"][prop_uri] = {
                    "usage_count": usage_count,
                    "example": example_value,
                }
        except Exception as e:
            print(f"⚠️ Error consultando usage stats: {e}")

        return discovered_properties

    except Exception as e:
        print(f"❌ Error en descubrimiento dinámico de propiedades: {e}")
        return None


def classify_properties_by_semantics(discovered_properties):
    """
    Clasifica las propiedades descubiertas en categorías semánticas
    basándose en el análisis de sus URIs y uso.

    Args:
        discovered_properties: Propiedades descubiertas vía SPARQL

    Returns:
        Dict con propiedades clasificadas semánticamente
    """
    semantic_classification = {
        "textual_descriptive": {  # Para anotaciones textuales descriptivas
            "properties": [],
            "weight": 0.8,
            "description": "Propiedades con texto descriptivo (títulos, descripciones, etc.)",
        },
        "categorical_metadata": {  # Para metadatos categóricos
            "properties": [],
            "weight": 0.6,
            "description": "Metadatos categóricos (keywords, language, license, etc.)",
        },
        "technical_identifiers": {  # Para identificadores técnicos
            "properties": [],
            "weight": 0.4,
            "description": "Identificadores técnicos (ids, versions, formats, etc.)",
        },
        "relational_links": {  # Para enlaces relacionales
            "properties": [],
            "weight": 0.3,
            "description": "Enlaces y relaciones entre entidades",
        },
    }

    # Patrones para clasificación automática
    patterns = {
        'textual_descriptive': [
            'title', 'description', 'comment', 'abstract', 'summary',
            'definition', 'note', 'explanation', 'label', 'name'
        ],
        'categorical_metadata': [
            'keyword', 'tag', 'category', 'type', 'language', 'license',
            'subject', 'theme', 'topic', 'format', 'mediatype'
        ],
        'technical_identifiers': [
            'id', 'identifier', 'version', 'endpoint', 'url', 'uri',
            'participant', 'originator', 'service'
        ],
        'relational_links': [
            'has', 'contains', 'includes', 'relates', 'links', 'connects',
            'creator', 'owner', 'representation', 'resource', 'offer'
        ]
    }

    # Combinar todas las propiedades descubiertas
    all_properties = {}
    all_properties.update(discovered_properties.get("annotation_properties", {}))
    all_properties.update(discovered_properties.get("data_properties", {}))
    all_properties.update(discovered_properties.get("object_properties", {}))  
    all_properties.update(discovered_properties.get("literal_properties", {}))

    # Clasificar cada propiedad
    for prop_uri, prop_info in all_properties.items():
        # Extraer nombre local de la URI
        if "#" in prop_uri:
            local_name = prop_uri.split("#")[-1].lower()
        elif "/" in prop_uri:
            local_name = prop_uri.split("/")[-1].lower()
        else:
            local_name = prop_uri.lower()

        # Obtener estadísticas de uso
        usage_count = (
            discovered_properties.get("usage_stats", {})
            .get(prop_uri, {})
            .get("usage_count", 0)
        )

        # Clasificar por patrones
        classified = False
        for category, keywords in patterns.items():
            if any(keyword in local_name for keyword in keywords):
                semantic_classification[category]["properties"].append(
                    {
                        "uri": prop_uri,
                        "local_name": local_name,
                        "usage_count": usage_count,
                        "info": prop_info,
                    }
                )
                classified = True
                break

        # Si no se clasificó, ponerlo en technical_identifiers por defecto
        if not classified and usage_count > 0:
            semantic_classification["technical_identifiers"]["properties"].append(
                {
                    "uri": prop_uri,
                    "local_name": local_name,
                    "usage_count": usage_count,
                    "info": prop_info,
                }
            )

    # Ordenar por frecuencia de uso dentro de cada categoría
    for category in semantic_classification:
        semantic_classification[category]["properties"].sort(
            key=lambda x: x["usage_count"], reverse=True
        )

    return semantic_classification


# ============================================================================
# CLASE PRINCIPAL PARA ENRIQUECIMIENTO DE ANOTACIONES
# ============================================================================


class AnnotationEnricher:
    """
    Clase para enriquecer embeddings y contexto usando anotaciones semánticas.
    Detecta automáticamente anotaciones en cualquier ontología y las clasifica.
    """

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        use_dynamic_discovery: bool = True,
        virtuoso_config: Optional[Dict] = None,
    ):
        """
        Inicializa el enriquecidor con un modelo de embeddings.

        Args:
            embedding_model_name: Nombre del modelo de embeddings a usar
            use_dynamic_discovery: Si usar descubrimiento dinámico vía SPARQL
        """
        self.embedding_model_name = embedding_model_name or get_active_embedding_model()
        self.embedding_model = None  # Se carga bajo demanda
        self.use_dynamic_discovery = use_dynamic_discovery

        # Detectar si usar estrategia adaptativa
        self.use_adaptive_strategy = (
            is_adaptive_strategy_enabled()
            and self.embedding_model_name == "adaptive_strategy"
        )

        if self.use_adaptive_strategy:
            print("🧠 Estrategia adaptativa de embeddings habilitada")
            self.adaptive_strategy = AdaptiveEmbeddingStrategy()
        else:
            self.adaptive_strategy = None

        # Almacenamiento de anotaciones extraídas
        self.entity_annotations = {}

        # Sistema dinámico de predicados
        self.discovered_properties = None
        self.semantic_classification = None

        # Sistema legacy de predicados (fallback)
        self.legacy_discovered_predicates = {
            "labels": set(),
            "descriptions": set(),
            "examples": set(),
            "hierarchical": set(),
            "semantic": set(),
            "unknown": set(),
        }

        # Alias para compatibilidad con kg_embedding.py
        self.discovered_predicates = self.legacy_discovered_predicates

        # Cache de embeddings para anotaciones
        self.annotation_embeddings_cache = {}

        # Inicializar sistema dinámico si está habilitado
        self.dynamic_detector = None
        ### CORRECCIÓN ###: Lógica de inicialización del sistema dinámico.
        # Ahora depende del parámetro `virtuoso_config` en lugar de una importación.
        if self.use_dynamic_discovery and virtuoso_config and VIRTUOSO_CLIENT_AVAILABLE:
            try:
                print("🚀 Intentando inicializar sistema dinámico con la configuración proporcionada...")
                virtuoso_client = VirtuosoClient(virtuoso_config)
                self.dynamic_detector = DynamicPropertyDetector(virtuoso_client)
                self.discovered_properties = (
                    self.dynamic_detector.discover_properties_from_virtuoso()
                )
                self.semantic_classification = (
                    self.dynamic_detector.get_annotation_properties_by_type()
                )
                if self.discovered_properties.get("annotation_properties"):
                    print("✅ Sistema de descubrimiento dinámico inicializado correctamente.")
                else:
                    print("⚠️ Sistema dinámico inicializado, pero no se encontraron propiedades de anotación.")
                    self.dynamic_detector = (
                        None  # Desactivar si no se encontró nada útil.
                    )

            except Exception as e:
                print(f"⚠️ No se pudo inicializar el sistema dinámico con Virtuoso: {e}")
                self.dynamic_detector = None  # Fallback a legacy si hay error
        else:
            if not virtuoso_config:
                print("📝 Usando sistema legacy de anotaciones (no se proporcionó config de Virtuoso).")
            if not VIRTUOSO_CLIENT_AVAILABLE:
                print("📝 Usando sistema legacy de anotaciones (cliente de Virtuoso no disponible).")

    def discover_properties_dynamically_from_virtuoso(
        self, virtuoso_client: "VirtuosoClient"
    ) -> Dict[str, Dict]:
        """
        Descubre propiedades automáticamente usando SPARQL en lugar de hardcoding.

        Args:
            virtuoso_client: Cliente de Virtuoso para ejecutar consultas

        Returns:
            Diccionario con propiedades clasificadas por tipo
        """
        if not VIRTUOSO_CLIENT_AVAILABLE:
            print("⚠️ Cliente de Virtuoso no disponible, usando sistema legacy")
            return {}

        print("🔍 Descubriendo propiedades dinámicamente desde Virtuoso...")

        # Crear detector dinámico
        detector = DynamicPropertyDetector(virtuoso_client)

        # Descubrir propiedades desde Virtuoso
        self.discovered_properties = detector.discover_properties_from_virtuoso()

        # Crear clasificación semántica
        self.semantic_classification = detector.get_annotation_properties_by_type()

        # Mostrar resumen
        summary = detector.get_summary()
        print(f"📊 Resumen del descubrimiento dinámico:")
        print(f"  • Total propiedades: {summary['total_properties_discovered']}")
        print(f"  • Annotation Properties: {summary['annotation_properties']}")
        print(f"  • Datatype Properties: {summary['datatype_properties']}")
        print(f"  • Object Properties: {summary['object_properties']}")
        print(f"  • Literal Properties: {summary['literal_properties']}")

        print(f"📝 Clasificación semántica:")
        for ann_type, count in summary["grouped_by_annotation_type"].items():
            if count > 0:
                print(f"  • {ann_type}: {count} propiedades")

        # ¡CRÍTICO! Asignar el detector dinámico para que esté disponible
        self.dynamic_detector = detector

        return self.discovered_properties
        """Carga el modelo de embeddings bajo demanda"""
        if self.embedding_model is None:
            if should_log("DEBUG"):
                print(f"🔄 Cargando modelo de embeddings: {self.embedding_model_name}")

            # Resolve adaptive_strategy to actual model name
            actual_model_name = self.embedding_model_name
            if self.embedding_model_name == "adaptive_strategy":
                try:
                    # Import resolve_embedding_model from kg_embedding
                    import sys
                    import os

                    sys.path.append(os.path.dirname(__file__))
                    from kg_embedding import resolve_embedding_model

                    actual_model_name = resolve_embedding_model(
                        "adaptive_strategy", "short"
                    )
                    if should_log("DEBUG"):
                        print(f"   🎯 Resuelto adaptive_strategy → {actual_model_name}")
                except Exception as e:
                    if should_log("DEBUG"):
                        print(f"   ⚠️ No se pudo resolver adaptive_strategy: {e}")
                    actual_model_name = "LaBSE"  # Fallback to default

            self.embedding_model = SentenceTransformer(actual_model_name)

            # Asegurar que el modelo esté en el dispositivo correcto
            if hasattr(self.embedding_model, "to"):
                self.embedding_model = self.embedding_model.to(CURRENT_DEVICE)
                if should_log("DEBUG"):
                    print(f"   📍 Modelo movido a {CURRENT_DEVICE}")

        return self.embedding_model

    async def discover_annotation_predicates_dynamic(self) -> Dict[str, Any]:
        """
        Descubre automáticamente predicados de anotación usando consultas SPARQL dinámicas.

        Returns:
            Diccionario con propiedades clasificadas semánticamente
        """
        if self.use_dynamic_discovery:
            try:
                print("🚀 Iniciando descubrimiento dinámico de propiedades vía SPARQL...")
                
                # Descubrir propiedades vía SPARQL
                self.discovered_properties = await discover_properties_via_sparql()

                if self.discovered_properties:
                    # Clasificar semánticamente
                    self.semantic_classification = classify_properties_by_semantics(
                        self.discovered_properties
                    )

                    # Mostrar resumen
                    self._print_dynamic_discovery_summary()

                    return self.semantic_classification
                else:
                    print("⚠️ Descubrimiento dinámico falló, usando sistema legacy")
                    return self._fallback_to_legacy_discovery()

            except Exception as e:
                print(f"❌ Error en descubrimiento dinámico: {e}")
                print("🔄 Fallback a sistema legacy")
                return self._fallback_to_legacy_discovery()
        else:
            print("📝 Usando sistema legacy de descubrimiento de predicados")
            return self._fallback_to_legacy_discovery()

    def _print_dynamic_discovery_summary(self):
        """Imprime un resumen del descubrimiento dinámico"""
        if not self.semantic_classification:
            return

        print("\n📊 RESUMEN DEL DESCUBRIMIENTO DINÁMICO:")
        print("=" * 60)

        total_props = 0
        for category, data in self.semantic_classification.items():
            count = len(data["properties"])
            total_props += count
            if count > 0:
                print(f"\n🎯 {category.upper().replace('_', ' ')} ({count} propiedades):")
                print(f"   Peso: {data['weight']} | {data['description']}")

                # Mostrar las 3 más usadas
                top_props = data["properties"][:3]
                for prop in top_props:
                    local_name = prop["local_name"]
                    usage = prop["usage_count"]
                    print(f"   • {local_name} (usado {usage} veces)")

                if len(data["properties"]) > 3:
                    print(f"   ... y {len(data['properties']) - 3} más")

        print(f"\n✅ Total: {total_props} propiedades clasificadas dinámicamente")

        # Mostrar estadísticas adicionales
        if self.discovered_properties:
            ann_props = len(self.discovered_properties.get("annotation_properties", {}))
            data_props = len(self.discovered_properties.get("data_properties", {}))
            obj_props = len(self.discovered_properties.get("object_properties", {}))

            print(f"\n📈 Desglose por tipo OWL:")
            print(f"   • Annotation Properties: {ann_props}")
            print(f"   • Data Properties: {data_props}")
            print(f"   • Object Properties: {obj_props}")

    def _fallback_to_legacy_discovery(self) -> Dict[str, Any]:
        """Sistema legacy como fallback"""
        # Mapear el sistema legacy al nuevo formato
        legacy_classification = {
            "textual_descriptive": {
                "properties": [],
                "weight": 0.8,
                "description": "Propiedades textuales (legacy)",
            },
            "categorical_metadata": {
                "properties": [],
                "weight": 0.6,
                "description": "Metadatos categóricos (legacy)",
            },
            "technical_identifiers": {
                "properties": [],
                "weight": 0.4,
                "description": "Identificadores técnicos (legacy)",
            },
            "relational_links": {
                "properties": [],
                "weight": 0.3,
                "description": "Enlaces relacionales (legacy)",
            },
        }

        # Poblar con predicados hardcodeados como fallback
        for category, predicates in ANNOTATION_PREDICATES.items():
            if category == "labels" or category == "descriptions":
                target_category = "textual_descriptive"
            elif category == "examples":
                target_category = "categorical_metadata"
            else:
                target_category = "technical_identifiers"

            for pred_uri in predicates:
                local_name = (
                    pred_uri.split("#")[-1]
                    if "#" in pred_uri
                    else pred_uri.split("/")[-1]
                )
                legacy_classification[target_category]["properties"].append(
                    {
                        "uri": pred_uri,
                        "local_name": local_name.lower(),
                        "usage_count": 0,
                        "info": {"type": "legacy"},
                    }
                )

        print("📝 Sistema legacy activado con predicados predefinidos")
        return legacy_classification
        """
        Descubre automáticamente predicados de anotación en los triples.
        
        Args:
            triples: Lista de triples (s, p, o)
            
        Returns:
            Diccionario con predicados clasificados por tipo
        """
        print("🔍 Descubriendo predicados de anotación...")

        # Reiniciar discovered_predicates
        for key in self.discovered_predicates:
            self.discovered_predicates[key] = set()

        # Contar frecuencia de predicados con valores literales
        predicate_counts = defaultdict(int)
        literal_predicates = set()

        for s, p, o in triples:
            pred_str = str(p)

            # Identificar predicados que tienen valores literales
            if isinstance(o, str) and not o.startswith("http"):
                literal_predicates.add(pred_str)
                predicate_counts[pred_str] += 1

        # Clasificar predicados automáticamente
        for predicate in literal_predicates:
            classified = False
            pred_lower = predicate.lower()

            # Clasificación por patrones en el nombre
            for category, known_preds in ANNOTATION_PREDICATES.items():
                if predicate in known_preds:
                    self.discovered_predicates[category].add(predicate)
                    classified = True
                    break

            if not classified:
                # Clasificación heurística por nombres
                if any(
                    term in pred_lower for term in ["label", "name", "title", "titulo"]
                ):
                    self.discovered_predicates["labels"].add(predicate)
                elif any(
                    term in pred_lower
                    for term in [
                        "description",
                        "comment",
                        "note",
                        "definition",
                        "abstract",
                    ]
                ):
                    self.discovered_predicates["descriptions"].add(predicate)
                elif any(term in pred_lower for term in ["example", "sample", "demo"]):
                    self.discovered_predicates["examples"].add(predicate)
                elif any(
                    term in pred_lower
                    for term in ["broader", "narrower", "subclass", "parent", "child"]
                ):
                    self.discovered_predicates["hierarchical"].add(predicate)
                elif any(
                    term in pred_lower
                    for term in ["related", "similar", "match", "equiv"]
                ):
                    self.discovered_predicates["semantic"].add(predicate)
                else:
                    self.discovered_predicates["unknown"].add(predicate)

        # Imprimir resultados del descubrimiento
        total_discovered = sum(
            len(preds) for preds in self.discovered_predicates.values()
        )
        print(f"📊 Descubiertos {total_discovered} predicados de anotación:")
        for category, predicates in self.discovered_predicates.items():
            if predicates:
                print(f"  • {category.capitalize()}: {len(predicates)} predicados")
                for pred in list(predicates)[:3]:  # Mostrar primeros 3
                    short_name = pred.split("/")[-1] if "/" in pred else pred
                    print(f"    - {short_name}")
                if len(predicates) > 3:
                    print(f"    ... y {len(predicates) - 3} más")

        return self.discovered_predicates

    async def extract_rich_annotations_dynamic(
        self, triples: List[Tuple]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Extrae anotaciones usando el sistema dinámico de descubrimiento de propiedades.

        Args:
            triples: Lista de triples (s, p, o)

        Returns:
            Diccionario con anotaciones por entidad
        """
        print("📝 Extrayendo anotaciones con sistema dinámico...")

        # Descubrir propiedades si no se ha hecho
        if not self.semantic_classification:
            await self.discover_annotation_predicates_dynamic()

        entity_annotations = {}
        extraction_stats = {
            "entities_processed": 0,
            "annotations_extracted": 0,
            "by_category": {},
        }

        # Crear mapeo de URIs a categorías para búsqueda rápida
        property_to_category = {}
        if self.semantic_classification:
            for category, properties_list in self.semantic_classification.items():
                extraction_stats["by_category"][category] = 0
                if isinstance(properties_list, list):
                    # Asignar peso según categoría
                    weight = self._get_category_weight(category)
                    for prop_uri in properties_list:
                        property_to_category[prop_uri] = {
                            "category": category,
                            "weight": weight,
                        }
                else:
                    print(f"⚠️ DEBUG: Unexpected data format for category '{category}': {properties_list}")
        else:
            print("⚠️ DEBUG: No semantic_classification available")

        # Procesar triples
        for s, p, o in triples:
            entity = str(s)
            predicate = str(p)
            value = str(o)

            # Inicializar estructura para la entidad
            if entity not in entity_annotations:
                entity_annotations[entity] = {}
                # Inicializar con todas las categorías dinámicas
                for category in self.semantic_classification.keys():
                    entity_annotations[entity][category] = []
                # Categorías adicionales fijas
                entity_annotations[entity]["raw_text"] = []
                entity_annotations[entity]["weighted_content"] = []
                extraction_stats["entities_processed"] += 1

            # Clasificar y extraer anotación
            if predicate in property_to_category:
                category_info = property_to_category[predicate]
                category = category_info["category"]
                weight = category_info["weight"]

                # Limpiar y procesar el valor
                cleaned_value = self._clean_annotation_value(value)
                if cleaned_value:
                    entity_annotations[entity][category].append(cleaned_value)
                    entity_annotations[entity]["raw_text"].append(cleaned_value)
                    entity_annotations[entity]["weighted_content"].append(
                        {
                            "text": cleaned_value,
                            "weight": weight,
                            "category": category,
                            "predicate": predicate,
                        }
                    )

                    extraction_stats["annotations_extracted"] += 1
                    extraction_stats["by_category"][category] += 1

            # Si es texto literal pero no clasificado, añadir con peso bajo
            elif not value.startswith("http") and len(value) > 3:
                cleaned_value = self._clean_annotation_value(value)
                if cleaned_value:
                    entity_annotations[entity]["raw_text"].append(cleaned_value)
                    entity_annotations[entity]["weighted_content"].append(
                        {
                            "text": cleaned_value,
                            "weight": 0.2,
                            "category": "unclassified",
                            "predicate": predicate,
                        }
                    )

        self.entity_annotations = entity_annotations

        # Mostrar estadísticas
        self._print_extraction_stats(extraction_stats)

        return entity_annotations

    def _get_category_weight(self, category: str) -> float:
        """
        Obtiene el peso para una categoría de anotación.

        Args:
            category: Nombre de la categoría

        Returns:
            Peso numérico para la categoría
        """
        weights = {
            "labels": 1.0,
            "descriptions": 0.9,
            "keywords": 0.8,
            "examples": 0.7,
            "version_info": 0.6,
            "format_info": 0.5,
            "provenance_info": 0.4,
            "language_info": 0.3,
            "rights_info": 0.3,
            "temporal_info": 0.3,
            "semantic_relations": 0.5,
            "other_annotation": 0.2,
        }
        return weights.get(category, 0.2)

    def _print_extraction_stats(self, stats):
        """Imprime estadísticas de extracción"""
        print(f"\n📊 ESTADÍSTICAS DE EXTRACCIÓN:")
        print("-" * 40)
        print(f"✅ Entidades procesadas: {stats['entities_processed']}")
        print(f"✅ Anotaciones extraídas: {stats['annotations_extracted']}")

        if stats["by_category"]:
            print(f"\n📂 Por categoría:")
            for category, count in stats["by_category"].items():
                if count > 0:
                    category_name = category.replace("_", " ").title()
                    print(f"   • {category_name}: {count}")

    def calculate_annotation_bonus_dynamic(
        self, query: str, entity: str, max_bonus: float = 0.4
    ) -> float:
        """
        Calcula bonus de relevancia usando el sistema dinámico de clasificación de propiedades.

        Args:
            query: Consulta de búsqueda
            entity: Entidad a evaluar
            max_bonus: Bonus máximo a otorgar

        Returns:
            Valor de bonus entre 0 y max_bonus
        """
        if entity not in self.entity_annotations:
            return 0.0

        query_lower = query.lower()
        query_terms = set(query_lower.split())
        bonus = 0.0

        ann = self.entity_annotations[entity]

        # Usar contenido con pesos para cálculo más preciso
        if "weighted_content" in ann:
            for content_item in ann["weighted_content"]:
                text = content_item["text"]
                weight = content_item["weight"]
                category = content_item["category"]

                # Extraer términos del texto
                text_terms = set(self._extract_key_terms(text.lower()))
                matches = query_terms.intersection(text_terms)

                if matches:
                    # Bonus basado en número de coincidencias, peso de categoría y longitud de coincidencias
                    match_score = len(matches) / max(len(query_terms), 1)
                    category_bonus = (
                        match_score * weight * 0.15
                    )  # Factor de escala ajustado
                    bonus += category_bonus

        # Fallback al sistema original si no hay weighted_content
        else:
            # Bonus por coincidencias en contenido textual descriptivo (peso alto)
            for text in ann.get("textual_descriptive", []):
                text_terms = set(self._extract_key_terms(text.lower()))
                matches = query_terms.intersection(text_terms)
                if matches:
                    bonus += len(matches) * 0.1

            # Bonus por coincidencias en metadatos categóricos (peso medio)
            for text in ann.get("categorical_metadata", []):
                text_terms = set(text.lower().split())
                matches = query_terms.intersection(text_terms)
                if matches:
                    bonus += len(matches) * 0.06

            # Bonus por coincidencias en identificadores técnicos (peso bajo)
            for text in ann.get("technical_identifiers", []):
                text_terms = set(text.lower().split())
                matches = query_terms.intersection(text_terms)
                if matches:
                    bonus += len(matches) * 0.03

        return min(bonus, max_bonus)
        """
        Extrae todas las anotaciones semánticas disponibles en los triples.
        
        Args:
            triples: Lista de triples (s, p, o)
            
        Returns:
            Diccionario con anotaciones por entidad
        """
        print("📝 Extrayendo anotaciones enriquecidas...")

        # Primero descubrir predicados si no se ha hecho
        if not any(self.discovered_predicates.values()):
            self.discover_annotation_predicates(triples)

        # Combinar predicados conocidos y descubiertos
        all_predicates = {}
        for category in self.discovered_predicates:
            all_predicates[category] = (
                ANNOTATION_PREDICATES.get(category, set())
                | self.discovered_predicates[category]
            )

        entity_annotations = {}

        for s, p, o in triples:
            entity = str(s)
            predicate = str(p)
            value = str(o)

            # Inicializar estructura para la entidad
            if entity not in entity_annotations:
                entity_annotations[entity] = {
                    "labels": [],
                    "descriptions": [],
                    "examples": [],
                    "hierarchical": [],
                    "semantic": [],
                    "raw_text": [],  # Todo el texto para embeddings
                }

            # Clasificar anotaciones por tipo
            classified = False
            for annotation_type, predicates in all_predicates.items():
                if annotation_type == "unknown":
                    continue

                if predicate in predicates:
                    # Limpiar y procesar el valor
                    cleaned_value = self._clean_annotation_value(value)
                    if cleaned_value:
                        entity_annotations[entity][annotation_type].append(
                            cleaned_value
                        )
                        entity_annotations[entity]["raw_text"].append(cleaned_value)
                        classified = True
                        break

            # Si no se clasificó pero es texto literal, añadir a raw_text
            if not classified and not value.startswith("http") and len(value) > 3:
                cleaned_value = self._clean_annotation_value(value)
                if cleaned_value:
                    entity_annotations[entity]["raw_text"].append(cleaned_value)

        self.entity_annotations = entity_annotations
        print(f"✅ Anotaciones extraídas para {len(entity_annotations)} entidades")

        return entity_annotations

    def create_enriched_entity_embeddings(
        self,
        triples: List[Tuple],
        basic_entity_embeddings: Optional[Dict] = None,
        use_adaptive_strategy: bool = True,
    ) -> Dict[str, Any]:
        """
        Crea embeddings enriquecidos para entidades usando anotaciones y maneja la impresión del resumen.
        """
        print("🔍 Creando embeddings enriquecidos con anotaciones...")

        # 1. Extraer las anotaciones usando el mejor método disponible (dinámico o legacy)
        self._extract_annotations_auto(triples)

        # 2. Obtener solo las entidades URI para generar embeddings
        entities = set(s for s, p, o in triples if str(s).startswith("http"))
        entities.update(o for s, p, o in triples if str(o).startswith("http"))
        print(f"📊 Entidades URI detectadas para embedding: {len(entities)}")

        # 3. Crear contextos enriquecidos para cada entidad
        enriched_contexts = {
            entity: self.create_enriched_context(entity) for entity in entities
        }

        # 4. Generar los embeddings
        print(f"📊 Generando embeddings para {len(enriched_contexts)} entidades...")
        model = self._get_embedding_model()
        entities_list = list(enriched_contexts.keys())
        contexts_list = [enriched_contexts[e] for e in entities_list]

        all_embeddings = model.encode(
            contexts_list,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=CURRENT_DEVICE,
        )

        enriched_embeddings = {
            entity: emb for entity, emb in zip(entities_list, all_embeddings)
        }
        print(f"✅ Embeddings enriquecidos creados para {len(enriched_embeddings)} entidades URI.")

        # 5. Imprimir el resumen AHORA, que ya tenemos las anotaciones procesadas
        summary = self.get_annotation_summary()
        print("📊 Resumen del sistema enriquecido:")
        print(f"  • Tipo de sistema: {summary.get('system_type', 'desconocido').upper()}")
        print(f"  • Entidades con anotaciones: {summary.get('total_entities', 0)}")

        if summary.get("annotation_stats"):
            for ann_type, stats in summary["annotation_stats"].items():
                if stats.get("total_annotations", 0) > 0:
                    print(
                        f"  • {ann_type.capitalize()}: {stats.get('entities_with_annotations', 0)} entidades, "
                        f"{stats.get('total_annotations', 0)} anotaciones"
                    )

        return enriched_embeddings

    def _create_adaptive_embeddings_dynamic(
        self, triples: List[Tuple], basic_entity_embeddings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Crea embeddings usando estrategia adaptativa con anotaciones dinámicas"""
        print("🧠 Usando estrategia adaptativa con anotaciones dinámicas...")

        # Obtener todas las entidades únicas
        entities = set()
        for s, p, o in triples:
            if isinstance(s, str) and (s.startswith("http") or "#" in s):
                entities.add(s)
            if isinstance(o, str) and (o.startswith("http") or "#" in o):
                entities.add(o)

        # Preparar contextos enriquecidos usando anotaciones dinámicas
        enriched_contexts = {}
        for entity in entities:
            enriched_contexts[entity] = self.create_enriched_context_dynamic(entity)

        print(f"📊 Generando embeddings adaptativos para {len(enriched_contexts)} entidades...")

        # Usar estrategia adaptativa
        strategy = AdaptiveEmbeddingStrategy()
        contexts_list = list(enriched_contexts.values())
        embeddings_dict, strategy_stats = strategy.create_adaptive_embeddings(
            contexts_list, show_analysis=True
        )

        # Mapear de vuelta a entidades
        entities_list = list(enriched_contexts.keys())
        final_embeddings = {}
        for i, entity in enumerate(entities_list):
            if i < len(embeddings_dict):
                final_embeddings[entity] = embeddings_dict[i]

        print(f"✅ Embeddings adaptativos dinámicos creados para {len(final_embeddings)} entidades")
        print(f"📈 Modelos utilizados: {list(strategy_stats['model_usage'].keys())}")

        return final_embeddings

    def _create_standard_embeddings_dynamic(
        self, triples: List[Tuple], basic_entity_embeddings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Crea embeddings usando un modelo único con anotaciones dinámicas
        """
        # Obtener entidades URI únicamente
        entities = set()
        for s, p, o in triples:
            if isinstance(s, str) and (s.startswith("http") or "#" in s):
                entities.add(s)
            if isinstance(o, str) and (o.startswith("http") or "#" in o):
                entities.add(o)

        print(f"📊 Entidades URI detectadas: {len(entities)}")

        # Preparar contextos enriquecidos usando anotaciones dinámicas
        enriched_contexts = {}
        for entity in entities:
            enriched_contexts[entity] = self.create_enriched_context_dynamic(entity)

        print(f"📊 Generando embeddings enriquecidos para {len(enriched_contexts)} entidades URI...")

        # Generar embeddings en lotes
        model = self._get_embedding_model()
        entities_list = list(enriched_contexts.keys())
        contexts_list = [enriched_contexts[entity] for entity in entities_list]

        # Procesar en lotes para eficiencia de memoria
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(contexts_list), batch_size):
            batch_contexts = contexts_list[i : i + batch_size]
            batch_embeddings = model.encode(
                batch_contexts, convert_to_tensor=True, device=CURRENT_DEVICE
            )

            # Convertir a lista y asegurar dispositivo
            batch_embeddings = ensure_tensor_device(batch_embeddings, CURRENT_DEVICE)
            all_embeddings.extend(batch_embeddings)

        # Crear diccionario final
        enriched_embeddings = dict(zip(entities_list, all_embeddings))

        # Combinar con embeddings básicos si se proporcionan
        if basic_entity_embeddings:
            for entity, embedding in basic_entity_embeddings.items():
                if entity not in enriched_embeddings:
                    enriched_embeddings[entity] = ensure_tensor_device(
                        embedding, CURRENT_DEVICE
                    )

        print(f"✅ Embeddings dinámicos creados para {len(enriched_embeddings)} entidades URI")
        return enriched_embeddings

    def create_enriched_context_dynamic(
        self, entity: str, max_length: int = 800
    ) -> str:
        """
        Crea contexto enriquecido para una entidad usando anotaciones dinámicas.

        Args:
            entity: URI o ID de la entidad
            max_length: Longitud máxima del contexto

        Returns:
            Contexto enriquecido como string
        """
        if entity not in self.entity_annotations:
            return self._create_fallback_context(entity)

        ann = self.entity_annotations[entity]
        context_parts = []

        # Nombre base de la entidad (original y normalizado)
        entity_name = self._extract_entity_name(entity)
        context_parts.append(entity_name)

        # Expansión semántica automática para entidades con pocos datos
        semantic_expansion = self._create_semantic_expansion(entity_name)
        context_parts.extend(semantic_expansion)

        # Añadir versión normalizada para mejorar matching
        normalized_name = normalize_entity_name_for_embedding(entity_name)
        if normalized_name != entity_name.lower():
            context_parts.append(normalized_name)

        # Añadir variantes del nombre para mejor cobertura
        variants = create_alternative_entity_representations(entity_name)
        for variant in variants[:2]:
            if variant not in context_parts:
                context_parts.append(variant)

        # Usar anotaciones dinámicas por orden de prioridad
        priority_order = [
            "labels",
            "descriptions",
            "keywords",
            "examples",
            "version_info",
            "format_info",
            "provenance_info",
            "language_info",
            "rights_info",
            "temporal_info",
        ]

        for ann_type in priority_order:
            if ann_type in ann and ann[ann_type]:
                if ann_type == "labels":
                    # Etiquetas tienen prioridad máxima
                    context_parts.extend(ann[ann_type][:3])
                elif ann_type == "descriptions":
                    # Usar la mejor descripción
                    best_desc = self._select_best_description(ann[ann_type])
                    if best_desc:
                        context_parts.append(best_desc)
                elif ann_type == "keywords":
                    # Añadir palabras clave importantes
                    context_parts.extend(ann[ann_type][:5])
                else:
                    # Otros tipos de anotación
                    context_parts.extend(ann[ann_type][:2])

        # Unir todo respetando la longitud máxima
        full_context = " | ".join(context_parts)

        # Truncar si es necesario
        if len(full_context) > max_length:
            truncated = full_context[: max_length - 3] + "..."
            return truncated

        return full_context

    def _extract_annotations_auto(self, triples: List[Tuple]):
        """
        Extrae anotaciones automáticamente usando el mejor método disponible.
        Prioriza el sistema dinámico si está disponible, fallback a legacy.

        Args:
            triples: Lista de triples RDF
        """
        # Verificar si el sistema dinámico está disponible y funcionando
        if (
            hasattr(self, "dynamic_detector")
            and self.dynamic_detector is not None
            and hasattr(self.dynamic_detector, "discovered_properties")
            and self.dynamic_detector.discovered_properties
        ):

            print("📝 Extrayendo anotaciones con sistema dinámico...")
            try:
                # Usar la versión síncrona del sistema dinámico
                self.entity_annotations = self._extract_annotations_dynamic_sync(
                    triples
                )
                print("✅ Anotaciones dinámicas extraídas exitosamente")
                return
            except Exception as e:
                print(f"⚠️ Error con sistema dinámico: {e}")
                print("🔄 Fallback a sistema legacy...")

        # Fallback al sistema legacy
        print("📝 Extrayendo anotaciones con método legacy...")
        self.entity_annotations = self.extract_rich_annotations(triples)

    def _extract_annotations_dynamic_sync(
        self, triples: List[Tuple]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Versión síncrona y corregida del extractor dinámico de anotaciones.
        """
        entity_annotations = defaultdict(lambda: defaultdict(list))

        if not self.semantic_classification:
            print("⚠️ No hay clasificación semántica disponible para la extracción dinámica.")
            return {}

        # Crear un mapa de predicados a categorías para una búsqueda rápida y eficiente.
        prop_to_cat_map = {}
        for category, prop_uris in self.semantic_classification.items():
            for uri in prop_uris:
                prop_to_cat_map[uri] = category

        # Procesar los triples
        for s_uri, p_uri, o_val in triples:
            # Si el predicado es una de las propiedades de anotación que hemos descubierto...
            if p_uri in prop_to_cat_map:
                category = prop_to_cat_map[p_uri]

                # Limpiar el valor del objeto (literal)
                cleaned_value = self._clean_annotation_value(str(o_val))

                # Si el valor es significativo, lo almacenamos
                if cleaned_value:
                    entity_annotations[str(s_uri)][category].append(cleaned_value)
                    # También lo añadimos al texto crudo para los embeddings
                    entity_annotations[str(s_uri)]["raw_text"].append(cleaned_value)

        print(f"✅ Anotaciones dinámicas extraídas para {len(entity_annotations)} entidades.")
        return dict(entity_annotations)

    def _classify_annotation_semantically(self, predicate: str, value: str) -> str:
        """
        Clasifica una anotación según su tipo semántico usando las propiedades dinámicas.
        """
        if not hasattr(self.dynamic_detector, "semantic_classification"):
            return "other_annotation"

        # Buscar en qué categoría semántica está el predicado
        for category, props in self.dynamic_detector.semantic_classification.items():
            for prop_uri in props:
                if predicate == prop_uri:
                    return category

        return "other_annotation"

    def extract_rich_annotations(
        self, triples: List[Tuple]
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        Método legacy para extraer anotaciones usando predicados hardcodeados.

        Args:
            triples: Lista de triples RDF

        Returns:
            Diccionario con anotaciones por entidad
        """
        print("📝 Extrayendo anotaciones con método legacy...")

        entity_annotations = {}

        for s, p, o in triples:
            # Solo procesar entidades URI
            if not (isinstance(s, str) and (s.startswith("http") or "#" in s)):
                continue

            # Solo procesar valores literales
            if isinstance(o, str) and not o.startswith("http") and len(o.strip()) > 0:
                entity = s
                predicate = p
                value = o.strip()

                # Inicializar estructura para esta entidad
                if entity not in entity_annotations:
                    entity_annotations[entity] = {
                        "labels": [],
                        "descriptions": [],
                        "keywords": [],
                        "technical": [],
                        "endpoints": [],  # AÑADIDO
                        "hierarchical": [],  # AÑADIDO
                        "semantic": [],
                        "raw_text": [],
                    }

                # Clasificar por predicado
                classified = False
                for annotation_type, predicates in ANNOTATION_PREDICATES.items():
                    if predicate in predicates:
                        cleaned_value = self._clean_annotation_value(value)
                        if cleaned_value:
                            entity_annotations[entity][annotation_type].append(
                                cleaned_value
                            )
                            entity_annotations[entity]["raw_text"].append(cleaned_value)
                            classified = True
                            break

                # Si no se clasificó, añadir a raw_text
                if not classified and len(value) > 3:
                    cleaned_value = self._clean_annotation_value(value)
                    if cleaned_value:
                        entity_annotations[entity]["raw_text"].append(cleaned_value)

        self.entity_annotations = entity_annotations
        print(f"✅ Anotaciones legacy extraídas para {len(entity_annotations)} entidades")

        return entity_annotations

    def _get_embedding_model(self):
        """Obtiene el modelo de embeddings actual"""
        if hasattr(self, "embedding_model") and self.embedding_model is not None:
            return self.embedding_model
        else:
            # Fallback al modelo por defecto
            from sentence_transformers import SentenceTransformer

            model_name = get_active_embedding_model()
            self.embedding_model = SentenceTransformer(
                model_name, device="cpu"
            )  # USO FORZADO DE CPU
            return self.embedding_model

    def _clean_annotation_value(self, value: str) -> str:
        """
        Limpia y normaliza valores de anotaciones.

        Args:
            value: Valor de anotación sin procesar

        Returns:
            Valor limpio o cadena vacía si no es válido
        """
        if not value or len(value.strip()) < 2:
            return ""

        # Remover etiquetas de idioma
        if "@" in value:
            value = value.split("@")[0]

        # Remover comillas
        value = value.strip("\"'")

        # Limpiar espacios múltiples
        value = re.sub(r"\s+", " ", value.strip())

        # Filtrar valores muy cortos o que parecen URIs
        if len(value) < 3 or value.startswith("http"):
            return ""

        return value

    def create_enriched_context(self, entity: str, max_length: int = 800) -> str:
        """
        Crea contexto enriquecido para una entidad usando sus anotaciones.

        Args:
            entity: URI o ID de la entidad
            max_length: Longitud máxima del contexto

        Returns:
            Contexto enriquecido como string
        """
        if entity not in self.entity_annotations:
            return self._create_fallback_context(entity)

        ann = self.entity_annotations[entity]
        context_parts = []

        # Nombre base de la entidad (original y normalizado)
        entity_name = self._extract_entity_name(entity)
        context_parts.append(entity_name)

        # Expansión semántica automática para entidades con pocos datos
        semantic_expansion = self._create_semantic_expansion(entity_name)
        context_parts.extend(semantic_expansion)

        # Añadir versión normalizada para mejorar matching
        normalized_name = normalize_entity_name_for_embedding(entity_name)
        if normalized_name != entity_name.lower():
            context_parts.append(normalized_name)

        # Añadir variantes del nombre para mejor cobertura
        variants = create_alternative_entity_representations(entity_name)
        for variant in variants[:2]:  # Solo las 2 mejores variantes para no saturar
            if variant not in context_parts and variant.lower() not in [
                p.lower() for p in context_parts
            ]:
                context_parts.append(variant)

        # Etiquetas preferidas (mayor peso)
        if ann["labels"]:
            # Tomar las mejores etiquetas (más largas suelen ser más descriptivas)
            best_labels = sorted(ann["labels"], key=len, reverse=True)[:2]
            context_parts.extend(best_labels)

        # Descripción más informativa
        if ann["descriptions"]:
            # Encontrar la descripción más útil
            best_desc = self._select_best_description(ann["descriptions"])
            if best_desc:
                # Extraer términos clave de la descripción
                key_terms = self._extract_key_terms(best_desc, max_terms=10)
                context_parts.extend(key_terms)

        # Ejemplos si están disponibles
        if ann.get("examples"):
            context_parts.extend(ann["examples"][:2])

        # Unir todo respetando la longitud máxima
        full_context = " | ".join(context_parts)

        # Truncar si es necesario
        if len(full_context) > max_length:
            full_context = full_context[: max_length - 3] + "..."

        return full_context

    def _extract_entity_name(self, entity: str) -> str:
        """Extrae el nombre corto de una URI"""
        if "/" in entity:
            parts = entity.split("/")
            return parts[-1] if parts[-1] else parts[-2]
        elif "#" in entity:
            return entity.split("#")[-1]
        return entity

    def _select_best_description(self, descriptions: List[str]) -> str:
        """Selecciona la mejor descripción de una lista"""
        if not descriptions:
            return ""

        # Preferir descripciones de longitud media (ni muy cortas ni muy largas)
        scored_descs = []
        for desc in descriptions:
            # Puntuar por longitud (óptimo entre 50-300 caracteres)
            length_score = 1.0
            if 50 <= len(desc) <= 300:
                length_score = 2.0
            elif len(desc) < 20 or len(desc) > 500:
                length_score = 0.5

            # Puntuar por contenido informativo
            info_score = len(desc.split()) / max(len(desc), 1) * 10

            total_score = length_score * info_score
            scored_descs.append((total_score, desc))

        # Devolver la de mayor puntuación
        return max(scored_descs, key=lambda x: x[0])[1]

    def _extract_key_terms(self, text: str, max_terms: int = 8) -> List[str]:
        """
        Extrae términos clave de un texto.

        Args:
            text: Texto del que extraer términos
            max_terms: Número máximo de términos a extraer

        Returns:
            Lista de términos clave
        """
        # Limpiar y tokenizar
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = cleaned.split()

        # Detectar idioma basado en stopwords
        detected_lang = self._detect_language(words)
        stopwords = MULTILINGUAL_STOPWORDS.get(detected_lang, set())

        # Filtrar palabras
        key_terms = []
        for word in words:
            if (
                len(word) > 3
                and word not in stopwords
                and not word.isdigit()
                and not word.startswith("http")
            ):
                key_terms.append(word)

        # Remover duplicados manteniendo orden
        seen = set()
        unique_terms = []
        for term in key_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms[:max_terms]

    def _detect_language(self, words: List[str]) -> str:
        """Detecta el idioma basado en stopwords comunes"""
        lang_scores = {}

        for lang, stopwords in MULTILINGUAL_STOPWORDS.items():
            score = sum(
                1 for word in words[:20] if word in stopwords
            )  # Evaluar primeras 20 palabras
            lang_scores[lang] = score

        # Devolver idioma con mayor puntuación, inglés por defecto
        return max(lang_scores, key=lang_scores.get) if lang_scores else "en"

    def create_enriched_entity_embeddings(
        self,
        triples: List[Tuple],
        basic_entity_embeddings: Optional[Dict] = None,
        use_adaptive_strategy: bool = True,
    ) -> Dict[str, Any]:
        """
        Crea embeddings enriquecidos para entidades usando anotaciones.

        Args:
            triples: Lista de triples
            basic_entity_embeddings: Embeddings básicos existentes (opcional)
            use_adaptive_strategy: Si usar estrategia adaptativa para contenido largo

        Returns:
            Diccionario con embeddings enriquecidos
        """
        print("🔍 Creando embeddings enriquecidos con anotaciones...")

        # Extraer anotaciones si no se ha hecho
        if not self.entity_annotations:
            self._extract_annotations_auto(triples)

        # Decidir si usar estrategia adaptativa
        should_use_adaptive = (
            use_adaptive_strategy
            and self.use_adaptive_strategy
            and self.adaptive_strategy is not None
        )

        if should_use_adaptive:
            print("🧠 Usando estrategia adaptativa automática...")
            return self._create_adaptive_embeddings(triples, basic_entity_embeddings)
        else:
            return self._create_standard_embeddings(triples, basic_entity_embeddings)

    def _create_adaptive_embeddings(
        self, triples: List[Tuple], basic_entity_embeddings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Crea embeddings usando estrategia adaptativa"""
        print("🧠 Usando estrategia adaptativa de embeddings...")

        # Extraer anotaciones si no se han extraído
        if not self.entity_annotations:
            print("📝 Extrayendo anotaciones primero...")
            self._extract_annotations_auto(triples)

        # Obtener todas las entidades únicas
        entities = set()
        for s, p, o in triples:
            entities.add(str(s))
            if not str(o).startswith("http") and len(str(o)) < 200:
                continue
            entities.add(str(o))

        # Preparar contextos enriquecidos
        enriched_contexts = {}
        for entity in entities:
            enriched_context = self.create_enriched_context(entity)
            enriched_contexts[entity] = enriched_context

        print(f"📊 Generando embeddings adaptativos para {len(enriched_contexts)} entidades...")

        # Usar estrategia adaptativa
        strategy = AdaptiveEmbeddingStrategy()
        contexts_list = list(enriched_contexts.values())
        embeddings_dict, strategy_stats = strategy.create_adaptive_embeddings(
            contexts_list, show_analysis=True
        )

        # Mapear de vuelta a entidades
        entities_list = list(enriched_contexts.keys())
        final_embeddings = {}
        for i, entity in enumerate(entities_list):
            context = enriched_contexts[entity]
            if context in embeddings_dict:
                final_embeddings[entity] = embeddings_dict[context]

        print(f"✅ Embeddings adaptativos creados para {len(final_embeddings)} entidades")
        print(f"📈 Modelos utilizados: {list(strategy_stats['model_usage'].keys())}")

        return final_embeddings

    def _create_standard_embeddings(
        self, triples: List[Tuple], basic_entity_embeddings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Crea embeddings usando un modelo único (método original)
        CORRECCIÓN CRÍTICA: Solo incluir URIs de entidades, no literales
        """
        # NUEVA LÓGICA: Solo obtener entidades que sean URIs
        entities = set()
        literal_descriptions = {}  # Guardar descripciones para enriquecer contexto

        for s, p, o in triples:
            # Solo añadir sujetos que sean URIs
            if str(s).startswith("http"):
                entities.add(str(s))

            # Solo añadir objetos que sean URIs (no literales)
            if str(o).startswith("http"):
                entities.add(str(o))
            else:
                # Si el objeto es un literal y no muy largo, guardarlo para enriquecer el contexto del sujeto
                if len(str(o)) < 500 and str(s).startswith("http"):
                    if str(s) not in literal_descriptions:
                        literal_descriptions[str(s)] = []
                    literal_descriptions[str(s)].append(str(o))

        print(f"📊 Entidades URI detectadas: {len(entities)}")
        print(f"📊 Literales descriptivos recolectados para {len(literal_descriptions)} entidades")

        # Preparar contextos enriquecidos SOLO para entidades URI
        enriched_contexts = {}
        for entity in entities:
            # Crear contexto enriquecido usando el método existente
            enriched_context = self.create_enriched_context(entity)

            # Añadir literales descriptivos si los hay
            if entity in literal_descriptions:
                additional_context = " ".join(
                    literal_descriptions[entity][:3]
                )  # Máximo 3 literales
                if additional_context.strip():
                    enriched_context += f" {additional_context}"

            enriched_contexts[entity] = enriched_context

        print(f"📊 Generando embeddings enriquecidos para {len(enriched_contexts)} entidades URI...")

        # Generar embeddings en lotes
        model = self._get_embedding_model()
        entities_list = list(enriched_contexts.keys())
        contexts_list = [enriched_contexts[entity] for entity in entities_list]

        # Procesar en lotes para eficiencia de memoria
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(contexts_list), batch_size):
            batch_contexts = contexts_list[i : i + batch_size]
            batch_embeddings = model.encode(
                batch_contexts,
                show_progress_bar=True,
                convert_to_tensor=True,  # Mantener como tensores
                device=CURRENT_DEVICE,
            )  # Especificar dispositivo
            # Asegurar que los embeddings estén en el dispositivo correcto
            batch_embeddings = ensure_tensor_device(batch_embeddings, CURRENT_DEVICE)
            all_embeddings.append(batch_embeddings)

        # Concatenar todos los embeddings y asegurar dispositivo
        if all_embeddings:
            all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
            # Asegurar que el tensor concatenado esté en el dispositivo correcto
            all_embeddings_tensor = ensure_tensor_device(
                all_embeddings_tensor, CURRENT_DEVICE
            )
            # Convertir a lista de tensores individuales todos en el mismo dispositivo
            all_embeddings = [
                ensure_tensor_device(all_embeddings_tensor[i], CURRENT_DEVICE)
                for i in range(len(entities_list))
            ]
        else:
            all_embeddings = []

        # Crear diccionario final
        enriched_embeddings = dict(zip(entities_list, all_embeddings))

        # Combinar con embeddings básicos si se proporcionan
        if basic_entity_embeddings:
            # Promedio ponderado: 70% enriquecido, 30% básico
            final_embeddings = {}
            for entity in enriched_embeddings:
                if entity in basic_entity_embeddings:
                    enriched_emb = enriched_embeddings[entity]
                    basic_emb = basic_entity_embeddings[entity]

                    # Asegurar que ambos embeddings estén en el mismo dispositivo
                    enriched_emb = ensure_tensor_device(enriched_emb, CURRENT_DEVICE)
                    basic_emb = ensure_tensor_device(basic_emb, CURRENT_DEVICE)

                    # Normalizar antes de combinar usando PyTorch
                    enriched_emb = enriched_emb / torch.norm(enriched_emb)
                    basic_emb = basic_emb / torch.norm(basic_emb)

                    # Combinar con pesos
                    combined = 0.7 * enriched_emb + 0.3 * basic_emb
                    combined = combined / torch.norm(combined)

                    final_embeddings[entity] = combined
                else:
                    # Asegurar dispositivo para embeddings únicos también
                    final_embeddings[entity] = ensure_tensor_device(
                        enriched_embeddings[entity], CURRENT_DEVICE
                    )

            # Añadir entidades que solo están en básicos (solo si son URIs)
            for entity in basic_entity_embeddings:
                if entity not in final_embeddings and str(entity).startswith("http"):
                    basic_emb = ensure_tensor_device(
                        basic_entity_embeddings[entity], CURRENT_DEVICE
                    )
                    final_embeddings[entity] = basic_emb

            enriched_embeddings = final_embeddings

        print(f"✅ Embeddings enriquecidos creados para {len(enriched_embeddings)} entidades URI")
        print(f"📊 Muestra de entidades procesadas: {[e.split('/')[-1].split('#')[-1] for e in list(enriched_embeddings.keys())[:5]]}")
        return enriched_embeddings

    def enhance_query_with_annotations(
        self, query: str, top_expansions: int = 2
    ) -> str:
        """
        Enriquece una consulta usando sinónimos y términos relacionados de las anotaciones.
        Más inteligente sobre el contexto y evita términos irrelevantes.

        Args:
            query: Consulta original
            top_expansions: Número máximo de expansiones a añadir

        Returns:
            Consulta enriquecida
        """
        if not self.entity_annotations:
            return query

        query_lower = query.lower()
        query_terms = query_lower.split()

        # Detectar tipo de consulta usando patrones dinámicos
        generic_query_patterns = [
            "que",
            "tipos",
            "cuales",
            "what",
            "types",
            "kinds",
            "which",
            "show",
            "list",
        ]
        is_generic_type_query = any(
            term in query_lower for term in generic_query_patterns
        )

        # Detectar queries sobre información/datos usando términos dinámicos
        info_patterns = [
            "datos",
            "data",
            "información",
            "info",
            "elements",
            "items",
            "entities",
        ]
        is_data_query = any(term in query_lower for term in info_patterns)

        # No expandir para preguntas muy genéricas que ya son claras
        if is_generic_type_query and is_data_query and len(query_terms) <= 5:
            if should_log("DEBUG"):
                print(
                    f"🔍 Query genérica detectada - no se expande para mantener precisión"
                )
            return query

        expanded_terms = set()

        # Usar términos específicos del dominio detectados automáticamente
        domain_core_terms = set()
        try:
            # Obtener términos del dominio actual desde kg_embedding
            import sys
            import os

            sys.path.append(os.path.dirname(__file__))
            import kg_embedding

            # Usar DOMAIN_SPECIFIC_TERMS detectados automáticamente
            if (
                hasattr(kg_embedding, "DOMAIN_SPECIFIC_TERMS")
                and kg_embedding.DOMAIN_SPECIFIC_TERMS
            ):
                # Tomar los términos más frecuentes/relevantes
                domain_core_terms = set(
                    list(kg_embedding.DOMAIN_SPECIFIC_TERMS.keys())[:10]
                )

            # Fallback: usar términos generales si no hay específicos
            if not domain_core_terms:
                domain_core_terms = {
                    "type",
                    "class",
                    "entity",
                    "element",
                    "item",
                    "concept",
                }

        except Exception as e:
            # Fallback seguro: términos genéricos universales
            domain_core_terms = {
                "type",
                "class",
                "entity",
                "element",
                "item",
                "concept",
            }

        # Solo buscar expansiones muy específicas
        for entity, annotations in self.entity_annotations.items():
            entity_name = entity.split("/")[-1].split("#")[-1].lower()

            # Verificar si la entidad es conceptualmente relevante para la query
            entity_relevance = 0
            for query_term in query_terms:
                if len(query_term) > 2 and query_term in entity_name:
                    entity_relevance += 1

            # Solo procesar entidades que tengan relevancia directa
            if entity_relevance > 0:
                for label in annotations["labels"][:1]:  # Solo la primera etiqueta
                    label_terms = [
                        t.strip() for t in label.lower().split() if len(t) > 2
                    ]

                    # Solo añadir términos del dominio core
                    relevant_terms = [
                        t
                        for t in label_terms
                        if t in domain_core_terms and t not in query_terms
                    ]
                    expanded_terms.update(
                        relevant_terms[:1]
                    )  # Máximo 1 término por entidad

                    if len(expanded_terms) >= top_expansions:
                        break

            if len(expanded_terms) >= top_expansions:
                break

        # Construir query final solo si hay términos realmente relevantes
        if expanded_terms:
            final_terms = query_terms + list(expanded_terms)[:top_expansions]
            enhanced_query = " ".join(final_terms)
            if should_log("DEBUG"):
                print(f"🔍 Query expandida: '{query}' → '{enhanced_query}'")
            return enhanced_query
        else:
            if should_log("DEBUG"):
                print(f"🔍 No se encontraron expansiones relevantes - manteniendo query original")
            return query
            print(f"🔍 Query expandida: '{query}' → '{enhanced_query}'")

        return enhanced_query

    def calculate_annotation_bonus(
        self, query: str, entity: str, max_bonus: float = 0.3
    ) -> float:
        """
        Calcula bonus de relevancia basado en coincidencias en anotaciones.

        Args:
            query: Consulta de búsqueda
            entity: Entidad a evaluar
            max_bonus: Bonus máximo a otorgar

        Returns:
            Valor de bonus entre 0 y max_bonus
        """
        if entity not in self.entity_annotations:
            return 0.0

        query_lower = query.lower()
        query_terms = set(query_lower.split())
        bonus = 0.0

        ann = self.entity_annotations[entity]

        # Bonus por coincidencias exactas en etiquetas (peso alto)
        for label in ann["labels"]:
            label_terms = set(label.lower().split())
            matches = query_terms.intersection(label_terms)
            if matches:
                # Bonus proporcional al número de coincidencias
                bonus += len(matches) * 0.08

        # Bonus por coincidencias en descripciones (peso medio)
        for desc in ann["descriptions"]:
            desc_terms = set(self._extract_key_terms(desc))
            matches = query_terms.intersection(desc_terms)
            if matches:
                bonus += len(matches) * 0.04

        # Bonus por coincidencias en ejemplos (peso bajo)
        for example in ann["examples"]:
            example_terms = set(example.lower().split())
            matches = query_terms.intersection(example_terms)
            if matches:
                bonus += len(matches) * 0.02

        return min(bonus, max_bonus)

    def get_annotation_summary(self) -> Dict[str, Any]:
        """
        Devuelve un resumen de las anotaciones extraídas, funcionando tanto para el
        sistema dinámico como para el legacy.
        """
        if not self.entity_annotations:
            return {
                "total_entities": 0,
                "system_type": "dynamic" if self.dynamic_detector else "legacy",
                "annotation_stats": {},
            }

        summary = {
            "total_entities": len(self.entity_annotations),
            "system_type": "dynamic" if self.dynamic_detector else "legacy",
            "annotation_stats": defaultdict(
                lambda: {"total_annotations": 0, "entities_with_annotations": 0}
            ),
        }

        # Contar directamente desde las anotaciones extraídas
        all_annotation_types = set()
        for entity_data in self.entity_annotations.values():
            for ann_type in entity_data.keys():
                if ann_type != "raw_text":  # Excluir el campo de texto crudo
                    all_annotation_types.add(ann_type)

        # Inicializar contadores para todas las categorías encontradas
        for ann_type in all_annotation_types:
            summary["annotation_stats"][
                ann_type
            ]  # defaultdict se encarga de la inicialización

        # Llenar los contadores
        for entity_data in self.entity_annotations.values():
            for ann_type, annotations in entity_data.items():
                if ann_type != "raw_text" and annotations:
                    stats = summary["annotation_stats"][ann_type]
                    stats["total_annotations"] += len(annotations)
                    stats["entities_with_annotations"] += 1

        # Convertir defaultdict a dict normal para la salida
        summary["annotation_stats"] = dict(summary["annotation_stats"])

        # Para compatibilidad, si es legacy, podemos intentar llenar `discovered_predicates`
        if summary["system_type"] == "legacy" and hasattr(
            self, "discovered_predicates"
        ):
            summary["discovered_predicates"] = {
                k: len(v) for k, v in self.discovered_predicates.items()
            }
        else:
            summary["discovered_predicates"] = {}

        return summary

    def _create_semantic_expansion(self, entity_name: str) -> List[str]:
        """
        Crea expansiones semánticas adaptativas para una entidad usando herramientas dinámicas.
        Se adapta automáticamente a cualquier ontología usando términos detectados y anotaciones.

        Args:
            entity_name: Nombre de la entidad a expandir

        Returns:
            Lista de expansiones semánticas adaptadas al dominio actual
        """
        expansions = []
        name_lower = entity_name.lower()

        # 1. Expansiones usando términos específicos del dominio detectados automáticamente
        try:
            # Importar las variables globales de kg_embedding que contienen términos dinámicos
            import sys
            import os

            sys.path.append(os.path.dirname(__file__))
            import kg_embedding

            # Usar DOMAIN_SPECIFIC_TERMS que se detecta automáticamente de la ontología actual
            if (
                hasattr(kg_embedding, "DOMAIN_SPECIFIC_TERMS")
                and kg_embedding.DOMAIN_SPECIFIC_TERMS
            ):
                for term_key, term_value in kg_embedding.DOMAIN_SPECIFIC_TERMS.items():
                    if term_key in name_lower:
                        # Agregar el término expandido si es diferente
                        if (
                            term_value.lower() != term_key
                            and term_value not in expansions
                        ):
                            expansions.append(term_value)
        except Exception as e:
            pass  # Continuar si no se puede acceder a los términos dinámicos

        # 2. Expansiones usando aliases enriquecidos construidos desde anotaciones SKOS/RDF
        try:
            # Usar CLASS_ALIASES que se construye automáticamente desde las anotaciones
            if hasattr(kg_embedding, "CLASS_ALIASES") and kg_embedding.CLASS_ALIASES:
                entity_key = entity_name.lower()
                # Buscar en aliases normalizados
                for alias_key, mapped_entity in kg_embedding.CLASS_ALIASES.items():
                    if entity_key in alias_key or alias_key in entity_key:
                        # Agregar el alias como expansión si es diferente
                        if (
                            mapped_entity.lower() != entity_key
                            and mapped_entity not in expansions
                        ):
                            expansions.append(mapped_entity)
        except Exception as e:
            pass  # Continuar si no se puede acceder a los aliases

        # 3. Expansiones usando mapas dinámicos construidos desde las anotaciones de la ontología
        try:
            # Usar DYNAMIC_TERMS_MAP que se llena automáticamente desde las anotaciones procesadas
            if (
                hasattr(kg_embedding, "DYNAMIC_TERMS_MAP")
                and kg_embedding.DYNAMIC_TERMS_MAP
            ):
                for (
                    dynamic_term,
                    mapped_entity,
                ) in kg_embedding.DYNAMIC_TERMS_MAP.items():
                    if dynamic_term in name_lower:
                        if (
                            mapped_entity not in expansions
                            and mapped_entity.lower() != name_lower
                        ):
                            expansions.append(mapped_entity)
        except Exception as e:
            pass  # Continuar si no se puede acceder al mapa dinámico

        # 4. Expansiones usando anotaciones directas si están disponibles en este enriquecidor
        if hasattr(self, "entity_annotations") and self.entity_annotations:
            # Buscar la entidad en las anotaciones
            for entity_uri, annotations in self.entity_annotations.items():
                entity_uri_name = entity_uri.split("/")[-1].split("#")[-1]
                if entity_uri_name.lower() == name_lower:
                    # Usar etiquetas como expansiones
                    for label in annotations.get("labels", [])[
                        :2
                    ]:  # Máximo 2 etiquetas
                        if label and len(label) > 2 and label.lower() != name_lower:
                            # Extraer palabras clave de la etiqueta
                            label_words = label.split()
                            for word in label_words:
                                if (
                                    len(word) > 3
                                    and word.lower() not in name_lower
                                    and word not in expansions
                                ):
                                    expansions.append(word.lower())
                    break

        # 5. Expansiones estructurales (CamelCase, separadores) - siempre disponible

        # Dividir CamelCase en palabras separadas
        camel_words = re.findall(r"[A-Z][a-z]*|[a-z]+", entity_name)
        if len(camel_words) > 1:
            # Agregar palabras individuales significativas
            for word in camel_words:
                if (
                    len(word) > 3
                    and word.lower() not in name_lower
                    and word.lower() not in expansions
                ):
                    expansions.append(word.lower())

        # Dividir por separadores comunes
        if "_" in entity_name or "-" in entity_name or "." in entity_name:
            separator_words = re.split(r"[_\-.]", entity_name)
            for word in separator_words:
                if (
                    len(word) > 3
                    and word.lower() not in name_lower
                    and word.lower() not in expansions
                ):
                    expansions.append(word.lower())

        # 6. Expansiones morfológicas comunes (plurales/singulares, prefijos/sufijos)
        # Agregar variante singular/plural
        if name_lower.endswith("s") and len(name_lower) > 4:
            singular = name_lower[:-1]
            if singular not in expansions:
                expansions.append(singular)
        elif not name_lower.endswith("s") and len(name_lower) > 3:
            plural = name_lower + "s"
            if plural not in expansions:
                expansions.append(plural)

        # Detectar y expandir prefijos comunes en cualquier ontología
        common_prefixes = ["non", "anti", "multi", "sub", "super", "pre", "post"]
        for prefix in common_prefixes:
            if name_lower.startswith(prefix) and len(name_lower) > len(prefix) + 2:
                base_word = name_lower[len(prefix) :]
                if base_word not in expansions:
                    expansions.append(base_word)

        # Limitar expansiones para evitar ruido
        max_expansions = (
            6  # Permitir más expansiones ya que son más específicas del dominio
        )
        return expansions[:max_expansions]

    def _create_fallback_context(self, entity: str) -> str:
        """
        Crea un contexto de fallback para entidades sin anotaciones.

        Args:
            entity: URI o ID de la entidad

        Returns:
            Contexto básico basado en el nombre de la entidad
        """
        # Extraer nombre base de la entidad
        entity_name = self._extract_entity_name(entity)

        # Crear versión normalizada
        normalized_name = normalize_entity_name_for_embedding(entity_name)

        # Crear variantes del nombre
        variants = create_alternative_entity_representations(entity_name)

        # Construir contexto básico
        context_parts = [entity_name]

        # Añadir versión normalizada si es diferente
        if normalized_name != entity_name.lower():
            context_parts.append(normalized_name)

        # Añadir algunas variantes
        for variant in variants[:2]:
            if variant not in context_parts:
                context_parts.append(variant)

        # Añadir expansión semántica básica
        semantic_expansion = self._create_semantic_expansion(entity_name)
        context_parts.extend(semantic_expansion[:2])  # Solo las 2 mejores expansiones

        return " | ".join(context_parts)


# ============================================================================
# FUNCIONES GLOBALES DE UTILIDAD
# ============================================================================


async def create_enhanced_embedding_system_dynamic(
    triples: List[Tuple],
    existing_embeddings: Optional[Dict] = None,
    embedding_model_name: Optional[str] = None,
    use_dynamic_discovery: bool = True,
) -> Tuple[Dict, AnnotationEnricher]:
    """
    Crea un sistema de embeddings enriquecido con descubrimiento dinámico de propiedades vía SPARQL.

    Args:
        triples: Lista de triples de la ontología
        existing_embeddings: Embeddings existentes (opcional)
        embedding_model_name: Modelo de embeddings a usar
        use_dynamic_discovery: Si usar descubrimiento dinámico vía SPARQL

    Returns:
        Tupla con (embeddings_enriquecidos, enricher)
    """
    print("🚀 Inicializando sistema de embeddings enriquecido DINÁMICO...")

    # Crear enriquecidor con sistema dinámico
    enricher = AnnotationEnricher(embedding_model_name, use_dynamic_discovery)

    # Descubrir propiedades dinámicamente y extraer anotaciones
    if use_dynamic_discovery:
        try:
            await enricher.extract_rich_annotations_dynamic(triples)
        except Exception as e:
            print(f"⚠️ Error en sistema dinámico: {e}")
            print("🔄 Fallback a sistema legacy...")
            enricher.extract_rich_annotations(triples)
    else:
        enricher.extract_rich_annotations(triples)

    # Crear embeddings enriquecidos
    enriched_embeddings = enricher.create_enriched_entity_embeddings(
        triples, existing_embeddings
    )

    # Mostrar resumen del sistema
    if enricher.semantic_classification:
        summary = enricher.get_dynamic_annotation_summary()
        print("📊 Resumen del sistema enriquecido DINÁMICO:")
        print(f"  • Entidades procesadas: {summary['total_entities']}")
        print(f"  • Propiedades dinámicas descubiertas: {summary['total_dynamic_properties']}")
        print(f"  • Categorías semánticas: {summary['semantic_categories']}")
    else:
        summary = enricher.get_annotation_summary()
        print("📊 Resumen del sistema enriquecido LEGACY:")
        print(f"  • Entidades procesadas: {summary['total_entities']}")
        print(f"  • Predicados descubiertos: {sum(summary['discovered_predicates'].values())}")

    return enriched_embeddings, enricher


def get_dynamic_annotation_summary(self) -> Dict[str, Any]:
    """Devuelve un resumen del sistema dinámico de anotaciones"""
    if not self.semantic_classification:
        return self.get_annotation_summary()  # Fallback al legacy

    summary = {
        "total_entities": len(self.entity_annotations),
        "total_dynamic_properties": 0,
        "semantic_categories": len(self.semantic_classification),
        "category_breakdown": {},
        "system_type": "dynamic",
    }

    for category, data in self.semantic_classification.items():
        prop_count = len(data["properties"])
        summary["total_dynamic_properties"] += prop_count
        summary["category_breakdown"][category] = {
            "property_count": prop_count,
            "weight": data["weight"],
            "description": data["description"],
        }

    return summary


# Agregar el método a la clase AnnotationEnricher
AnnotationEnricher.get_dynamic_annotation_summary = get_dynamic_annotation_summary


def create_enhanced_embedding_system_dynamic(
    triples: List[Tuple],
    existing_embeddings: Optional[Dict] = None,
    embedding_model_name: Optional[str] = None,
    virtuoso_config: Optional[Dict] = None,
) -> Tuple[Dict, AnnotationEnricher]:
    """
    Crea un sistema de embeddings enriquecido usando detección dinámica de propiedades.

    Args:
        triples: Lista de triples de la ontología
        existing_embeddings: Embeddings existentes (opcional)
        embedding_model_name: Modelo de embeddings a usar
        virtuoso_config: Configuración para Virtuoso (opcional)

    Returns:
        Tupla con (embeddings_enriquecidos, enricher)
    """
    print("🚀 Inicializando sistema dinámico de embeddings enriquecido...")

    # Crear cliente de Virtuoso si está disponible
    virtuoso_client = None
    if VIRTUOSO_CLIENT_AVAILABLE and virtuoso_config:
        try:
            virtuoso_client = VirtuosoClient(virtuoso_config)
            print("✅ Cliente de Virtuoso inicializado")
        except Exception as e:
            print(f"⚠️ Error inicializando Virtuoso: {e}")

    # Crear enriquecidor con sistema dinámico
    enricher = AnnotationEnricher(embedding_model_name, use_dynamic_discovery=True)

    # Descubrir propiedades dinámicamente si Virtuoso está disponible
    if virtuoso_client:
        try:
            enricher.discover_properties_dynamically_from_virtuoso(virtuoso_client)
            print("✅ Propiedades descubiertas dinámicamente")
        except Exception as e:
            print(f"⚠️ Error en descubrimiento dinámico: {e}")
            print("🔄 Fallback a sistema legacy...")

    # Crear embeddings enriquecidos
    if enricher.discovered_properties:
        # Usar sistema dinámico
        print("📊 Usando sistema dinámico de anotaciones...")
        enriched_embeddings = enricher.create_enriched_entity_embeddings_dynamic(
            triples, existing_embeddings
        )
    else:
        # Fallback a sistema legacy
        print("📊 Usando sistema legacy de anotaciones...")
        enriched_embeddings = enricher.create_enriched_entity_embeddings(
            triples, existing_embeddings
        )

    # Mostrar resumen final
    if hasattr(enricher, "discovered_properties") and enricher.discovered_properties:
        total_props = sum(
            len(props)
            for props in enricher.discovered_properties.values()
            if isinstance(props, dict)
        )
        print(f"📈 Sistema dinámico activo con {total_props} propiedades")
    else:
        summary = enricher.get_annotation_summary()
        print(f"📈 Sistema legacy activo con {summary['total_entities']} entidades")

    return enriched_embeddings, enricher


def create_enhanced_embedding_system(
    triples: List[Tuple],
    existing_embeddings: Optional[Dict] = None,
    embedding_model_name: Optional[str] = None,
    virtuoso_config: Optional[Dict] = None,
) -> Tuple[Dict, AnnotationEnricher]:
    """
    Crea un sistema de embeddings enriquecido con anotaciones.

    Args:
        triples: Lista de triples de la ontología
        existing_embeddings: Embeddings existentes (opcional)
        embedding_model_name: Modelo de embeddings a usar

    Returns:
        Tupla con (embeddings_enriquecidos, enricher)
    """
    print("🚀 Inicializando sistema de embeddings enriquecido...")

    # Crear enriquecidor
    enricher = AnnotationEnricher(embedding_model_name, virtuoso_config=virtuoso_config)

    # Crear embeddings enriquecidos
    enriched_embeddings = enricher.create_enriched_entity_embeddings(
        triples, existing_embeddings
    )

    # Mostrar resumen
    # Reemplaza todo el bloque de impresión del resumen con esto:
    # Reemplaza el bloque de impresión del resumen con esto:
    summary = enricher.get_annotation_summary()
    print("📊 Resumen del sistema enriquecido:")
    print(f"  • Tipo de sistema: {summary.get('system_type', 'desconocido').upper()}")
    print(f"  • Entidades con anotaciones: {summary.get('total_entities', 0)}")

    # Imprimir estadísticas detalladas si existen
    if summary.get("annotation_stats"):
        for ann_type, stats in summary["annotation_stats"].items():
            if stats.get("total_annotations", 0) > 0:  # Solo mostrar si hay datos
                print(
                    f"  • {ann_type.capitalize()}: {stats.get('entities_with_annotations', 0)} entidades, "
                    f"{stats.get('total_annotations', 0)} anotaciones"
                )

    return enriched_embeddings, enricher


def enhanced_find_related_entities(
    query: str,
    enriched_embeddings: Dict,
    enricher: AnnotationEnricher,
    threshold: float = 0.3,
    top_n: int = 10,
    visible_nodes: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Búsqueda mejorada usando embeddings enriquecidos con sistema de boost dinámico.

    Combina:
    1. Similitud por embeddings (base)
    2. Bonus por anotaciones (sistema existente)
    3. Boost dinámico por coincidencias en nombres de entidades (nuevo)

    Args:
        query: Consulta en lenguaje natural
        enriched_embeddings: Embeddings enriquecidos
        enricher: Instancia del enriquecidor
        threshold: Umbral mínimo de similitud
        top_n: Número máximo de resultados

    Returns:
        Lista de (entidad, puntuación) ordenada por relevancia
    """
    if not enriched_embeddings:
        print("⚠️ No hay embeddings enriquecidos disponibles")
        return []

    # Inicializar logger de búsqueda
    logger = SearchLogger()
    logger.log_search_start(query, len(enriched_embeddings))

    # Log de configuración inicial
    if should_log("MINIMAL"):
        print(f"🎯 Threshold configurado: {threshold}")
        print(f"🎯 Top_n configurado: {top_n}")
        print(f"🎯 Visible nodes: {visible_nodes}")
        entity_sample = list(enriched_embeddings.keys())[:5]
        entity_names = [e.split("/")[-1].split("#")[-1] for e in entity_sample]
        print(f"🎯 Muestra de entidades disponibles: {entity_names}")

    # Crear mapeo multilingüe genérico basado en la ontología actual
    entity_names = [
        entity.split("/")[-1].split("#")[-1] for entity in enriched_embeddings.keys()
    ]
    ontology_mapping = create_generic_multilingual_mapping(entity_names)

    if should_log("DEBUG"):
        print(f"🌐 Mapeo multilingüe generado para {len(ontology_mapping)} términos")
        # Mostrar algunos ejemplos del mapeo
        sample_mappings = list(ontology_mapping.items())[:3]
        for term, candidates in sample_mappings:
            print(f"   {term} -> {candidates}")

    # Expandir consulta con anotaciones
    expanded_query = enricher.enhance_query_with_annotations(query)
    logger.log_query_expansion(query, expanded_query)

    # Analizar dimensiones de embeddings almacenados
    embedding_dimensions = {}
    for entity, embedding in enriched_embeddings.items():

        if isinstance(embedding, (list, tuple)):
            dim = len(embedding)
        elif isinstance(embedding, np.ndarray):
            dim = embedding.shape[-1]
        elif hasattr(embedding, "shape"):
            dim = embedding.shape[-1]
        else:
            dim = len(embedding)

        if dim not in embedding_dimensions:
            embedding_dimensions[dim] = []
        embedding_dimensions[dim].append(entity)

    if should_log("DEBUG"):
        print(
            f"🔍 Dimensiones detectadas en embeddings: {list(embedding_dimensions.keys())}"
        )

    # CORRECCIÓN CRÍTICA: Usar el mismo modelo que el enricher para mantener consistencia
    # Esto garantiza que el embedding de query y los embeddings de entidades
    # provengan del mismo espacio vectorial
    try:
        # Usar directamente el modelo del enricher para generar el embedding de query
        # Esto puede ser la estrategia adaptativa o un modelo único, pero será consistente
        if hasattr(enricher, "embedding_strategy") and enricher.embedding_strategy:
            # Si hay estrategia adaptativa, usarla para el query
            query_embedding = enricher.embedding_strategy.encode([expanded_query])[0]
            query_is_adaptive = True
            if should_log("DEBUG"):
                print(f"🎯 Query embedding creado usando estrategia adaptativa")
        else:
            # Usar el modelo principal del enricher
            model = enricher._get_embedding_model()
            query_embedding = model.encode(
                [expanded_query], convert_to_tensor=True, device=CURRENT_DEVICE
            )[0]
            query_is_adaptive = False
            if should_log("DEBUG"):
                print(f"🎯 Query embedding creado usando modelo principal: {query_embedding.shape}")

        # NUEVA CORRECCIÓN: Asegurar que el query embedding esté en el dispositivo correcto
        query_embedding = ensure_tensor_device(query_embedding, CURRENT_DEVICE)

    except Exception as e:
        print(f"⚠️ Error crítico creando query embedding: {e}")
        print("🔄 Intentando fallback con modelo por defecto...")
        try:
            # Fallback: usar modelo básico
            from sentence_transformers import SentenceTransformer

            fallback_model = SentenceTransformer("sentence-transformers/LaBSE")
            fallback_model = fallback_model.to(
                CURRENT_DEVICE
            )  # Mover modelo a dispositivo correcto
            query_embedding = fallback_model.encode(
                [expanded_query], convert_to_tensor=True, device=CURRENT_DEVICE
            )[0]
            query_embedding = ensure_tensor_device(query_embedding, CURRENT_DEVICE)
            query_is_adaptive = False
            print(f"✅ Fallback exitoso con LaBSE")
        except Exception as fallback_error:
            print(f"❌ Error crítico en fallback: {fallback_error}")
            return []

    # NUEVA CORRECCIÓN: Normalizar todos los embeddings de entidades al dispositivo correcto
    print(f"🔧 Normalizando embeddings al dispositivo {CURRENT_DEVICE}...")
    enriched_embeddings = normalize_embeddings_device(
        enriched_embeddings, CURRENT_DEVICE
    )

    # Preparar términos de búsqueda para boost dinámico con filtro de símbolos
    import re

    # Limpiar la query de símbolos de puntuación
    cleaned_query = re.sub(r'[¿?¡!,;:()"\'\[\]{}]', " ", query)
    # Limpiar espacios múltiples
    cleaned_query = re.sub(r"\s+", " ", cleaned_query.strip())

    # Extraer términos significativos (> 2 caracteres) y filtrar stopwords básicas
    stopwords_basic = {
        "que",
        "del",
        "de",
        "la",
        "el",
        "los",
        "las",
        "un",
        "una",
        "es",
        "para",
        "con",
        "en",
        "al",
        "se",
        "por",
    }
    query_terms = [
        term.lower().strip()
        for term in cleaned_query.split()
        if len(term.strip()) > 2 and term.lower().strip() not in stopwords_basic
    ]

    if should_log("DEBUG"):
        print(f"🔍 Query original: '{query}'")
        print(f"🧹 Query limpia: '{cleaned_query}'")
        print(f"🎯 Términos extraídos para boost: {query_terms}")

    # Pre-filtrado de entidades para optimizar procesamiento
    relevant_entities = {}
    irrelevant_count = 0

    for entity, embedding in enriched_embeddings.items():
        entity_name = entity.split("/")[-1].split("#")[-1].lower()

        # Filtrar entidades obviamente irrelevantes
        # 1. Evitar entidades de sistema muy básicas
        if entity_name in [
            "int",
            "string",
            "float",
            "double",
            "boolean",
            "sample",
            "object",
            "null",
        ]:
            irrelevant_count += 1
            continue

        # 2. Evitar implementaciones muy específicas si la query es genérica
        is_generic_query = any(
            term in query.lower()
            for term in ["que", "tipos", "cuales", "what", "types"]
        )
        if is_generic_query and (
            "implementation" in entity_name and len(entity_name) > 20
        ):
            irrelevant_count += 1
            continue

        # 3. Solo procesar si hay algún potencial de relevancia
        has_potential = False

        # METEO FIX: Permitir METEO pasar pre-filtrado
        if "meteo" in entity_name:
            has_potential = True
            if should_log("DEBUG"):
                print(f"🎯 METEO FIX: Permitiendo {entity_name} pasar pre-filtrado")
        else:
            # Relevancia por términos de query (coincidencias exactas y parciales)
            for term in query_terms:
                if term in entity_name:
                    has_potential = True
                    break

            # Sistema de coincidencias parciales con puntuación
            if not has_potential:
                partial_match_score = calculate_partial_match_score(
                    query_terms, entity_name
                )
                if (
                    partial_match_score >= 0.4
                ):  # Umbral para coincidencias parciales significativas
                    has_potential = True
                    if should_log("DEBUG"):
                        print(f"🎯 COINCIDENCIA PARCIAL: {entity_name} (score: {partial_match_score:.2f})")

            # Relevancia por conceptos clave detectados dinámicamente
            key_concepts = []
            try:
                # Usar términos detectados automáticamente del dominio actual
                import kg_embedding

                if (
                    hasattr(kg_embedding, "DOMAIN_SPECIFIC_TERMS")
                    and kg_embedding.DOMAIN_SPECIFIC_TERMS
                ):
                    # Usar los términos más relevantes del dominio actual
                    key_concepts = list(kg_embedding.DOMAIN_SPECIFIC_TERMS.keys())[:15]

                # Agregar términos genéricos universales si no hay específicos
                if not key_concepts:
                    key_concepts = ["type", "class", "entity", "element", "concept"]

            except Exception as e:
                # Fallback: términos genéricos que funcionan en cualquier ontología
                key_concepts = ["type", "class", "entity", "element", "concept"]

            if any(concept in entity_name for concept in key_concepts):
                has_potential = True

        # Relevancia por nodos visibles
        if visible_nodes:
            entity_display_name = entity.split("/")[-1].split("#")[-1]
            if entity_display_name in visible_nodes:
                has_potential = True

        if has_potential:
            relevant_entities[entity] = embedding
        else:
            irrelevant_count += 1

    logger.log_pre_filtering(len(relevant_entities), irrelevant_count)
    logger.log_scoring_mode(is_hybrid_scoring_enabled())

    # Calcular similitudes con sistema de puntuación múltiple
    similarities = []
    debug_samples = []

    for entity, embedding in relevant_entities.items():
        # 1. Similitud base por embeddings (MEJORADA con normalización)
        entity_name = entity.split("/")[-1].split("#")[-1]

        # CORRECCIÓN CRÍTICA: Cálculo de similitud con modelo consistente
        try:
            # Asegurar que el embedding esté en el dispositivo correcto antes de procesar
            embedding = ensure_tensor_device(embedding, CURRENT_DEVICE)

            # NUEVA LÓGICA: Comparación directa usando embeddings compatibles
            if query_is_adaptive and hasattr(enricher, "embedding_strategy"):
                # Si tanto query como entidad usan estrategia adaptativa, la comparación es válida
                base_embedding_sim = safe_cosine_similarity(
                    query_embedding, embedding, CURRENT_DEVICE
                )
            elif not query_is_adaptive:
                # Query usa modelo único, verificar compatibilidad de dimensiones
                query_dim = (
                    query_embedding.shape[-1]
                    if hasattr(query_embedding, "shape")
                    else len(query_embedding)
                )
                entity_dim = (
                    embedding.shape[-1]
                    if hasattr(embedding, "shape")
                    else len(embedding)
                )

                if query_dim == entity_dim:
                    # Dimensiones compatibles, comparación válida
                    base_embedding_sim = safe_cosine_similarity(
                        query_embedding, embedding, CURRENT_DEVICE
                    )
                else:
                    # Dimensiones incompatibles - esto indica un problema serio
                    print(f"⚠️ INCOMPATIBILIDAD: Query dim={query_dim}, Entity '{entity_name}' dim={entity_dim}")
                    # Usar similitud basada en nombres como fallback más conservador
                    base_embedding_sim = 0.0
            else:
                # Caso mixto: query adaptativo pero entidad de modelo único - usar fallback
                print(f"⚠️ Caso mixto detectado para {entity_name}, usando fallback")
                base_embedding_sim = 0.0

            # 1. Similitud por embeddings mejorada con normalización
            try:
                # IMPORTANTE: Solo proceder si tenemos una similitud de embedding válida
                if base_embedding_sim > 0.0:
                    # Usar similitud base directamente - el embedding semántico debe ser la base
                    max_similarity = base_embedding_sim

                    # Bonus moderado por coincidencias exactas de palabras
                    query_words = set(query.lower().split())
                    entity_words = set(
                        normalize_entity_name_for_embedding(entity_name).split()
                    )

                    common_words = query_words.intersection(entity_words)
                    if common_words:
                        # Reducimos el bonus para que no domine sobre el embedding semántico
                        word_bonus = len(common_words) / max(
                            len(query_words), len(entity_words)
                        )
                        max_similarity += word_bonus * 0.08  # Reducido de 0.15 a 0.08

                    # Bonus semántico para similitudes muy altas (más conservador)
                    if base_embedding_sim >= 0.75:  # Umbral más alto
                        semantic_bonus = (
                            base_embedding_sim - 0.75
                        ) * 0.3  # Reducido de 0.5 a 0.3
                        max_similarity += semantic_bonus

                    embedding_sim = min(max_similarity, 1.0)
                else:
                    # Si no hay similitud de embedding válida, usar solo coincidencias de palabras (muy reducido)
                    query_words = set(query.lower().split())
                    entity_words = set(
                        normalize_entity_name_for_embedding(entity_name).split()
                    )

                    common_words = query_words.intersection(entity_words)
                    if common_words:
                        # Penalizar fuertemente la ausencia de embedding semántico
                        embedding_sim = (
                            len(common_words)
                            / max(len(query_words), len(entity_words))
                            * 0.3
                        )
                    else:
                        embedding_sim = 0.0

                # Bonus multilingüe solo si hay base semántica sólida
                if embedding_sim >= 0.4:  # Solo aplicar si ya hay buena similitud base
                    entity_name_lower = entity_name.lower()
                    query_lower = query.lower()

                    # Detectar conceptos multilingües usando patrones básicos y fiables
                    multilingual_detected = False
                    universal_multilingual_patterns = [
                        ("tipo", "type"),
                        ("tipos", "types"),
                        ("clase", "class"),
                        ("clases", "classes"),
                        ("dato", "data"),
                        ("datos", "data"),
                        ("entidad", "entity"),
                        ("entidades", "entities"),
                    ]

                    for term_es, term_en in universal_multilingual_patterns:
                        if term_es in query_lower and term_en in entity_name_lower:
                            multilingual_detected = True
                            break
                        elif term_en in query_lower and term_es in entity_name_lower:
                            multilingual_detected = True
                            break

                    if multilingual_detected:
                        multilingual_bonus = 0.12  # Reducido de 0.2 a 0.12
                        embedding_sim += multilingual_bonus
                        embedding_sim = min(embedding_sim, 1.0)

            except Exception as e:
                print(f"⚠️ Error en enhanced similarity para {entity_name}: {e}")
                # Fallback muy conservador
                query_words = set(query.lower().split())
                entity_words = set(
                    normalize_entity_name_for_embedding(entity_name).split()
                )
                common_words = query_words.intersection(entity_words)
                if common_words:
                    embedding_sim = (
                        len(common_words)
                        / max(len(query_words), len(entity_words))
                        * 0.2
                    )
                else:
                    embedding_sim = 0.0

        except Exception as e:
            print(f"⚠️ Error crítico procesando {entity_name}: {e}")
            # Fallback final muy básico
            query_words = set(query.lower().split())
            entity_words = set(normalize_entity_name_for_embedding(entity_name).split())
            common_words = query_words.intersection(entity_words)
            if common_words:
                embedding_sim = (
                    len(common_words) / max(len(query_words), len(entity_words)) * 0.15
                )
            else:
                embedding_sim = 0.0

        # 2. Bonus por anotaciones (sistema dinámico o legacy)
        if (
            hasattr(enricher, "calculate_annotation_bonus_dynamic")
            and enricher.semantic_classification
        ):
            annotation_bonus = enricher.calculate_annotation_bonus_dynamic(
                query, entity
            )
        else:
            # Fallback al sistema original
            annotation_bonus = enricher.calculate_annotation_bonus(query, entity)

        # 3. Boost dinámico por coincidencias en nombre de entidad
        name_boost = calculate_dynamic_name_boost(entity, query_terms, max_boost=0.4)

        # 4. SCORING HÍBRIDO KGE + TEXT EMBEDDINGS (OPTIMIZADO)
        # Solo aplicar scoring híbrido si hay relevancia mínima para eficiencia
        base_relevance = embedding_sim + annotation_bonus + name_boost

        if (
            is_hybrid_scoring_enabled() and base_relevance >= 0.1
        ):  # Umbral de relevancia mínima
            # Importar variables del modelo KGE
            import kg_embedding

            hybrid_score, hybrid_details = calculate_hybrid_score(
                query=query,
                target_entity=entity,
                text_similarity=embedding_sim,
                enriched_embeddings=enriched_embeddings,
                kge_model=getattr(kg_embedding, "TRAINED_KGE_MODEL", None),
                entity_factory=getattr(kg_embedding, "KGE_ENTITY_FACTORY", None),
            )

            # Usar el score híbrido como base
            base_score = hybrid_score

            # Debug para scoring híbrido solo en entidades relevantes
            if (
                len(debug_samples) < 3 and base_relevance >= 0.3
            ):  # Solo debug para entidades más relevantes
                logger.log_hybrid_scoring(
                    entity.split("/")[-1].split("#")[-1], hybrid_details
                )
        else:
            # Usar solo text embeddings si el híbrido no está habilitado o no hay relevancia mínima
            base_score = embedding_sim

        # 5. Boost para nodos visibles (MEJORADO)
        visible_boost = 0.0
        if visible_nodes:
            entity_name = entity.split("/")[-1].split("#")[-1]
            if entity_name in visible_nodes:
                visible_boost = 1.2  # Boost MUY significativo para nodos visibles
                logger.log_boost(entity_name, "visible", visible_boost)

        # 6. Boost multilingüe genérico (SIN HARDCODING)
        concept_boost = 0.0
        entity_name_lower = entity.split("/")[-1].split("#")[-1].lower()

        # Aplicar boost multilingüe genérico
        boosted_score, boost_amount, boost_description = apply_multilingual_boost(
            query, entity_name_lower, base_score, ontology_mapping
        )

        if boost_amount > 0:
            concept_boost = (
                boost_amount - 1.0
            )  # Convertir multiplicador a additive boost
            logger.log_boost(
                entity_name_lower, "multilingual", concept_boost, boost_description
            )

        # Boost adicional para similitud directa entre términos (respaldo)
        for query_term in query_terms:
            if len(query_term) >= 3:  # Solo términos significativos
                # Similitud directa (contiene el término)
                if query_term in entity_name_lower:
                    additional_boost = 0.8
                    concept_boost = max(concept_boost, additional_boost)
                    logger.log_boost(
                        entity_name_lower,
                        "direct_match",
                        additional_boost,
                        f"'{query_term}' in entity name",
                    )

                # Similitud por raíz común (primeras 4 letras)
                elif len(query_term) >= 4 and query_term[:4] in entity_name_lower:
                    additional_boost = 0.4
                    concept_boost = max(concept_boost, additional_boost)
                    logger.log_boost(
                        entity_name_lower,
                        "root_match",
                        additional_boost,
                        f"'{query_term[:4]}' root match",
                    )

                # Similitud inversa (entity name en query term)
                elif len(entity_name_lower) >= 3 and entity_name_lower in query_term:
                    additional_boost = 0.6
                    concept_boost = max(concept_boost, additional_boost)
                    logger.log_boost(
                        entity_name_lower,
                        "inverse_match",
                        additional_boost,
                        f"entity name in '{query_term}'",
                    )

        # Boost extra para nodos visibles que también son conceptualmente relevantes
        if visible_boost > 0 and concept_boost > 0:
            concept_boost *= 1.5  # Multiplicador adicional para nodos visibles + conceptualmente relevantes
            logger.log_boost(
                entity_name_lower,
                "visible+conceptual",
                concept_boost,
                "visible node with conceptual relevance",
            )

        # Puntuación final combinada (FIXED: Sistema híbrido aditivo-multiplicativo)
        # NUEVO SISTEMA DE SCORING: Priorizar embedding semántico
        # El embedding debe ser la base sólida, los boosts solo para refinar

        if is_hybrid_scoring_enabled():
            base_relevance = base_score
        else:
            base_relevance = embedding_sim

        # NUEVO ENFOQUE: Sistema principalmente multiplicativo que premia la similaridad semántica
        if (
            base_relevance >= 0.5
        ):  # Similitud semántica alta - usar sistema multiplicativo fuerte
            base_component = base_relevance
            multiplier = 1.0
            if annotation_bonus > 0:
                multiplier *= 1.0 + annotation_bonus * 0.3  # Reducido de 0.5
            if name_boost > 0:
                multiplier *= 1.0 + name_boost * 0.2  # Reducido de 0.3
            if visible_boost > 0:
                multiplier *= 1.0 + visible_boost * 0.15  # Reducido de 0.2
            if concept_boost > 0:
                multiplier *= 1.0 + concept_boost * 0.25  # Reducido de 0.4
            final_score = base_component * multiplier
            scoring_method = "multiplicative_high_semantic"

        elif (
            base_relevance >= 0.3
        ):  # Similitud semántica moderada - sistema mixto conservador
            base_component = base_relevance * 0.8  # El embedding domina
            boost_component = (
                annotation_bonus + name_boost + visible_boost + concept_boost
            ) * 0.3  # Boosts limitados
            final_score = base_component + boost_component
            scoring_method = "mixed_moderate_semantic"

        elif (
            base_relevance >= 0.1
        ):  # Similitud semántica baja - penalizar pero permitir algunos boosts
            base_component = base_relevance * 0.6  # Penalizar embedding bajo
            boost_component = (
                annotation_bonus + name_boost + visible_boost + concept_boost
            ) * 0.4  # Boosts moderados
            final_score = base_component + boost_component
            scoring_method = "penalized_low_semantic"

        else:  # Similitud semántica muy baja - penalizar fuertemente
            # Solo permitir scores muy bajos, independientemente de los boosts
            base_component = max(0.05, base_relevance * 0.5)
            boost_component = (
                annotation_bonus + name_boost + visible_boost + concept_boost
            ) * 0.2  # Boosts muy limitados
            final_score = min(
                0.3, base_component + boost_component
            )  # Limitar score máximo
            scoring_method = "heavily_penalized_no_semantic"

        # Debug para primeras 5 entidades
        if len(debug_samples) < 5:
            entity_name = entity.split("/")[-1].split("#")[-1]

            debug_entry = {
                "entity": entity_name,
                "annotation_bonus": annotation_bonus,
                "name_boost": name_boost,
                "visible_boost": visible_boost,
                "concept_boost": concept_boost,
                "final_score": final_score,
                "scoring_method": scoring_method,  # Usar el método determinado dinámicamente
            }

            if is_hybrid_scoring_enabled():
                debug_entry.update({"base_score": base_score, "hybrid_enabled": True})
            else:
                debug_entry.update(
                    {"embedding_sim": embedding_sim, "hybrid_enabled": False}
                )

            debug_samples.append(debug_entry)

        # Log detallado para entender por qué no se encuentran entidades
        if should_log("MINIMAL"):
            entity_name = entity.split("/")[-1].split("#")[-1]
            print(f"🔍 Procesando {entity_name}: emb_sim={embedding_sim:.3f}, ann_bonus={annotation_bonus:.3f}, name_boost={name_boost:.3f}, vis_boost={visible_boost:.3f}, concept_boost={concept_boost:.3f} -> final={final_score:.3f} (threshold={threshold}) [{scoring_method}]")

        if final_score >= threshold:
            similarities.append((entity, final_score))
        elif should_log("MINIMAL") and final_score > 0:
            # Log entidades que tienen score positivo pero no pasan el threshold
            entity_name = entity.split("/")[-1].split("#")[-1]
            print(f"❌ {entity_name} no pasa threshold: {final_score:.3f} < {threshold}")

    # Log de resumen de procesamiento
    if should_log("MINIMAL"):
        print(f"📊 Procesadas {len(relevant_entities)} entidades relevantes")
        print(f"📊 {len(similarities)} entidades pasaron el threshold de {threshold}")

        if len(similarities) == 0 and len(relevant_entities) > 0:
            # Si no hay resultados, mostrar información básica
            print(f"🔍 No hay resultados que pasen el threshold {threshold}")
            entity_names = [
                e.split("/")[-1].split("#")[-1]
                for e in list(relevant_entities.keys())[:5]
            ]
            print(f"   Entidades evaluadas (muestra): {entity_names}")

    # Mostrar resumen de boosts aplicados
    logger.log_boost_summary()

    # Debug output simplificado
    if should_log("DEBUG") and debug_samples:
        mode = (
            "HÍBRIDA (KGE + Text)"
            if is_hybrid_scoring_enabled()
            else "Solo Text Embeddings"
        )
        print(f"📊 Ejemplo de puntuación {mode}:")
        for sample in debug_samples[:3]:  # Solo mostrar 3 ejemplos
            entity_name = sample["entity"]
            final_score = sample["final_score"]
            print(f"   {entity_name}: {final_score:.3f}")

    # Ordenar por puntuación y retornar top_n
    similarities.sort(key=lambda x: x[1], reverse=True)

    # *** APLICAR FILTRO DE ENTIDADES NO DESEADAS ***
    from kg_embedding import filter_unwanted_entities

    # Convertir formato para filtro
    entities_with_scores = [
        {"entity": entity, "score": score} for entity, score in similarities
    ]
    filtered_entities = filter_unwanted_entities(entities_with_scores)

    # Convertir de vuelta al formato original
    results = [
        (entity_info["entity"], entity_info["score"])
        for entity_info in filtered_entities
    ]

    # Contar entidades filtradas
    entities_filtered = len(similarities) - len(results)

    # Aplicar límite final
    results = results[:top_n]  # Aplicar límite después del filtrado

    # Log de resultados finales
    logger.log_results(results, entities_filtered)

    return results

def is_stopword_or_functional(term: str) -> bool:
    """
    Verifica si un término es una stopword o palabra funcional que no debe contribuir al boost.
    """
    if len(term) < 3:
        return True  # Términos muy cortos generalmente no son significativos

    # Stopwords multilingües que causan false positives
    stopwords = {
        # Español
        'con', 'por', 'para', 'que', 'del', 'las', 'los', 'una', 'uno', 'son', 'está', 'este', 'esta',
        'desde', 'hasta', 'entre', 'sobre', 'bajo', 'sin', 'como', 'muy', 'más', 'también', 'pero',
        'sin', 'cual', 'cuando', 'donde', 'porque', 'mientras', 'durante', 'mediante', 'según',
        
        # Inglés
        'the', 'and', 'for', 'are', 'can', 'has', 'was', 'you', 'all', 'any', 'may', 'our', 'out',
        'who', 'how', 'now', 'man', 'new', 'use', 'her', 'way', 'day', 'get', 'own', 'say', 'see',
        'him', 'two', 'oil', 'sit', 'set', 'but', 'not', 'had', 'his', 'one', 'its', 'why', 'let',
        'put', 'end', 'why', 'try', 'ask', 'too', 'old', 'run', 'cut', 'low', 'off', 'far', 'sea',
        'hot', 'lot', 'got', 'top', 'yet', 'big', 'job', 'few', 'box', 'war', 'car', 'eye', 'yes',
        
        # Palabras conectoras y funcionales
        'with', 'from', 'they', 'have', 'this', 'that', 'will', 'been', 'into', 'only', 'just',
        'also', 'some', 'what', 'time', 'very', 'when', 'much', 'good', 'make', 'take', 'than',
        'them', 'well', 'were', 'back', 'over', 'then', 'here', 'would', 'could', 'should',
        
        # Alemán
        'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'und', 'mit', 'von', 'für',
        'auf', 'bei', 'ist', 'war', 'hat', 'wie', 'aus', 'vor', 'zur', 'vom', 'zum', 'man',
        
        # Francés  
        'les', 'des', 'une', 'dans', 'pour', 'avec', 'par', 'sur', 'qui', 'que', 'est', 'ont',
        'son', 'ses', 'aux', 'ces', 'nos', 'vos',
        
        # Términos técnicos genéricos que causan problemas
        'owl', 'rdf', 'rdfs', 'xml', 'uri', 'url', 'www', 'http', 'xsd', 'dom', 'api', 'src', 'ref'
    }

    return term.lower() in stopwords

def is_context_relevant_term(term: str, query_context: str) -> bool:
    """
    Determina si un término es relevante para el contexto de la consulta.
    GENERALIZADO: Detecta automáticamente el dominio de la consulta y evalúa relevancia.
    """
    if len(term) < 3:
        return False

    # Lista expandida de dominios comunes con detección automática
    domain_keywords = {
        "food": [
            "food",
            "dish",
            "ingredient",
            "cooking",
            "recipe",
            "meal",
            "cuisine",
            "eat",
            "taste",
            "flavor",
            "nutrition",
        ],
        "medical": [
            "patient",
            "disease",
            "treatment",
            "symptom",
            "diagnosis",
            "medicine",
            "health",
            "medical",
            "clinic",
            "hospital",
        ],
        "technology": [
            "computer",
            "software",
            "program",
            "code",
            "algorithm",
            "data",
            "system",
            "network",
            "internet",
            "web",
            "app",
            "technology",
            "digital",
        ],
        "automotive": [
            "car",
            "vehicle",
            "engine",
            "automotive",
            "drive",
            "motor",
            "transport",
            "fuel",
            "mechanic",
            "repair",
        ],
        "academic": [
            "university",
            "course",
            "student",
            "professor",
            "research",
            "study",
            "education",
            "academic",
            "school",
            "learning",
        ],
        "geography": [
            "country",
            "city",
            "region",
            "location",
            "place",
            "continent",
            "geography",
            "map",
            "border",
            "territory",
        ],
        "business": [
            "company",
            "business",
            "market",
            "product",
            "service",
            "customer",
            "finance",
            "economy",
            "corporate",
            "industry",
        ],
        "science": [
            "research",
            "experiment",
            "theory",
            "hypothesis",
            "analysis",
            "scientific",
            "laboratory",
            "method",
            "observation",
        ],
        "legal": [
            "law",
            "legal",
            "court",
            "judge",
            "attorney",
            "contract",
            "regulation",
            "compliance",
            "justice",
            "rights",
        ],
        "arts": [
            "art",
            "music",
            "painting",
            "sculpture",
            "creative",
            "artist",
            "cultural",
            "aesthetic",
            "design",
            "artistic",
        ],
    }

    query_lower = query_context.lower()
    term_lower = term.lower()

    # Detectar el dominio principal de la consulta
    detected_domains = []
    for domain, keywords in domain_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            detected_domains.append(domain)

    # Si no se detecta un dominio específico, ser permisivo
    if not detected_domains:
        return True

    # Si el término pertenece a alguno de los dominios detectados, es relevante
    for domain in detected_domains:
        domain_terms = domain_keywords[domain]
        if any(domain_term in term_lower for domain_term in domain_terms):
            return True

    # También verificar si el término está en DOMAIN_SPECIFIC_TERMS (detectados automáticamente)
    try:
        import kg_embedding
        if hasattr(kg_embedding, 'DOMAIN_SPECIFIC_TERMS') and kg_embedding.DOMAIN_SPECIFIC_TERMS and term_lower in kg_embedding.DOMAIN_SPECIFIC_TERMS:
            return True
    except ImportError:
        pass

    # Términos genéricos siempre son relevantes
    generic_terms = {
        "type",
        "class",
        "category",
        "group",
        "kind",
        "sort",
        "variety",
        "form",
    }
    if term_lower in generic_terms:
        return True

    # Términos largos suelen ser específicos y relevantes
    if len(term) > 6:
        return True

    return False

def is_semantic_negation(term: str, entity_name: str) -> bool:
    """
    Detecta si un término es semánticamente contradictorio con una entidad.
    GENERALIZADO: Sistema basado en patrones comunes y detección automática de oposiciones.
    """
    term_lower = term.lower()
    entity_lower = entity_name.lower()

    # Patrones generales de oposición semántica comunes en ontologías
    # Estos son patrones universales que aparecen en muchos dominios
    semantic_oppositions = {
        # Conceptos naturales/artificiales (común en ontologías genéricas)
        "natural": ["artificial", "sintético", "procesado", "manufacturado"],
        "artificial": ["natural", "orgánico", "nativo", "biológico"],
        # Conceptos de temperatura/intensidad (común en muchas ontologías)
        "hot": ["cold", "cool", "mild", "frozen"],
        "cold": ["hot", "warm", "heated"],
        "spicy": ["mild", "bland", "sweet"],
        "mild": ["spicy", "hot", "strong", "intense"],
        # Conceptos de tamaño/grosor
        "thick": ["thin", "slim", "narrow"],
        "thin": ["thick", "deep", "wide"],
        "large": ["small", "tiny", "mini"],
        "small": ["large", "big", "huge"],
        # Conceptos de velocidad/tiempo
        "fast": ["slow", "gradual"],
        "slow": ["fast", "quick", "rapid"],
        "new": ["old", "ancient", "vintage"],
        "old": ["new", "modern", "recent"],
        # Conceptos de calidad/estado
        "good": ["bad", "poor", "terrible"],
        "bad": ["good", "excellent", "perfect"],
        "clean": ["dirty", "messy", "polluted"],
        "dirty": ["clean", "pure", "pristine"],
        # Conceptos de actividad/estado
        "active": ["passive", "inactive", "dormant"],
        "passive": ["active", "dynamic", "energetic"],
        "open": ["closed", "sealed", "locked"],
        "closed": ["open", "accessible", "available"],
        # Conceptos de posición/orientación
        "top": ["bottom", "base", "lower"],
        "bottom": ["top", "upper", "peak"],
        "left": ["right"],
        "right": ["left"],
        "front": ["back", "rear"],
        "back": ["front"],
        # Conceptos de luz/oscuridad
        "light": ["dark", "heavy", "dense"],
        "dark": ["light", "bright", "illuminated"],
        "bright": ["dark", "dim", "dull"],
        "dim": ["bright", "intense", "vivid"],
        # Conceptos de inclusión/exclusión
        "include": ["exclude", "omit", "remove"],
        "exclude": ["include", "contain", "add"],
        "with": ["without", "lacking"],
        "without": ["with", "including", "containing"],
        # Conceptos de salud/enfermedad
        "healthy": ["sick", "diseased", "unhealthy"],
        "sick": ["healthy", "well", "recovered"],
        # Conceptos de naturaleza/artificial
        "natural": ["artificial", "synthetic", "manufactured"],
        "artificial": ["natural", "organic", "native"],
        # Conceptos de seguridad
        "safe": ["dangerous", "risky", "harmful"],
        "dangerous": ["safe", "secure", "protected"],
    }

    # Verificar si hay oposición semántica directa
    if term_lower in semantic_oppositions:
        opposition_terms = semantic_oppositions[term_lower]
        if any(opp_term in entity_lower for opp_term in opposition_terms):
            return True

    # Verificar prefijos/sufijos negativos automáticamente
    return detect_negative_prefix_contradiction(term_lower, entity_lower)


def detect_negative_prefix_contradiction(term: str, entity_name: str) -> bool:
    """
    Detecta si una entidad tiene prefijos negativos que contradicen el término de búsqueda.
    Sistema general que funciona para cualquier ontología.

    Por ejemplo:
    - "concepto" vs "NonConcepto" → True (contradictorio)
    - "elemento" vs "NoElemento" → True (contradictorio)
    - "propiedad" vs "SinPropiedad" → True (contradictorio)
    """
    term_lower = term.lower()
    entity_lower = entity_name.lower()

    # Prefijos negativos comunes en ontologías
    negative_prefixes = [
        "non",  # Prefijo común de negación en inglés
        "not",  # Prefijo común de negación en inglés
        "un",  # UnhealthyFood
        "in",  # InappropriateFood
        "anti",  # AntiInflammatoryFood
        "dis",  # DisallowedIngredient
        "counter",  # CounterIndication
        "opposite",  # OppositeType
        "contra",  # ContraIndicated
    ]

    # Sufijos negativos
    negative_suffixes = [
        "free",  # ElementoFree cuando buscan "elemento"
        "less",  # ElementoLess cuando buscan "elemento"
        "without",  # SinElemento cuando buscan "elemento"
    ]

    # Verificar prefijos negativos
    for prefix in negative_prefixes:
        negative_term = f"{prefix}{term_lower}"
        if negative_term in entity_lower:
            return True

        # También verificar con separadores comunes
        negative_term_sep = f"{prefix}_{term_lower}"
        if negative_term_sep in entity_lower:
            return True

        negative_term_sep2 = f"{prefix}-{term_lower}"
        if negative_term_sep2 in entity_lower:
            return True

    # Verificar sufijos negativos cuando el término aparece
    if term_lower in entity_lower:
        for suffix in negative_suffixes:
            negative_pattern = f"{term_lower}{suffix}"
            if negative_pattern in entity_lower:
                return True

            negative_pattern_sep = f"{term_lower}_{suffix}"
            if negative_pattern_sep in entity_lower:
                return True

            negative_pattern_sep2 = f"{term_lower}-{suffix}"
            if negative_pattern_sep2 in entity_lower:
                return True
    return False

def calculate_dynamic_name_boost(
    entity: str, query_terms: List[str], max_boost: float = 0.5
) -> float:
    """
    Calcula boost dinámico basado en coincidencias textuales en el nombre de la entidad.
    Sistema mejorado que otorga scores más granulares y diferenciados con filtrado de stopwords.
    Maneja mejor palabras compuestas como VegetarianPizza vs "vegetarian pizza"

    Args:
        entity: URI de la entidad
        query_terms: Lista de términos de búsqueda (ya en minúsculas)
        max_boost: Boost máximo a otorgar

    Returns:
        Valor de boost entre 0 y max_boost
    """
    # Extraer nombre de la entidad desde la URI
    entity_name_original = entity.split("/")[-1].split("#")[-1]
    entity_name = entity_name_original.lower()

    # Crear versión normalizada para mejor matching
    entity_name_normalized = normalize_entity_name_for_embedding(entity_name_original)
    entity_variants = [entity_name_original, entity_name_normalized]
    all_entity_forms = [entity_name, entity_name_normalized] + [
        v.lower() for v in entity_variants
    ]
    all_entity_forms = list(set(all_entity_forms))  # Eliminar duplicados

    boost = 0.0
    matched_terms = []
    quality_score = 0.0  # Para calcular calidad del match

    for term in query_terms:
        if len(term) < 3:  # Ignorar términos muy cortos
            continue

        # *** Filtrar stopwords y términos irrelevantes ***
        if is_stopword_or_functional(term):
            continue

        # Combinar términos para análisis de contexto
        full_query = " ".join(query_terms)
        if not is_context_relevant_term(term, full_query):
            continue

        # *** Detectar negaciones semánticas ***
        # Verificar contra todas las formas de la entidad
        has_semantic_negation = any(
            is_semantic_negation(term, entity_form) for entity_form in all_entity_forms
        )
        if has_semantic_negation:
            penalty = -0.8
            boost += penalty
            matched_terms.append(f"semantic_negation:{penalty:.3f}")
            continue

        # *** Detectar prefijos negativos ***
        has_negative_prefix = any(
            detect_negative_prefix_contradiction(term, entity_form)
            for entity_form in all_entity_forms
        )
        if has_negative_prefix:
            penalty = -0.6
            boost += penalty
            matched_terms.append(f"negative_prefix:{penalty:.3f}")
            continue

        # Buscar el mejor match entre todas las formas de la entidad
        best_term_boost = 0.0
        best_match_quality = 0.0
        best_match_type = ""

        for entity_form in all_entity_forms:
            term_boost = 0.0
            match_quality = 0.0
            match_type = ""

            # 1. Coincidencia exacta del término completo
            if term == entity_form:
                term_boost = 0.25  # Máximo boost para match perfecto
                match_quality = 1.0
                match_type = f"perfect:{term}"

            # 2. Coincidencia exacta del término como subcadena
            elif term in entity_form:
                # Calcular calidad basada en posición y longitud relativa
                position_factor = 1.0 if entity_name.startswith(term) else 0.8
                length_factor = len(term) / len(entity_name)
                term_boost = 0.18 * position_factor * (0.5 + length_factor)
                match_quality = 0.9 * position_factor
                matched_terms.append(f"exact:{term}")

            # 3. El nombre de la entidad comienza con el término
            elif entity_name.startswith(term):
                length_factor = len(term) / len(entity_name)
                term_boost = 0.15 * (0.6 + length_factor)
                match_quality = 0.8
                matched_terms.append(f"prefix:{term}")

            # 4. El término está al inicio de palabras compuestas (CamelCase/underscore)
            elif any(
                word.startswith(term)
                for word in entity_name.split("_")
                + [w.lower() for w in _extract_camel_case_words(entity_name)]
            ):
                term_boost = 0.12
                match_quality = 0.7
                matched_terms.append(f"compound:{term}")

            # 5. Variaciones lingüísticas con scoring granular
            elif _check_term_variations_scored(term, entity_name):
                variation_score = _get_variation_score(term, entity_name)
                term_boost = 0.08 * variation_score
                match_quality = 0.6 * variation_score
                matched_terms.append(f"variant:{term}")

            # 6. Coincidencias parciales inteligentes
            elif _check_intelligent_substring(term, entity_name):
                substring_score = _get_substring_quality(term, entity_name)
                term_boost = 0.05 * substring_score
                match_quality = 0.4 * substring_score
                matched_terms.append(f"partial:{term}")

        boost += term_boost
        quality_score += match_quality

    # Bonus dinámico por múltiples coincidencias (escala con calidad)
    if len(matched_terms) > 1:
        multi_bonus = (
            0.03 * (len(matched_terms) - 1) * (quality_score / len(matched_terms))
        )
        boost += multi_bonus
        matched_terms.append(f"multi_bonus:{multi_bonus:.3f}")

    # Penalización por nombres muy largos (reduce false positives)
    if len(entity_name) > 25:
        length_penalty = boost * 0.1 * ((len(entity_name) - 25) / 25)
        boost -= length_penalty

    final_boost = min(boost, max_boost)

    # Debug mejorado con scoring detallado
    if final_boost > 0.05:
        print(f"   🎯 {entity_name}: boost={final_boost:.4f} quality={quality_score:.2f} [{', '.join(matched_terms)}]")

    return final_boost


def _check_term_variations(term: str, entity_name: str) -> bool:
    """
    Verifica variaciones comunes de términos (plurales, derivados, etc.)

    Args:
        term: Término de búsqueda
        entity_name: Nombre de la entidad

    Returns:
        True si se encuentra una variación
    """
    # Variaciones comunes a verificar
    variations = [
        f"{term}s",  # plural
        f"{term}y",  # cheesy, spicy
        f"{term}ing",  # cooking, topping
        f"{term}ed",  # baked, sliced
        term[:-1] if term.endswith("s") and len(term) > 3 else "",  # singular
        term[:-1] if term.endswith("y") and len(term) > 3 else "",  # base form
    ]

    # Filtrar variaciones vacías y verificar coincidencias
    for variation in variations:
        if variation and len(variation) > 2 and variation in entity_name:
            return True

    return False


def _extract_camel_case_words(text: str) -> List[str]:
    """Extrae palabras de texto en CamelCase"""
    return re.findall(r"[A-Z][a-z]*", text.replace("_", "").title())


def _check_term_variations_scored(term: str, entity_name: str) -> bool:
    """Versión con scoring de variaciones de términos"""
    return _check_term_variations(term, entity_name)


def _get_variation_score(term: str, entity_name: str) -> float:
    """Calcula score de calidad para variaciones de términos"""
    variations = [
        (f"{term}s", 0.9),  # plural - alta similaridad
        (f"{term}y", 0.8),  # adjetivo - alta similaridad
        (f"{term}ing", 0.7),  # gerundio - media similaridad
        (f"{term}ed", 0.6),  # pasado - media similaridad
        (term[:-1] if term.endswith("s") and len(term) > 3 else "", 0.9),  # singular
        (term[:-1] if term.endswith("y") and len(term) > 3 else "", 0.8),  # base form
    ]

    best_score = 0.0
    for variation, score in variations:
        if variation and len(variation) > 2 and variation in entity_name:
            best_score = max(best_score, score)

    return best_score


def _check_intelligent_substring(term: str, entity_name: str) -> bool:
    """Verifica coincidencias de subcadena inteligentes"""
    if len(term) < 4:
        return False

    # Buscar subcadenas en posiciones significativas
    words = entity_name.replace("_", " ").replace("-", " ").split()
    for word in words:
        if len(word) >= len(term) and term in word:
            return True

    return False


def _get_substring_quality(term: str, entity_name: str) -> float:
    """Calcula calidad de coincidencia de subcadena"""
    if len(term) < 4:
        return 0.0

    best_quality = 0.0
    words = entity_name.replace("_", " ").replace("-", " ").split()

    for word in words:
        if term in word:
            # Factor basado en longitud relativa
            length_factor = len(term) / len(word)
            # Factor basado en posición (inicio es mejor)
            position_factor = 1.0 if word.startswith(term) else 0.7
            quality = length_factor * position_factor
            best_quality = max(best_quality, quality)

    return best_quality


# ============================================================================
# FUNCIONES DE SCORING HÍBRIDO KGE + TEXT EMBEDDINGS
# ============================================================================


def calculate_kge_similarity(
    query_entities: List[str], target_entity: str, kge_model=None, entity_factory=None
) -> float:
    """
    Calcula la similitud usando embeddings del modelo KGE entrenado.
    Más robusto y usa múltiples estrategias de cálculo.

    Args:
        query_entities: Lista de entidades derivadas de la query
        target_entity: Entidad objetivo
        kge_model: Modelo KGE entrenado
        entity_factory: Factory de entidades para mapear a IDs

    Returns:
        Similitud KGE normalizada entre 0 y 1
    """
    if kge_model is None or entity_factory is None or not query_entities:
        return 0.0

    try:
        import torch
        from sentence_transformers import util

        # Obtener embeddings de entidades del modelo KGE
        entity_to_id = entity_factory.entity_to_id

        # Verificar si la entidad objetivo existe en el modelo
        if target_entity not in entity_to_id:
            # Si la entidad objetivo no está en el modelo KGE, usar similitud por nombre
            return _calculate_fallback_similarity(query_entities, target_entity)

        target_id = entity_to_id[target_entity]
        target_id_tensor = torch.tensor([target_id], device=CURRENT_DEVICE)
        target_embedding = kge_model.entity_representations[0](target_id_tensor)[0]
        
        # NUEVA LÍNEA: Convertir embedding complejo a real si es necesario
        if target_embedding.dtype.is_complex:
            target_embedding = target_embedding.real
        
        # Asegurar que el embedding del target esté en el dispositivo correcto
        target_embedding = ensure_tensor_device(target_embedding, CURRENT_DEVICE)

        # Estrategia 1: Similitud directa con entidades de la query
        direct_similarities = []

        for query_entity in query_entities:
            if query_entity in entity_to_id:
                query_id = entity_to_id[query_entity]
                query_id_tensor = torch.tensor([query_id], device=CURRENT_DEVICE)
                query_embedding = kge_model.entity_representations[0](query_id_tensor)[
                    0
                ]
                
                # NUEVA LÍNEA: Convertir embedding complejo a real si es necesario
                if query_embedding.dtype.is_complex:
                    query_embedding = query_embedding.real
                
                # Asegurar que el embedding de la query esté en el dispositivo correcto
                query_embedding = ensure_tensor_device(query_embedding, CURRENT_DEVICE)

                # Calcular similitud coseno usando función segura
                similarity = safe_cosine_similarity(
                    query_embedding, target_embedding, CURRENT_DEVICE
                )
                direct_similarities.append(similarity)

        # Estrategia 2: Similitud promedio si tenemos múltiples entidades
        if len(direct_similarities) > 1:
            # Usar promedio ponderado - dar más peso a las mejores similitudes
            sorted_sims = sorted(direct_similarities, reverse=True)
            if len(sorted_sims) >= 2:
                weighted_avg = sorted_sims[0] * 0.6 + sorted_sims[1] * 0.4
                max_similarity = max(max(direct_similarities), weighted_avg)
            else:
                max_similarity = max(direct_similarities)
        elif len(direct_similarities) == 1:
            max_similarity = direct_similarities[0]
        else:
            # Estrategia 3: Si no hay coincidencias directas, buscar entidades relacionadas
            max_similarity = _find_related_entity_similarity(
                query_entities, target_entity, kge_model, entity_factory
            )

        # Normalizar a [0,1] y aplicar suavizado
        normalized_sim = (max_similarity + 1) / 2  # De [-1,1] a [0,1]

        # Aplicar boost si la similitud es alta
        if normalized_sim > 0.7:
            normalized_sim = min(
                1.0, normalized_sim * 1.1
            )  # Boost del 10% para similitudes altas

        return float(normalized_sim)

    except Exception as e:
        print(f"⚠️ Error calculando similitud KGE: {e}")
        # Fallback a similitud por nombres
        return _calculate_fallback_similarity(query_entities, target_entity)


def _calculate_fallback_similarity(
    query_entities: List[str], target_entity: str
) -> float:
    """
    Calcula similitud de fallback basada en nombres cuando KGE no está disponible.
    """
    if not query_entities:
        return 0.0

    target_name = target_entity.split("/")[-1].split("#")[-1].lower()
    target_components = _extract_entity_components(
        target_entity.split("/")[-1].split("#")[-1]
    )
    target_components_lower = [comp.lower() for comp in target_components]

    max_similarity = 0.0

    for query_entity in query_entities:
        query_name = query_entity.split("/")[-1].split("#")[-1].lower()
        query_components = _extract_entity_components(
            query_entity.split("/")[-1].split("#")[-1]
        )
        query_components_lower = [comp.lower() for comp in query_components]

        # Similitud exacta
        if query_name == target_name:
            max_similarity = max(max_similarity, 0.9)

        # Similitud de componentes
        common_components = set(query_components_lower) & set(target_components_lower)
        if common_components:
            component_similarity = len(common_components) / max(
                len(query_components_lower), len(target_components_lower)
            )
            max_similarity = max(max_similarity, component_similarity * 0.7)

        # Similitud substring
        if query_name in target_name or target_name in query_name:
            max_similarity = max(max_similarity, 0.5)

    return max_similarity


def _find_related_entity_similarity(
    query_entities: List[str], target_entity: str, kge_model, entity_factory
) -> float:
    """
    Busca similitud con entidades relacionadas cuando no hay match directo.
    """
    try:
        import torch
        from sentence_transformers import util

        entity_to_id = entity_factory.entity_to_id

        # Obtener embedding de la entidad objetivo
        if target_entity not in entity_to_id:
            return 0.0

        target_id = entity_to_id[target_entity]
        target_embedding = kge_model.entity_representations[0](
            torch.tensor([target_id])
        )[0]
        
        # NUEVA LÍNEA: Convertir embedding complejo a real si es necesario
        if target_embedding.dtype.is_complex:
            target_embedding = target_embedding.real

        # Buscar entidades con nombres similares en el espacio KGE
        target_name_lower = target_entity.split("/")[-1].split("#")[-1].lower()
        similar_entities = []

        for entity_uri in entity_to_id.keys():
            entity_name = entity_uri.split("/")[-1].split("#")[-1].lower()

            # Buscar entidades que compartan componentes con las query entities
            for query_entity in query_entities:
                query_name = query_entity.split("/")[-1].split("#")[-1].lower()

                # Si comparten algún componente semántico
                if any(
                    comp in entity_name for comp in query_name.split() if len(comp) > 2
                ):
                    similar_entities.append(entity_uri)
                    break

        # Calcular similitud con entidades relacionadas
        max_related_similarity = 0.0

        for related_entity in similar_entities[:3]:  # Limitar a 3 para eficiencia
            if related_entity in entity_to_id:
                related_id = entity_to_id[related_entity]
                related_embedding = kge_model.entity_representations[0](
                    torch.tensor([related_id])
                )[0]

                # NUEVA LÍNEA: Convertir embedding complejo a real si es necesario
                if related_embedding.dtype.is_complex:
                    related_embedding = related_embedding.real

                similarity = safe_cosine_similarity(
                    related_embedding, target_embedding, CURRENT_DEVICE
                )
                max_related_similarity = max(
                    max_related_similarity, similarity * 0.7
                )  # Penalizar por ser indirecto

        return max_related_similarity

    except Exception:
        return 0.0


def extract_entities_from_query(query: str, enriched_embeddings: Dict) -> List[str]:
    """
    Extrae entidades candidatas de la query para usar en KGE similarity.
    Usa mapeo multilingüe y análisis semántico más profundo.

    Args:
        query: Query en lenguaje natural
        enriched_embeddings: Diccionario de embeddings para buscar entidades

    Returns:
        Lista de URIs de entidades relacionadas con la query
    """
    query_lower = query.lower()
    candidate_entities = []
    entity_scores = {}  # Para ordenar por relevancia

    # 1. Crear mapeo multilingüe dinámico basado en entidades disponibles
    available_entities = list(enriched_embeddings.keys())
    entity_names = [
        entity.split("/")[-1].split("#")[-1] for entity in available_entities
    ]
    multilingual_mapping = create_generic_multilingual_mapping(entity_names)

    # 2. Buscar entidades usando mapeo multilingüe
    for spanish_term, english_candidates in multilingual_mapping.items():
        if spanish_term in query_lower:
            for english_candidate in english_candidates:
                # Buscar entidades que contengan este candidato
                for entity_uri in available_entities:
                    entity_name = entity_uri.split("/")[-1].split("#")[-1].lower()

                    # Coincidencia exacta con nombre completo
                    if english_candidate.lower() == entity_name:
                        entity_scores[entity_uri] = (
                            entity_scores.get(entity_uri, 0) + 3.0
                        )
                    # Coincidencia con componentes de entidad compuesta
                    elif english_candidate.lower() in entity_name:
                        entity_components = _extract_entity_components(
                            entity_uri.split("/")[-1].split("#")[-1]
                        )
                        entity_components_lower = [
                            comp.lower() for comp in entity_components
                        ]

                        if english_candidate.lower() in entity_components_lower:
                            entity_scores[entity_uri] = (
                                entity_scores.get(entity_uri, 0) + 2.0
                            )
                        else:
                            entity_scores[entity_uri] = (
                                entity_scores.get(entity_uri, 0) + 1.0
                            )

    # 3. Buscar entidades usando términos directos de la query
    query_terms = [term for term in query_lower.split() if len(term) > 2]

    for entity_uri in available_entities:
        entity_name = entity_uri.split("/")[-1].split("#")[-1].lower()
        entity_components = _extract_entity_components(
            entity_uri.split("/")[-1].split("#")[-1]
        )
        entity_components_lower = [comp.lower() for comp in entity_components]

        for term in query_terms:
            # Coincidencia exacta con nombre
            if term == entity_name:
                entity_scores[entity_uri] = entity_scores.get(entity_uri, 0) + 2.5
            # Coincidencia con componente exacto
            elif term in entity_components_lower:
                entity_scores[entity_uri] = entity_scores.get(entity_uri, 0) + 2.0
            # Coincidencia substring
            elif term in entity_name and len(term) >= 4:
                entity_scores[entity_uri] = entity_scores.get(entity_uri, 0) + 1.0

    # 4. Ordenar por score y devolver los mejores candidatos
    sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)

    # Devolver hasta 5 entidades más relevantes con score >= 1.0
    candidate_entities = [entity for entity, score in sorted_entities if score >= 1.0][:5]

    return candidate_entities


def calculate_hybrid_score(
    query: str,
    target_entity: str,
    text_similarity: float,
    enriched_embeddings: Dict,
    kge_model=None,
    entity_factory=None,
) -> Tuple[float, Dict]:
    """
    Calcula el score híbrido combinando KGE y text embeddings.

    Args:
        query: Query original
        target_entity: Entidad objetivo
        text_similarity: Similitud de text embeddings ya calculada
        enriched_embeddings: Embeddings enriquecidos
        kge_model: Modelo KGE entrenado
        entity_factory: Factory de entidades

    Returns:
        Tupla de (score_final, detalles_scoring)
    """
    if not is_hybrid_scoring_enabled():
        return text_similarity, {
            "text_similarity": text_similarity,
            "hybrid_enabled": False,
        }

    config = get_hybrid_scoring_config()

    # Calcular similitud KGE
    query_entities = extract_entities_from_query(query, enriched_embeddings)
    kge_similarity = calculate_kge_similarity(
        query_entities, target_entity, kge_model, entity_factory
    )

    # Pesos de la configuración
    kge_weight = config["weights"]["kge_similarity"]
    text_weight = config["weights"]["text_similarity"]

    # Score híbrido básico
    hybrid_score = (kge_similarity * kge_weight) + (text_similarity * text_weight)

    # Aplicar boosts
    boosts = config["boosts"]
    boost_multiplier = 1.0

    entity_name = target_entity.split("/")[-1].split("#")[-1].lower()

    # Boost para matches exactos en el nombre
    if any(term.lower() in entity_name for term in query.split() if len(term) > 2):
        boost_multiplier *= boosts["exact_match"]

    # Aplicar boost
    final_score = hybrid_score * boost_multiplier

    # Detalles para debug
    details = {
        "text_similarity": text_similarity,
        "kge_similarity": kge_similarity,
        "hybrid_score": hybrid_score,
        "boost_multiplier": boost_multiplier,
        "final_score": final_score,
        "query_entities_found": len(query_entities),
        "hybrid_enabled": True,
    }

    return final_score, details


def create_generic_multilingual_mapping(entity_names):
    """
    Crea un mapeo multilingüe completamente dinámico basado en los nombres de entidades de la ontología actual.
    No usa términos hardcodeados y se adapta automáticamente a cualquier dominio.

    Args:
        entity_names: Lista de nombres de entidades de la ontología actual

    Returns:
        Diccionario con mapeos término -> entidad_candidata
    """
    mapping = {}

    # 1. PATRONES UNIVERSALES GENÉRICOS (no específicos de dominio)
    universal_base_patterns = {
        # Términos estructurales universales
        "tipos": ["type", "kind", "class"],
        "tipo": ["type", "kind", "class"],
        "clases": ["class", "type"],
        "clase": ["class", "type"],
        "elementos": ["element", "item", "component"],
        "elemento": ["element", "item", "component"],
        "componentes": ["component", "element", "part"],
        "componente": ["component", "element", "part"],
        "objetos": ["object", "item", "entity"],
        "objeto": ["object", "item", "entity"],
        "entidades": ["entity", "object", "item"],
        "entidad": ["entity", "object", "item"],
        "instancias": ["instance", "object"],
        "instancia": ["instance", "object"],
        "conceptos": ["concept", "notion"],
        "concepto": ["concept", "notion"],
        "categorías": ["category", "class", "group"],
        "categoría": ["category", "class", "group"],
        "grupos": ["group", "collection", "set"],
        "grupo": ["group", "collection", "set"],
        "colecciones": ["collection", "set", "group"],
        "colección": ["collection", "set", "group"],
        "elementos": ["element", "item"],
        "propiedades": ["property", "attribute"],
        "propiedad": ["property", "attribute"],
        "atributos": ["attribute", "property"],
        "atributo": ["attribute", "property"],
        "relaciones": ["relation", "relationship"],
        "relación": ["relation", "relationship"],
        "conexiones": ["connection", "link"],
        "conexión": ["connection", "link"],
        "vínculos": ["link", "connection"],
        "vínculo": ["link", "connection"],
    }

    # 2. DETECTAR TÉRMINOS ESPECÍFICOS DEL DOMINIO AUTOMÁTICAMENTE
    domain_specific_patterns = {}

    try:
        # Usar términos detectados automáticamente desde kg_embedding
        import kg_embedding

        # Obtener términos de dominio detectados automáticamente
        if (
            hasattr(kg_embedding, "DOMAIN_SPECIFIC_TERMS")
            and kg_embedding.DOMAIN_SPECIFIC_TERMS
        ):
            for term_key, term_value in kg_embedding.DOMAIN_SPECIFIC_TERMS.items():
                # Crear mapeos bidireccionales automáticos
                if len(term_key) > 2:
                    domain_specific_patterns[term_key] = [term_value, term_key]
                    # Agregar plural si es apropiado
                    if not term_key.endswith("s"):
                        plural_key = term_key + "s"
                        domain_specific_patterns[plural_key] = [term_value, term_key]

        # Usar mapas dinámicos si están disponibles
        if (
            hasattr(kg_embedding, "DYNAMIC_TERMS_MAP")
            and kg_embedding.DYNAMIC_TERMS_MAP
        ):
            for dynamic_term, mapped_entity in kg_embedding.DYNAMIC_TERMS_MAP.items():
                if len(dynamic_term) > 2:
                    domain_specific_patterns[dynamic_term] = [mapped_entity]

    except Exception as e:
        # Si no se puede acceder a kg_embedding, continuar sin términos específicos
        pass

    # 3. ANÁLISIS AUTOMÁTICO DE ENTIDADES ACTUALES
    detected_patterns = {}

    # Analizar patrones en los nombres de entidades actuales
    word_frequency = {}
    for entity_name in entity_names:
        # Dividir CamelCase y extraer palabras
        words = re.findall(r"[A-Z][a-z]*|[a-z]+", entity_name)
        for word in words:
            if len(word) > 2:  # Solo palabras significativas
                word_lower = word.lower()
                word_frequency[word_lower] = word_frequency.get(word_lower, 0) + 1

    # Usar palabras frecuentes como base para mapeo
    frequent_words = [word for word, freq in word_frequency.items() if freq >= 2]

    # Crear mapeos automáticos para palabras frecuentes
    for word in frequent_words:
        if len(word) > 3:
            # Mapeo directo
            detected_patterns[word] = [word]
            # Mapeo plural/singular automático
            if word.endswith("s") and len(word) > 4:
                singular = word[:-1]
                detected_patterns[singular] = [word, singular]
            elif not word.endswith("s"):
                plural = word + "s"
                detected_patterns[plural] = [word]

    # 4. COMBINAR TODOS LOS PATRONES
    all_patterns = {
        **universal_base_patterns,
        **domain_specific_patterns,
        **detected_patterns,
    }

    # 5. CREAR MAPEO FINAL
    for term, candidates in all_patterns.items():
        matching_entities = []

        # Buscar entidades que coincidan con los candidatos
        for candidate in candidates:
            for entity_name in entity_names:
                entity_lower = entity_name.lower()
                candidate_lower = candidate.lower()

                # Coincidencia exacta o por substring inteligente
                if (
                    candidate_lower == entity_lower
                    or candidate_lower in entity_lower
                    or entity_lower in candidate_lower
                    or
                    # Coincidencia por raíz (primeras 4 letras)
                    (
                        len(candidate_lower) >= 4
                        and len(entity_lower) >= 4
                        and candidate_lower[:4] == entity_lower[:4]
                    )
                ):

                    if entity_name not in matching_entities:
                        matching_entities.append(entity_name)

        # Solo agregar al mapeo si hay coincidencias
        if matching_entities:
            mapping[term] = matching_entities[:5]  # Máximo 5 candidatos por término

    return mapping


def apply_multilingual_boost(
    query: str, entity_name: str, base_score: float, ontology_mapping: dict
) -> tuple:
    """
    Aplica boost multilingüe genérico basado en la ontología actual.
    Detecta mejor la relación semántica entre consultas y entidades compuestas.

    Args:
        query: Consulta del usuario
        entity_name: Nombre de la entidad
        base_score: Score base
        ontology_mapping: Mapeo generado por create_generic_multilingual_mapping

    Returns:
        Tupla (score_final, boost_aplicado, descripcion_boost)
    """
    query_lower = query.lower()
    entity_lower = entity_name.lower()

    # 0. Boost dinámico para frases completas basado en la ontología actual
    # En lugar de hardcodear, detectamos patrones de entidades compuestas dinámicamente

    # Extraer todas las entidades disponibles y sus componentes
    available_entities = list(ontology_mapping.values())
    flat_entities = [entity for sublist in available_entities for entity in sublist]

    # Detectar frases compuestas en la query que podrían mapear a entidades
    query_terms = query_lower.split()

    # Buscar combinaciones de términos consecutivos en la query
    for i in range(len(query_terms)):
        for j in range(
            i + 2, min(i + 4, len(query_terms) + 1)
        ):  # Frases de 2-3 palabras
            phrase = " ".join(query_terms[i:j])

            # Solo procesar frases significativas (sin palabras muy cortas)
            if (
                all(len(word) >= 3 for word in phrase.split())
                and len(phrase.split()) >= 2
            ):

                # Buscar en el mapeo multilingüe si esta frase tiene traducciones
                potential_targets = []

                # 1. Buscar mapeo directo de la frase completa
                if phrase in ontology_mapping:
                    potential_targets.extend(ontology_mapping[phrase])

                # 2. Buscar mapeos de términos individuales y combinarlos
                phrase_words = phrase.split()
                mapped_components = []

                for word in phrase_words:
                    if word in ontology_mapping:
                        mapped_components.extend(ontology_mapping[word])

                # Si tenemos componentes mapeados, buscar entidades que los contengan
                if len(mapped_components) >= 2:
                    for entity in flat_entities:
                        entity_components = _extract_entity_components(entity)
                        entity_components_lower = [
                            comp.lower() for comp in entity_components
                        ]

                        # Contar cuántos componentes de la frase están en la entidad
                        matches = 0
                        for mapped_comp in mapped_components:
                            if mapped_comp.lower() in entity_components_lower:
                                matches += 1

                        # Si la entidad actual coincide y tiene suficientes componentes
                        if entity.lower() == entity_name.lower() and matches >= 2:
                            boost_strength = 2.5 + (matches * 0.2)  # Boost proporcional
                            return (
                                base_score * boost_strength,
                                boost_strength,
                                f"frase_dinamica_{phrase}->{entity}",
                            )
                        elif entity.lower() == entity_name.lower() and matches >= 1:
                            boost_strength = 2.0 + (matches * 0.3)
                            return (
                                base_score * boost_strength,
                                boost_strength,
                                f"frase_parcial_{phrase}->{entity}",
                            )

    # 1. Boost por coincidencia directa (cualquier idioma)
    query_words = [w.strip() for w in query_lower.split() if len(w.strip()) >= 3]

    for word in query_words:
        # Coincidencia directa
        if word in entity_lower:
            return base_score * 2.0, 2.0, f"coincidencia_directa_{word}"

        # Coincidencia parcial significativa
        if len(word) >= 4 and word in entity_lower:
            return base_score * 1.5, 1.5, f"coincidencia_parcial_{word}"

    # 2. Boost por mapeo multilingüe MEJORADO con análisis de componentes
    multilingual_boost = 0.0
    matched_components = []

    for spanish_term, english_candidates in ontology_mapping.items():
        if spanish_term in query_lower:
            for english_candidate in english_candidates:
                candidate_lower = english_candidate.lower()

                # Coincidencia exacta con entidad completa
                if candidate_lower == entity_lower:
                    return (
                        base_score * 2.5,
                        2.5,
                        f"mapeo_exacto_{spanish_term}->{english_candidate}",
                    )

                # Coincidencia exacta con componente de entidad compuesta
                elif candidate_lower in entity_lower:
                    # Verificar que sea un componente completo, no solo substring
                    entity_components = _extract_entity_components(entity_name)
                    for component in entity_components:
                        if candidate_lower == component.lower():
                            component_boost = 2.0  # Boost alto para componente exacto
                            matched_components.append(f"{spanish_term}->{component}")
                            multilingual_boost = max(
                                multilingual_boost, component_boost
                            )
                            break
                    else:
                        # Substring pero no componente completo - boost menor
                        if (
                            len(candidate_lower) >= 4
                        ):  # Solo para términos significativos
                            substring_boost = 1.3
                            matched_components.append(
                                f"{spanish_term}~>{english_candidate}"
                            )
                            multilingual_boost = max(
                                multilingual_boost, substring_boost
                            )

    # Boost especial para entidades compuestas que contienen múltiples conceptos de la query
    if len(matched_components) >= 2:
        multilingual_boost *= 1.2  # Bonus adicional por múltiples componentes
        return (
            base_score * multilingual_boost,
            multilingual_boost,
            f"mapeo_multiple_{';'.join(matched_components)}",
        )
    elif multilingual_boost > 0:
        return (
            base_score * multilingual_boost,
            multilingual_boost,
            f"mapeo_componente_{matched_components[0]}",
        )

    # 3. Análisis semántico profundo para entidades compuestas
    semantic_boost = _calculate_semantic_component_boost(
        query_lower, entity_name, ontology_mapping
    )
    if semantic_boost > 0:
        return base_score * semantic_boost, semantic_boost, f"semantico_compuesto"

    # 4. Boost por similitud fonética/ortográfica usando mapeo dinámico
    phonetic_boost = 0.0
    matched_pattern = ""

    try:
        # Usar mapeo dinámico construido automáticamente
        import kg_embedding

        # Verificar mapeos dinámicos primero
        if (
            hasattr(kg_embedding, "DYNAMIC_TERMS_MAP")
            and kg_embedding.DYNAMIC_TERMS_MAP
        ):
            for term_es, term_en in kg_embedding.DYNAMIC_TERMS_MAP.items():
                if term_es in query_lower and term_en.lower() in entity_lower:
                    phonetic_boost = 2.2
                    matched_pattern = f"{term_es}->{term_en}"
                    break

        # Usar términos de dominio específico detectados automáticamente
        if (
            phonetic_boost == 0.0
            and hasattr(kg_embedding, "DOMAIN_SPECIFIC_TERMS")
            and kg_embedding.DOMAIN_SPECIFIC_TERMS
        ):
            for term_key, term_value in kg_embedding.DOMAIN_SPECIFIC_TERMS.items():
                # Buscar correspondencias bidireccionales
                if term_key in query_lower and term_value.lower() in entity_lower:
                    phonetic_boost = 2.0
                    matched_pattern = f"{term_key}->{term_value}"
                    break
                elif term_value.lower() in query_lower and term_key in entity_lower:
                    phonetic_boost = 2.0
                    matched_pattern = f"{term_value}->{term_key}"
                    break
    except Exception as e:
        # Si falla, usar patrones básicos universales como fallback
        basic_patterns = [
            ("tipo", "type"),
            ("tipos", "type"),
            ("clase", "class"),
            ("clases", "class"),
            ("elemento", "element"),
            ("elementos", "element"),
            ("objeto", "object"),
            ("objetos", "object"),
        ]

        for spanish_word, english_word in basic_patterns:
            if spanish_word in query_lower and english_word in entity_lower:
                phonetic_boost = 1.8
                matched_pattern = f"{spanish_word}->{english_word}"
                break

    if phonetic_boost > 0:
        return (
            base_score * phonetic_boost,
            phonetic_boost,
            f"similitud_fonetica_{matched_pattern}",
        )

    return base_score, 0.0, "sin_boost"


def _extract_entity_components(entity_name: str) -> list:
    """
    Extrae componentes semánticos de un nombre de entidad compuesta.
    Maneja CamelCase, underscore_case y otros patrones.
    """

    components = []

    # Dividir por underscore
    underscore_parts = entity_name.split("_")
    components.extend(underscore_parts)

    # Dividir CamelCase
    camel_parts = re.findall(r"[A-Z][a-z]*|[a-z]+", entity_name)
    components.extend(camel_parts)

    # Limpiar y filtrar componentes
    cleaned_components = []
    for comp in components:
        comp_clean = comp.strip()
        if len(comp_clean) >= 3:  # Solo componentes significativos
            cleaned_components.append(comp_clean)

    return list(set(cleaned_components))  # Eliminar duplicados


def _calculate_semantic_component_boost(
    query: str, entity_name: str, ontology_mapping: dict
) -> float:
    """
    Calcula boost semántico para entidades compuestas basado en cuántos
    conceptos de la query están representados en la entidad.

    Ejemplo: "tipos de datos" debe boost mucho "DataType" porque contiene tanto "Data" como "Type"
    """
    query_concepts = []
    entity_components = _extract_entity_components(entity_name)
    entity_components_lower = [comp.lower() for comp in entity_components]

    # Analizar la query para extraer conceptos - DINÁMICO sin hardcodear
    query_lower = query.lower()

    # 1. Usar SOLO el mapeo dinámico de la ontología actual
    for spanish_term, english_candidates in ontology_mapping.items():
        if spanish_term in query_lower:
            query_concepts.extend(
                [candidate.lower() for candidate in english_candidates]
            )

    # 2. Análisis semántico de frases compuestas dinámico
    query_terms = query_lower.split()

    # Buscar combinaciones de términos que podrían formar conceptos compuestos
    for i in range(len(query_terms)):
        for j in range(
            i + 2, min(i + 4, len(query_terms) + 1)
        ):  # Frases de 2-3 palabras
            phrase = " ".join(query_terms[i:j])

            # Si la frase completa está en el mapeo, usar sus conceptos
            if phrase in ontology_mapping:
                query_concepts.extend(
                    [candidate.lower() for candidate in ontology_mapping[phrase]]
                )
            else:
                # Si no, combinar conceptos de términos individuales
                phrase_concepts = []
                for word in phrase.split():
                    if word in ontology_mapping:
                        phrase_concepts.extend(
                            [candidate.lower() for candidate in ontology_mapping[word]]
                        )

                # Si tenemos al menos 2 conceptos de la frase, agregarlos
                if len(phrase_concepts) >= 2:
                    query_concepts.extend(phrase_concepts)

    # 3. Análisis semántico de entidades compuestas
    entity_lower = entity_name.lower()
    matching_concepts = 0
    total_query_concepts = len(set(query_concepts))

    if total_query_concepts == 0:
        return 0.0

    # Contar coincidencias conceptuales con mayor flexibilidad
    matched_concepts = set()
    for concept in set(query_concepts):
        concept_matched = False

        # Coincidencia exacta con nombre completo
        if concept == entity_lower:
            matched_concepts.add(f"exact:{concept}")
            concept_matched = True

        # Coincidencia con componentes de la entidad
        for entity_comp in entity_components_lower:
            if concept == entity_comp:
                matched_concepts.add(f"component_exact:{concept}")
                concept_matched = True
                break
            elif len(concept) >= 3 and (
                concept in entity_comp or entity_comp in concept
            ):
                matched_concepts.add(f"component_partial:{concept}")
                concept_matched = True
                break

        # Coincidencia substring en nombre completo
        if not concept_matched and len(concept) >= 4 and concept in entity_lower:
            matched_concepts.add(f"substring:{concept}")

    matching_concepts = len(matched_concepts)

    if matching_concepts == 0:
        return 0.0

    # Calcular boost basado en cobertura conceptual
    coverage_ratio = matching_concepts / total_query_concepts

    # Boost base según el ratio de cobertura
    if coverage_ratio >= 0.8:  # 80% o más de conceptos coinciden
        base_boost = 2.5
    elif coverage_ratio >= 0.6:  # 60% o más
        base_boost = 2.0
    elif coverage_ratio >= 0.4:  # 40% o más
        base_boost = 1.5
    else:
        base_boost = 1.2

    # Bonus adicional para entidades compuestas que tienen múltiples coincidencias exactas
    exact_matches = sum(
        1
        for match in matched_concepts
        if match.startswith("exact:") or match.startswith("component_exact:")
    )
    if exact_matches >= 2:
        base_boost *= 1.3  # Bonus del 30% para múltiples coincidencias exactas

    return base_boost


def calculate_partial_match_score(query_terms: List[str], entity_name: str) -> float:
    """
    Calcula un score de coincidencia parcial para términos compuestos.

    Ejemplo:
    - query_term = "meteo1.0.0"
    - entity_name = "meteodataset"
    - "meteo" está en "meteodataset" y representa 50% de "meteo1.0.0"
    - Score = 0.5

    Args:
        query_terms: Lista de términos de la query
        entity_name: Nombre de la entidad en minúsculas

    Returns:
        float: Score entre 0.0 y 1.0 representando la mejor coincidencia parcial
    """
    best_score = 0.0

    for term in query_terms:
        # Para términos compuestos (con puntos, números), extraer subcomponentes
        subcomponents = extract_term_subcomponents(term)

        for subcomp in subcomponents:
            if len(subcomp) >= 3 and subcomp in entity_name:
                # Calcular score basado en la proporción del subcomponente respecto al término original
                component_ratio = len(subcomp) / len(term)

                # Bonus por longitud del subcomponente (componentes más largos son más específicos)
                length_bonus = min(len(subcomp) / 8.0, 1.0)  # Máximo bonus a los 8 caracteres
                
                # Score final combinando proporción y longitud
                partial_score = component_ratio * 0.7 + length_bonus * 0.3

                # Bonus adicional si el subcomponente está al inicio (más significativo)
                if entity_name.startswith(subcomp):
                    partial_score *= 1.2

                best_score = max(best_score, partial_score)

    return min(best_score, 1.0)  # Limitar a 1.0


def extract_term_subcomponents(term: str) -> List[str]:
    """
    Extrae subcomponentes significativos de un término compuesto.

    Ejemplos:
    - "meteo1.0.0" → ["meteo", "meteo1", "1.0.0"]
    - "dataservice" → ["data", "service", "dataservice"]
    - "APIendpoint" → ["api", "endpoint", "apiendpoint"]

    Args:
        term: Término a descomponer

    Returns:
        Lista de subcomponentes ordenados por relevancia (más largos primero)
    """
    subcomponents = set()

    # 1. Agregar el término completo
    subcomponents.add(term.lower())

    # 2. Dividir por puntos, guiones, números
    parts = re.split(r"[.\-_0-9]+", term)
    for part in parts:
        if len(part) >= 3:
            subcomponents.add(part.lower())

    # 3. Dividir CamelCase
    camel_parts = re.findall(r"[A-Z][a-z]*|[a-z]+", term)
    for part in camel_parts:
        if len(part) >= 3:
            subcomponents.add(part.lower())

    # 4. Extraer prefijos y sufijos significativos
    if len(term) > 6:
        # Prefijos de 3-5 caracteres
        for i in range(3, min(6, len(term))):
            prefix = term[:i].lower()
            if prefix.isalpha():  # Solo prefijos alfabéticos
                subcomponents.add(prefix)

        # Sufijos de 3-5 caracteres
        for i in range(3, min(6, len(term))):
            suffix = term[-i:].lower()
            if suffix.isalpha():  # Solo sufijos alfabéticos
                subcomponents.add(suffix)

    # 5. Convertir a lista y ordenar por longitud (más largos primero)
    result = sorted(list(subcomponents), key=len, reverse=True)

    return result

