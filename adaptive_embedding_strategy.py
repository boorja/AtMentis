"""
Estrategia adaptativa de embeddings para diferentes tipos de contenido ontológico.
Usa modelos especializados según la longitud y tipo de texto.
"""

import re
import numpy as np

from collections import Counter, defaultdict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple


@dataclass
class ContentAnalysis:
    """Análisis de contenido para determinar el modelo óptimo"""

    length_chars: int
    word_count: int
    sentence_count: int
    is_technical: bool
    is_domain_specific: bool
    complexity_score: float


class AdaptiveEmbeddingStrategy:
    """
    Estrategia que selecciona el modelo de embedding más apropiado
    según las características del contenido.
    """

    def __init__(self, domain_indicators: Optional[set] = None):
        """
        Inicializa la estrategia adaptativa

        Args:
            domain_indicators: Set opcional de palabras clave específicas del dominio.
                              Si no se proporciona, se detectarán automáticamente.
        """
        self.models = {}
        self.model_configs = {
            # Para textos cortos y labels (hasta 100 chars)
            "short": {
                "model_name": "sentence-transformers/LaBSE",
                "max_chars": 100,
                "description": "Optimizado para labels y textos cortos multilingües",
            },
            # Para descripciones medias (100-300 chars)
            "medium": {
                "model_name": "sentence-transformers/all-mpnet-base-v2",
                "max_chars": 300,
                "description": "Excelente para párrafos y comprensión semántica",
            },
            # Para textos largos y técnicos (300+ chars)
            "long": {
                "model_name": "sentence-transformers/all-MiniLM-L12-v2",
                "max_chars": 1000,
                "description": "Maneja textos largos manteniendo eficiencia",
            },
            # Para contenido muy técnico
            "technical": {
                "model_name": "sentence-transformers/msmarco-distilbert-base-v4",
                "max_chars": 500,
                "description": "Especializado en contenido técnico y científico",
            },
        }

        # General technical keywords for ontology/semantic web content
        self.technical_keywords = {
            'ontology', 'class', 'property', 'domain', 'range', 'restriction',
            'cardinality', 'axiom', 'reasoning', 'inference', 'subclass',
            'equivalent', 'disjoint', 'constraint', 'semantic', 'owl', 'rdf',
            'individual', 'instance', 'datatype', 'annotation', 'namespace',
            'uri', 'iri', 'rdfs', 'xml', 'schema', 'taxonomy', 'hierarchy',
            'subsumption', 'classification', 'consistency', 'satisfiability'
        }

        # Domain-specific indicators will be automatically detected
        self.domain_indicators = domain_indicators or set()
        self.domain_analysis_enabled = True

    def analyze_content(self, text: str) -> ContentAnalysis:
        """Analiza el contenido para determinar sus características"""
        if not text:
            return ContentAnalysis(0, 0, 0, False, False, 0.0)

        # Estadísticas básicas
        length_chars = len(text)
        words = text.split()
        word_count = len(words)
        sentence_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])

        # Análisis técnico basado en palabras clave de ontologías
        text_lower = text.lower()
        technical_matches = sum(
            1 for keyword in self.technical_keywords if keyword in text_lower
        )
        is_technical = technical_matches >= 2

        # Análisis de especificidad del dominio (dinámico)
        domain_matches = 0
        if self.domain_indicators:
            domain_matches = sum(
                1 for indicator in self.domain_indicators if indicator in text_lower
            )

        # Si no hay indicadores de dominio definidos, usar un enfoque heurístico
        if not self.domain_indicators:
            # Buscar patrones comunes en ontologías: términos específicos repetidos
            words_lower = [w.lower() for w in words if len(w) > 3]
            word_freq = {}
            for word in words_lower:
                word_freq[word] = word_freq.get(word, 0) + 1

            # Considerar dominio específico si hay palabras repetidas no técnicas
            non_technical_repeated = [
                word
                for word, freq in word_freq.items()
                if freq > 1 and word not in self.technical_keywords
            ]
            domain_matches = len(non_technical_repeated)

        is_domain_specific = domain_matches >= 1

        # Puntuación de complejidad mejorada
        complexity_score = (
            (length_chars / 100) * 0.3
            + (technical_matches / max(len(self.technical_keywords), 1)) * 0.4
            + (sentence_count / 5) * 0.2
            + (domain_matches / max(len(self.domain_indicators) or 10, 1)) * 0.1
        )

        return ContentAnalysis(
            length_chars=length_chars,
            word_count=word_count,
            sentence_count=sentence_count,
            is_technical=is_technical,
            is_domain_specific=is_domain_specific,
            complexity_score=min(complexity_score, 1.0),
        )

    def select_model_type(self, analysis: ContentAnalysis) -> str:
        """Selecciona el tipo de modelo más apropiado"""
        # Priorizar por contenido técnico
        if analysis.is_technical and analysis.length_chars > 150:
            return "technical"

        # Seleccionar por longitud
        if analysis.length_chars <= 100:
            return "short"
        elif analysis.length_chars <= 300:
            return "medium"
        else:
            return "long"

    def get_model(self, model_type: str) -> SentenceTransformer:
        """Obtiene o carga el modelo especificado"""
        if model_type not in self.models:
            config = self.model_configs[model_type]
            print(f"🔄 Cargando modelo {model_type}: {config['model_name']}")
            print(f"   📝 {config['description']}")
            self.models[model_type] = SentenceTransformer(config["model_name"])

        return self.models[model_type]

    def create_adaptive_embeddings(
        self, texts: List[str], show_analysis: bool = True, learn_domain: bool = True
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Crea embeddings usando estrategia adaptativa

        Args:
            texts: Lista de textos para procesar
            show_analysis: Si mostrar análisis detallado
            learn_domain: Si aprender indicadores de dominio automáticamente

        Returns:
            embeddings_dict: Diccionario con embeddings
            strategy_stats: Estadísticas de la estrategia usada
        """
        if not texts:
            return {}, {}

        # Aprender características del dominio si está habilitado
        domain_stats = {}
        if learn_domain and self.domain_analysis_enabled:
            domain_stats = self.learn_from_ontology(texts, update_indicators=True)
            if show_analysis and domain_stats.get("detected_domain_indicators", 0) > 0:
                print(f"🎯 Detectados {domain_stats['detected_domain_indicators']} indicadores de dominio")
                print(f"   Ejemplos: {', '.join(domain_stats['domain_indicators'][:5])}")

        # Analizar todos los textos
        analyses = [self.analyze_content(text) for text in texts]

        # Agrupar por tipo de modelo recomendado
        model_groups = defaultdict(list)
        for i, analysis in enumerate(analyses):
            model_type = self.select_model_type(analysis)
            model_groups[model_type].append((i, texts[i], analysis))

        if show_analysis:
            print(f"\n🧠 ESTRATEGIA ADAPTATIVA DE EMBEDDINGS:")
            print(f"   📊 Total de textos: {len(texts)}")
            if domain_stats:
                print(f"   🎯 Indicadores de dominio: {len(self.domain_indicators)}")
                print(f"   📈 Distribución de complejidad: {domain_stats.get('complexity_distribution', {})}")
            
            for model_type, group_items in model_groups.items():
                config = self.model_configs[model_type]
                print(f"   🎯 {model_type}: {len(group_items)} textos → {config['model_name']}")
        
        # Crear embeddings por grupo
        all_embeddings = {}
        strategy_stats = {
            "total_texts": len(texts),
            "model_usage": {},
            "avg_complexity": np.mean([a.complexity_score for a in analyses]),
            "domain_stats": domain_stats,
        }

        for model_type, group_items in model_groups.items():
            if not group_items:
                continue

            model = self.get_model(model_type)
            group_texts = [item[1] for item in group_items]

            # Crear embeddings en batches
            batch_size = 16
            for i in range(0, len(group_texts), batch_size):
                batch_texts = group_texts[i : i + batch_size]
                batch_indices = [
                    group_items[j][0]
                    for j in range(i, min(i + batch_size, len(group_items)))
                ]

                embeddings = model.encode(batch_texts, convert_to_tensor=False)

                for idx, embedding in zip(batch_indices, embeddings):
                    all_embeddings[texts[idx]] = embedding

            strategy_stats["model_usage"][model_type] = {
                "count": len(group_items),
                "model_name": self.model_configs[model_type]["model_name"],
                "avg_length": np.mean([item[2].length_chars for item in group_items]),
                "technical_ratio": np.mean(
                    [item[2].is_technical for item in group_items]
                ),
            }

        if show_analysis:
            print(f"   ✅ Completado: {len(all_embeddings)} embeddings creados")
            print(f"   🎲 Complejidad promedio: {strategy_stats['avg_complexity']:.3f}")

        return all_embeddings, strategy_stats

    def get_strategy_summary(self) -> str:
        """Retorna un resumen de la estrategia disponible"""
        summary = "🧠 ESTRATEGIA ADAPTATIVA DE EMBEDDINGS:\n"
        for model_type, config in self.model_configs.items():
            summary += f"   {model_type}: {config['model_name']} (hasta {config['max_chars']} chars)\n"
            summary += f"      {config['description']}\n"
        return summary

    def auto_detect_domain_indicators(
        self, texts: List[str], min_frequency: int = 3
    ) -> set:
        """
        Detecta automáticamente indicadores de dominio a partir de los textos

        Args:
            texts: Lista de textos para analizar
            min_frequency: Frecuencia mínima para considerar una palabra como indicador

        Returns:
            Set de palabras clave específicas del dominio detectadas
        """
        # Extraer palabras candidatas
        word_counts = Counter()

        for text in texts:
            if not text:
                continue

            # Limpiar y tokenizar
            clean_text = re.sub(r"[^\w\s]", " ", text.lower())
            words = [
                w for w in clean_text.split() if len(w) > 3
            ]  # Solo palabras > 3 chars

            # Filtrar palabras comunes y técnicas
            domain_words = [
                w
                for w in words
                if w not in self.technical_keywords
                and w
                not in {
                    "this",
                    "that",
                    "with",
                    "from",
                    "they",
                    "have",
                    "been",
                    "were",
                    "will",
                    "said",
                    "each",
                    "which",
                    "their",
                    "time",
                    "would",
                    "there",
                    "could",
                    "other",
                }
            ]

            word_counts.update(domain_words)

        # Seleccionar palabras frecuentes como indicadores de dominio
        domain_indicators = {
            word for word, count in word_counts.items() if count >= min_frequency
        }

        return domain_indicators

    def learn_from_ontology(
        self, texts: List[str], update_indicators: bool = True
    ) -> Dict[str, any]:
        """
        Aprende características específicas de la ontología a partir de los textos

        Args:
            texts: Lista de textos de la ontología
            update_indicators: Si actualizar los indicadores de dominio automáticamente

        Returns:
            Diccionario con estadísticas del análisis
        """
        if not texts:
            return {}

        # Detectar indicadores de dominio automáticamente
        detected_indicators = self.auto_detect_domain_indicators(texts)

        if update_indicators:
            # Combinar con indicadores existentes
            self.domain_indicators.update(detected_indicators)

        # Análisis de características de la ontología
        analyses = [self.analyze_content(text) for text in texts]

        stats = {
            "total_texts": len(texts),
            "detected_domain_indicators": len(detected_indicators),
            "domain_indicators": list(detected_indicators)[
                :20
            ],  # Mostrar solo los primeros 20
            "avg_length": np.mean([a.length_chars for a in analyses]),
            "technical_ratio": np.mean([a.is_technical for a in analyses]),
            "complexity_distribution": {
                "low": sum(1 for a in analyses if a.complexity_score < 0.3),
                "medium": sum(1 for a in analyses if 0.3 <= a.complexity_score < 0.7),
                "high": sum(1 for a in analyses if a.complexity_score >= 0.7),
            },
        }

        return stats

    def get_domain_specific_summary(self) -> str:
        """Retorna un resumen específico del dominio aprendido"""
        if not self.domain_indicators:
            return "🎯 DOMINIO: No detectado aún (usar learn_from_ontology() primero)"

        summary = f"🎯 DOMINIO DETECTADO ({len(self.domain_indicators)} indicadores):\n"
        indicators_list = list(self.domain_indicators)[
            :10
        ]  # Mostrar solo los primeros 10
        summary += f"   Palabras clave: {', '.join(indicators_list)}\n"
        if len(self.domain_indicators) > 10:
            summary += f"   ... y {len(self.domain_indicators) - 10} más\n"
        return summary


# Función de conveniencia para usar la estrategia adaptativa
def create_adaptive_embeddings_for_annotations(
    annotations_dict: Dict[str, Dict],
    domain_indicators: Optional[set] = None,
    show_analysis: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Crea embeddings adaptativos para un diccionario de anotaciones

    Args:
        annotations_dict: Diccionario {entidad: {tipo: [textos]}}
        domain_indicators: Set opcional de indicadores de dominio específicos
        show_analysis: Si mostrar análisis detallado

    Returns:
        embeddings_dict: {texto: embedding}
        strategy_stats: Estadísticas de estrategia
    """
    strategy = AdaptiveEmbeddingStrategy(domain_indicators=domain_indicators)

    # Extraer todos los textos únicos
    all_texts = set()
    for entity_annotations in annotations_dict.values():
        for annotation_type, texts in entity_annotations.items():
            if isinstance(texts, list):
                all_texts.update(texts)
            else:
                all_texts.add(str(texts))

    # Crear embeddings adaptativos
    return strategy.create_adaptive_embeddings(list(all_texts), show_analysis)


def create_adaptive_strategy_for_ontology(
    triples: List[tuple], min_frequency: int = 3, show_analysis: bool = True
) -> AdaptiveEmbeddingStrategy:
    """
    Crea una estrategia adaptativa específicamente configurada para una ontología

    Args:
        triples: Lista de triples RDF (subject, predicate, object)
        min_frequency: Frecuencia mínima para detectar indicadores de dominio
        show_analysis: Si mostrar análisis del proceso

    Returns:
        Estrategia adaptativa configurada para la ontología
    """
    # Extraer textos de los triples para análisis
    texts = []
    for s, p, o in triples:
        # Extraer nombres de entidades y literales
        if isinstance(s, str):
            texts.append(s.split("#")[-1].split("/")[-1])
        if isinstance(p, str):
            texts.append(p.split("#")[-1].split("/")[-1])
        if isinstance(o, str):
            # Si parece un literal (contiene espacios o es largo), agregarlo completo
            if " " in o or len(o) > 50:
                texts.append(o)
            else:
                texts.append(o.split("#")[-1].split("/")[-1])

    # Crear estrategia y aprender del dominio
    strategy = AdaptiveEmbeddingStrategy()

    if show_analysis:
        print("🔍 Analizando ontología para configurar estrategia adaptativa...")

    domain_stats = strategy.learn_from_ontology(texts, update_indicators=True)

    if show_analysis:
        print(f"✅ Estrategia configurada para ontología:")
        print(f"   📊 {domain_stats['total_texts']} elementos analizados")
        print(f"   🎯 {domain_stats['detected_domain_indicators']} indicadores de dominio detectados")
        print(f"   📈 Complejidad promedio: {domain_stats.get('avg_length', 0):.1f} caracteres")
        print(strategy.get_domain_specific_summary())

    return strategy


if __name__ == "__main__":
    # Ejemplo de uso generalizado para cualquier ontología

    # Ejemplo 1: Estrategia básica sin dominio específico
    print("🧪 EJEMPLO 1: Estrategia básica")
    strategy = AdaptiveEmbeddingStrategy()
    print(strategy.get_strategy_summary())

    # Ejemplo 2: Con indicadores de dominio personalizados
    print("\n🧪 EJEMPLO 2: Con dominio personalizado (medical)")
    medical_indicators = {"patient", "disease", "treatment", "symptom", "diagnosis"}
    strategy_medical = AdaptiveEmbeddingStrategy(domain_indicators=medical_indicators)
    print(strategy_medical.get_domain_specific_summary())

    # Ejemplo 3: Aprendizaje automático de dominio
    print("\n🧪 EJEMPLO 3: Aprendizaje automático")
    sample_texts = [
        "Patient",  # Corto
        "A medical condition affecting the cardiovascular system",  # Medio
        "This class represents a complex neurological disorder characterized by progressive degeneration of motor neurons, leading to muscle weakness and atrophy. The condition is often diagnosed through electromyography and clinical assessment.",  # Largo y técnico
    ]

    strategy_auto = AdaptiveEmbeddingStrategy()
    embeddings, stats = strategy_auto.create_adaptive_embeddings(
        sample_texts, learn_domain=True
    )

    print(f"\n📊 Resultados:")
    print(f"   Embeddings creados: {len(embeddings)}")
    print(f"   Complejidad promedio: {stats['avg_complexity']:.3f}")
    if "domain_stats" in stats:
        print(f"   Indicadores detectados: {stats['domain_stats'].get('detected_domain_indicators', 0)}")

    print(strategy_auto.get_domain_specific_summary())
