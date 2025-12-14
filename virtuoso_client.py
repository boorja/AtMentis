#!/usr/bin/env python3
"""
Cliente para ejecutar consultas SPARQL usando la API de Virtuoso ya implementada.
"""
from typing import Dict, List, Any, Optional

# Configuración por defecto (puede ser sobrescrita)
DEFAULT_VIRTUOSO_CONFIG = {
    "endpoint": "http://192.168.216.102:8890/sparql-auth",
    "database": "https://khaos.example.org/v4",
    "username": "dba",
    "password": "password",
    "api_url": "http://192.168.216.102:32323/query_virtuoso",
}


class VirtuosoClient:
    """Cliente para ejecutar consultas SPARQL usando la API de Virtuoso existente."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el cliente con la configuración de Virtuoso.

        Args:
            config: Diccionario con configuración personalizada
        """
        self.config = config or DEFAULT_VIRTUOSO_CONFIG.copy()

    def execute_query(self, sparql_query: str) -> Dict[str, Any]:
        """
        Ejecuta una consulta SPARQL directamente al endpoint de Virtuoso.

        Args:
            sparql_query: Consulta SPARQL a ejecutar

        Returns:
            Diccionario con el resultado de la consulta

        Raises:
            Exception: Si hay error en la consulta
        """
        try:
            # Hacer consulta directa al endpoint SPARQL de Virtuoso
            from SPARQLWrapper import SPARQLWrapper, JSON
            from SPARQLWrapper.SPARQLExceptions import SPARQLWrapperException

            sparql = SPARQLWrapper(self.config["endpoint"])
            sparql.setCredentials(self.config["username"], self.config["password"])
            sparql.addDefaultGraph(self.config["database"])
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)

            result = sparql.queryAndConvert()
            return result

        except SPARQLWrapperException as e:
            raise Exception(f"Error SPARQL: {e}")
        except Exception as e:
            raise Exception(f"Error de conexión directa: {e}")

    def get_annotation_properties(self) -> List[Dict[str, str]]:
        """
        Obtiene todas las Annotation Properties disponibles en la ontología.

        Returns:
            Lista de diccionarios con URI y label de cada annotation property
        """
        query = """
        SELECT DISTINCT ?annotationProperty ?label
        WHERE {
            ?annotationProperty a owl:AnnotationProperty .
            OPTIONAL { ?annotationProperty rdfs:label ?label }
        }
        ORDER BY ?annotationProperty
        """

        try:
            result = self.execute_query(query)
            print(f"🔍 DEBUG: Respuesta directa SPARQL: {type(result)}")
            print(f"🔍 DEBUG: Keys en respuesta: {list(result.keys()) if isinstance(result, dict) else 'No es dict'}")

            # Extraer datos de la respuesta estándar SPARQL JSON
            properties = []

            if (
                isinstance(result, dict)
                and "results" in result
                and "bindings" in result["results"]
            ):
                print(f"🔍 DEBUG: Formato estándar SPARQL JSON - {len(result['results']['bindings'])} resultados")
                for binding in result["results"]["bindings"]:
                    if "annotationProperty" in binding:
                        prop_data = {"uri": binding["annotationProperty"]["value"]}
                        if "label" in binding and binding["label"].get("value"):
                            prop_data["label"] = binding["label"]["value"]
                        else:
                            prop_data["label"] = ""
                        properties.append(prop_data)
                        print(f"🔍 DEBUG: Añadida annotation property: {prop_data['uri']}")
            else:
                print(
                    "🔍 DEBUG: Formato de respuesta no reconocido para annotation properties"
                )
                print(f"🔍 DEBUG: Estructura: {result}")

            print(f"📝 Encontradas {len(properties)} annotation properties")
            return properties

        except Exception as e:
            print(f"⚠️ Error obteniendo annotation properties: {e}")
            return []

    def get_datatype_properties(self) -> List[Dict[str, str]]:
        """
        Obtiene todas las Datatype Properties disponibles en la ontología.

        Returns:
            Lista de diccionarios con URI, label y range de cada datatype property
        """
        query = """
        SELECT DISTINCT ?dataProperty ?label ?range
        WHERE {
            ?dataProperty a owl:DatatypeProperty .
            OPTIONAL { ?dataProperty rdfs:label ?label }
            OPTIONAL { ?dataProperty rdfs:range ?range }
        }
        ORDER BY ?dataProperty
        """

        try:
            result = self.execute_query(query)
            print(f"🔍 DEBUG: Datatype Properties SPARQL result: {result}")
            properties = self._extract_properties_from_result(result)
            print(f"📊 Encontradas {len(properties)} datatype properties")
            return properties
        except Exception as e:
            print(f"⚠️ Error obteniendo datatype properties: {e}")
            return []

    def get_object_properties(self) -> List[Dict[str, str]]:
        """
        Obtiene todas las Object Properties disponibles en la ontología.

        Returns:
            Lista de diccionarios con URI, label, domain y range de cada object property
        """
        query = """
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
            result = self.execute_query(query)
            print(f"🔍 DEBUG: Object Properties SPARQL result: {result}")
            properties = self._extract_properties_from_result(result)
            print(f"🔗 Encontradas {len(properties)} object properties")
            return properties
        except Exception as e:
            print(f"⚠️ Error obteniendo object properties: {e}")
            return []

    def get_property_usage_stats(self) -> List[Dict[str, Any]]:
        """
        Obtiene estadísticas de uso de propiedades en la ontología.

        Returns:
            Lista de diccionarios con predicate, usage_count y example_value
        """
        query = """
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
            result = self.execute_query(query)
            return self._extract_usage_stats_from_result(result)
        except Exception as e:
            print(f"⚠️ Error obteniendo estadísticas de uso: {e}")
            return []

    def get_literal_properties(self) -> List[Dict[str, str]]:
        """
        Obtiene propiedades que tienen valores literales (posibles anotaciones).

        Returns:
            Lista de diccionarios con property y example_literal
        """
        query = """
        SELECT DISTINCT ?property (SAMPLE(?literal) as ?example_literal)
        WHERE {
            ?subject ?property ?literal .
            FILTER(isLiteral(?literal))
            FILTER(STRLEN(STR(?literal)) > 3)
            FILTER(STRLEN(STR(?literal)) < 500)
        }
        GROUP BY ?property
        ORDER BY ?property
        LIMIT 50
        """

        try:
            result = self.execute_query(query)
            return self._extract_properties_from_result(result)
        except Exception as e:
            print(f"⚠️ Error obteniendo propiedades literales: {e}")
            return []

    def _extract_properties_from_result(
        self, result: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """
        Extrae información de propiedades del resultado SPARQL.

        Args:
            result: Resultado directo de SPARQL en formato estándar

        Returns:
            Lista de diccionarios con información de propiedades
        """
        properties = []

        # Verificar si es formato estándar SPARQL JSON
        if "results" in result and "bindings" in result["results"]:
            bindings = result["results"]["bindings"]
            print(f"🔍 DEBUG: Procesando {len(bindings)} resultados SPARQL")

            for binding in bindings:
                prop_data = {}

                # Extraer URI principal (puede ser annotationProperty, dataProperty, objectProperty, etc.)
                for key in [
                    "annotationProperty",
                    "dataProperty",
                    "objectProperty",
                    "property",
                ]:
                    if key in binding:
                        prop_data["uri"] = binding[key]["value"]
                        break

                # Extraer label si está disponible
                if "label" in binding and "value" in binding["label"]:
                    prop_data["label"] = binding["label"]["value"]
                else:
                    prop_data["label"] = ""

                # Extraer range si está disponible
                if "range" in binding and "value" in binding["range"]:
                    prop_data["range"] = binding["range"]["value"]
                else:
                    prop_data["range"] = ""

                # Extraer domain si está disponible
                if "domain" in binding and "value" in binding["domain"]:
                    prop_data["domain"] = binding["domain"]["value"]
                else:
                    prop_data["domain"] = ""

                # Solo añadir si tenemos una URI válida
                if "uri" in prop_data:
                    properties.append(prop_data)
                    print(f"🔍 DEBUG: Añadida propiedad: {prop_data['uri']}")

        print(f"📊 Extraídas {len(properties)} propiedades del resultado")
        return properties

    def _extract_usage_stats_from_result(
        self, result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extrae estadísticas de uso de la respuesta de la API.

        Args:
            result: Resultado de la API de Virtuoso

        Returns:
            Lista de diccionarios con estadísticas de uso
        """
        usage_stats = []

        # Para estadísticas, probablemente viene en literals
        for node_id, literals in result.get("literals", {}).items():
            stat_data = {"predicate": node_id}

            # Buscar usage_count y example_value en literals
            for prop, values in literals.items():
                if "usage_count" in prop.lower():
                    stat_data["usage_count"] = int(values[0]) if values else 0
                elif "example" in prop.lower():
                    stat_data["example_value"] = values[0] if values else ""

            if "usage_count" in stat_data:
                usage_stats.append(stat_data)

        # Ordenar por usage_count descendente
        usage_stats.sort(key=lambda x: x.get("usage_count", 0), reverse=True)

        return usage_stats


def test_virtuoso_connection():
    """Prueba la conexión con Virtuoso y muestra algunas propiedades."""
    print("🔍 Probando conexión con Virtuoso...")

    client = VirtuosoClient()

    try:
        # Probar consulta simple
        simple_query = "SELECT DISTINCT ?type WHERE { ?s a ?type } LIMIT 5"
        result = client.execute_query(simple_query)
        print(f"✅ Conexión exitosa. Resultado: {len(result.get('nodes', []))} nodos")

        # Obtener annotation properties
        print("\n📝 Obteniendo Annotation Properties...")
        ann_props = client.get_annotation_properties()
        print(f"Encontradas {len(ann_props)} annotation properties")
        for prop in ann_props[:5]:
            print(f"  • {prop['uri']} → {prop.get('label', 'Sin label')}")

        # Obtener datatype properties
        print("\n📊 Obteniendo Datatype Properties...")
        data_props = client.get_datatype_properties()
        print(f"Encontradas {len(data_props)} datatype properties")
        for prop in data_props[:5]:
            print(f"  • {prop['uri']} → {prop.get('label', 'Sin label')}")

        # Obtener estadísticas de uso
        print("\n📈 Obteniendo estadísticas de uso...")
        usage_stats = client.get_property_usage_stats()
        print(f"Encontradas {len(usage_stats)} propiedades con estadísticas")
        for stat in usage_stats[:5]:
            print(f"  • {stat['predicate']}: {stat.get('usage_count', 0)} usos")

        return client

    except Exception as e:
        print(f"❌ Error en conexión: {e}")
        return None


if __name__ == "__main__":
    test_virtuoso_connection()
