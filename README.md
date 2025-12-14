# Graph Visualizer – Asistente Inteligente para Grafos de Conocimiento

<div align="center">
  <img src="src/assets/addlogo.png" alt="Graph Visualizer Logo" width="200" height="200"/>
</div> 

<div align="center">

![Versión](https://img.shields.io/badge/versión-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)

</div>

## 📚 Tabla de contenido

- [📋 Descripción](#-descripción)
- [🌟 Funcionalidades destacadas](#-funcionalidades-destacadas)
- [🔧 Requisitos del sistema](#-requisitos-del-sistema)
- [💻 Instalación](#-instalación)
- [🚀 Uso del sistema](#-uso-del-sistema)
  - [🎯 Pantalla de inicio interactiva](#-nueva-pantalla-de-inicio-interactiva)
  - [📊 Selección de grafos desde Virtuoso](#-opción-1-ver-grafos-disponibles-en-virtuoso)
  - [📤 Subida de ontologías](#-opción-2-subir-nueva-ontología)
  - [🧠 Inicialización del Knowledge Graph](#-pantalla-de-inicialización-del-knowledge-graph)
- [🔄 Arquitectura del sistema](#-arquitectura-del-sistema)
- [📡 API REST](#-api-rest)
- [⚙️ Configuración avanzada](#-configuración-avanzada)
- [📂 Estructura del proyecto](#-estructura-del-proyecto)
- [⚠️ Solución de problemas](#-solución-de-problemas)

## 📋 Descripción

**Graph Visualizer** es una plataforma para la exploración y consulta inteligente de grafos de conocimiento. Ofrece una visualización interactiva de ontologías almacenadas en servidores Virtuoso, junto con un asistente conversacional que responde preguntas utilizando modelos de lenguaje y sistemas de embeddings.

El sistema combina técnicas de procesamiento de lenguaje natural, representación vectorial, enriquecimiento semántico automático y visualización gráfica para brindar una experiencia completa de navegación ontológica con comprensión contextual avanzada.

## 🌟 Funcionalidades destacadas

- **Visualización interactiva** con Cosmograph
- **Asistente conversacional contextualizado** con formato Markdown
- **Sistema de embeddings adaptativos** que selecciona automáticamente el modelo óptimo (por implementar)
- **Enriquecimiento semántico automático** con descubrimiento automático de predicados
- **Navegación jerárquica** por clases e instancias con análisis de profundidad
- **Consultas SPARQL automáticas** optimizadas para Virtuoso
- **Estrategia adaptativa de modelos de embedding** según tipo y longitud del contenido (actualmente en revisión)
- **Sistema de caché inteligente** con expiración automática 
- **Exploración contextual** basada en visibilidad del grafo actual
- **API REST completa** para integración con sistemas externos
- **Análisis semántico profundo** con múltiples modelos especializados

## 🔧 Requisitos del sistema

### Dependencias principales

- **Python 3.10+**
- **Flask 2.0+** y Flask-CORS para el servidor web
- **FastAPI 0.68+** para la API de consultas SPARQL
- **PyKEEN** para modelos de grafos de conocimiento
- **SentenceTransformers** con modelos multilingües
- **RDFLib** para procesamiento de ontologías
- **SPARQLWrapper** para consultas a Virtuoso
- **D3.js** y **Cosmograph** para visualización
- **Servidor Virtuoso** con ontología cargada

### Modelos de embeddings soportados

- **LaBSE**: Multilingüe de alta calidad para textos cortos
- **all-mpnet-base-v2**: Excelente comprensión semántica general
- **all-MiniLM-L12-v2**: Eficiente para textos largos
- **Estrategia adaptativa**: Selección automática según contenido (en revisión)

## 💻 Instalación

```bash
git clone https://github.com/tu-usuario/Graph_Visualizer.git
cd Graph_Visualizer
pip install -r requirements.txt
npm install
```

### Configuración del sistema

#### Configuración del servidor principal (`server.py`):

```python
# Configuración del modelo LLM
MODEL_URL = "http://tu-servidor-llm:puerto/v1/chat/completions"
MODEL_NAME = "nombre-de-tu-modelo"

# Configuración de Virtuoso
VIRTUOSO_CONFIG = {
    "endpoint": "http://tu-servidor-virtuoso:8890/sparql",
    "database": "http://tu-ontologia-base/",
    "username": "tu-usuario",
    "password": "tu-contraseña"
}
```

#### Configuración de la pantalla de inicio (`interactive-startup.js`):

```javascript
// Configuración de endpoints para la pantalla de inicio
const CONFIG = {
  BACKEND_URL: 'http://tu-servidor:5000',        // Servidor Flask principal
  VIRTUOSO_URL: 'http://tu-servidor:32323',      // Servidor Virtuoso
  STATE_KEY: 'atmentis_app_state',               // Clave para estado persistente
  STATE_MAX_AGE: 7 * 24 * 60 * 60 * 1000,      // 7 días de persistencia
  VALID_EXTENSIONS: ['.owl', '.ttl', '.rdf', '.n3'] // Formatos soportados
};
```

#### Configuración de modelos (`model_config.py`):

```python
# Configurar estrategia de embeddings
EMBEDDING_MODELS = {
    "default": "paraphrase-multilingual-mpnet-base-v2",
    "adaptive": "adaptive_strategy",  # Recomendado
    "high_quality": "sentence-transformers/LaBSE"
}

# Configurar modelo de grafo de conocimiento
KG_MODELS = {
    "default": {
        "name": "ComplEx",  # Modelo principal actual
        "embedding_dim": 200,
        "num_epochs": 1500
    }
}
```

## 🚀 Uso del sistema

### Inicializar los servicios

```bash
# Terminal 1: API de consultas SPARQL (FastAPI)
python main.py

# Terminal 2: Servidor principal del asistente (Flask)
python server.py

# Terminal 3: Frontend de visualización interactiva
npm start
```

### Acceso y pantalla de inicio

Visita: `http://localhost:1234`

#### 🎯 Nueva Pantalla de Inicio Interactiva

El sistema presenta una **pantalla de inicio interactivA** que permite seleccionar ontologías de diferentes fuentes:

<div align="center">
  <img src="docs/startup-screen.png" alt="Pantalla de inicio interactiva" width="600"/>
</div>

**Elementos de la interfaz:**
- **Nodo central AtMentis**: Logo principal del sistema
- **Nodos de opción**: Dos opciones principales para cargar ontologías

##### 📊 Opción 1: Ver Grafos Disponibles en Virtuoso

**Funcionalidad:**
1. **Clic en "Ver Grafos"** - Abre modal con grafos disponibles en el servidor Virtuoso
2. **Listado automático** - Conecta con Virtuoso y muestra todas las ontologías disponibles
3. **Información detallada** - Muestra URI completa y número de tripletas por grafo
4. **Selección condicionada** - Detecta si ya hay un grafo cargado previamente, para reutilizar embeddings

**Proceso de selección:**
```
📊 Ver Grafos → Modal con lista → Selección → Verificación de estado → Carga
```

##### 📤 Opción 2: Subir Nueva Ontología

**Funcionalidad:**
1. **Clic en "Subir Ontología"** - Abre modal de carga de archivos
2. **Drag & Drop** - Arrastra archivos directamente al área de carga
3. **Explorador de archivos** - Clic para seleccionar archivo del sistema
4. **Validación automática** - Verifica formato antes de procesar

**Formatos soportados:**
- `.owl` - Web Ontology Language
- `.ttl` - Turtle syntax  
- `.rdf` - RDF/XML format
- `.n3` - Notation3

**Proceso de carga:**
```
📁 Seleccionar archivo → Validación → Subida → Procesamiento → Carga temporal
```

**Características de la subida:**
- **Barra de progreso**: Indicador visual del proceso de carga
- **Validación previa**: Verifica que el archivo sea una ontología válida
- **Carga temporal**: Los archivos subidos se marcan como temporales
- **Limpieza automática**: Se eliminan automáticamente al cambiar de grafo o cerrar sesión
- **Procesamiento en tiempo real**: Muestra número de tripletas procesadas

##### ⚡ Sistema de Estado Persistente

**Gestión automática de sesiones:**
- **Estado guardado**: Recuerda la última ontología utilizada
- **Restauración automática**: Al reabrir la aplicación, restaura el estado anterior
- **Verificación de disponibilidad**: Comprueba que el grafo siga disponible en Virtuoso

**Ventajas del sistema:**
- ✅ **Optimización de recursos**: Evita reentrenamientos innecesarios
- ✅ **Gestión temporal**: Limpia automáticamente archivos temporales
- ✅ **Experiencia fluida**: Transición transparente entre sesiones

##### 🧠 Pantalla de Inicialización del Knowledge Graph

Una vez seleccionada una ontología, el sistema muestra una **pantalla de inicialización** que monitorea todo el proceso de entrenamiento:

<div align="center">
  <img src="docs/kg-initialization.png" alt="Inicialización del Knowledge Graph" width="600"/>
</div>

**Elementos de la pantalla:**
- **Barra de progreso**: Indicador visual del porcentaje completado (0-100%)
- **Paso actual**: Descripción detallada de la operación en curso
- **Detalles técnicos**: Configuración del modelo (ComplEx 256D, LaBSE Multilingüe)
- **Logs en tiempo real**: Registro detallado de todas las operaciones

#### Interacción con el grafo

- **Clic en nodos:** Expande clases y muestra subclases
- **Zoom:** Rueda del ratón para acercar/alejar
- **Arrastrar:** Mueve y reposiciona el grafo
- **Botones de control:**
  - **Pausar/Reanudar:** Control de la simulación física
  - **Retroceder:** Regresa al estado anterior del grafo
  - **Volver al menú:** Regresa a la pantalla de inicio

#### 🏠 Navegación entre ontologías

**Botón "Volver al Menú":**
- **Funcionalidad**: Regresa a la pantalla de inicio sin cerrar la aplicación
- **Gestión inteligente**: 
  - Si hay una ontología temporal cargada, la limpia automáticamente
  - Preserva ontologías permanentes de Virtuoso
  - Permite cambiar entre diferentes grafos sin reiniciar el servidor

**Flujo de navegación:**
```
Pantalla inicio → Seleccionar ontología → Visualización → Volver al menú → Nueva selección
```

#### Uso del asistente conversacional

1. **Escribe tu pregunta** en lenguaje natural (español o inglés)
2. **Envía la consulta** con clic en "Enviar" o presiona Enter
3. **Recibe respuesta** contextualizada en formato Markdown

#### 🏷️ Sistema de etiquetas interactivas

El asistente incluye un sistema de etiquetas que aparecen automáticamente en las respuestas para facilitar la navegación y expansión del grafo:

| Etiqueta | Estado | Función | Descripción |
|----------|--------|---------|-------------|
| **@Browse** | ✅ **Funcional** | Expandir grafo inteligente | Expande solo las entidades más relevantes basándose en análisis LLM con threshold de confianza |
| **@Select** | 🚧 **En desarrollo** | Seleccionar nodo | Selecciona y centra automáticamente un nodo específico en la visualización |
| **@Create** | 🚧 **En desarrollo** | Crear nuevo nodo | Permite crear nuevos nodos o relaciones en el grafo |

#### 🔍 Funcionalidad @Browse - Expansión inteligente

**@Browse** utiliza un sistema avanzado de análisis semántico para expandir únicamente las entidades más relevantes:

**Ventajas del sistema de threshold:**
- ✅ **Rendimiento**: Mantiene la fluidez de la visualización
- ✅ **Contexto**: Expande solo entidades semánticamente coherentes

**Ejemplos de uso del sistema @Browse:**

El sistema @Browse funciona agregando la etiqueta en **tu pregunta**, no en la respuesta del asistente.

**Ejemplo 1: Consulta básica con expansión automática**
```
👤 Usuario: "¿Qué tipos de vehículos existen? @Browse"

🤖 Asistente: Los tipos principales de vehículos incluyen:
- Vehículos de motor: automóviles, motocicletas, camiones, autobuses
- Vehículos sin motor: bicicletas, patinetes, vehículos de tracción animal
- Vehículos acuáticos: barcos, submarinos, kayaks
- Vehículos aéreos: aviones, helicópteros, globos aerostáticos
```
**Resultado:** Los nodos relacionados con vehículos se expanden automáticamente en la visualización.

**Ejemplo 2: Consulta específica con análisis inteligente**
```
👤 Usuario: "Explícame sobre protocolos de red @Browse"

🤖 Asistente: Los protocolos de red definen las reglas de comunicación:
- HTTP/HTTPS para transferencia web
- TCP/UDP para transporte de datos
- IP para enrutamiento entre redes
- DNS para resolución de nombres
```
**Resultado:** Solo los protocolos más relevantes aparecen en el grafo según el análisis del LLM.

### 🧠 Sistema avanzado de procesamiento de consultas

Cuando un usuario hace una pregunta, el sistema ejecuta un proceso de análisis y respuesta:

#### 1. **Recepción y análisis inicial**
   - Recibe la consulta del usuario (ej: "¿qué tipos de [entidad] hay?")
   - Identifica el contexto visual actual (nodos y enlaces mostrados en la interfaz)

#### 2. **Enriquecimiento semántico automatizado**
   - **Análisis de anotaciones**: Extrae automáticamente labels, descripciones y metadatos de la ontología
   - **Detección de vocabularios**: Identifica los predicados presentes

#### 3. **Estrategia adaptativa de embeddings** (en revisión para implementación)
   - **Análisis de contenido**: Clasifica el texto por longitud y complejidad técnica
   - **Selección de modelo**: Elige automáticamente el modelo de embedding óptimo:
     - **LaBSE**: Para labels y textos cortos (≤100 caracteres)
     - **all-mpnet-base-v2**: Para consultas medias y comprensión general
     - **all-MiniLM-L12-v2**: Para descripciones largas y contexto extenso
   - **Cálculo vectorial**: Genera representaciones semánticas especializadas

#### 4. **Sistema de puntuación inteligente**
   - **Bonificación por visibilidad**: para entidades visibles en el grafo actual
   - **Similitud semántica**: Scoring basado en distancia coseno de embeddings
   - **Coincidencias exactas**: Máxima puntuación para matches directos

#### 5. **Construcción del contexto específico**
   - Selecciona las entidades mejor puntuadas como núcleo de la respuesta
   - Extrae tripletas RDF relacionadas con estas entidades clave
   - Incluye relaciones jerárquicas, propiedades y metadatos relevantes en el prompt final

#### 6. **Generación de Respuesta con Razonamiento Adaptativo (Deep Thinking)**

El sistema abandona el enfoque de una sola consulta y adopta un proceso de razonamiento adaptativo en múltiples pasos para maximizar la precisión y relevancia de la respuesta, basándose exclusivamente en el conocimiento de la ontología.

*   **Paso 1: Análisis de Intención de la Consulta**
    *   Primero, el sistema clasifica la intención de la pregunta del usuario para determinar su naturaleza.

*   **Paso 2: Selección de Estrategia de Razonamiento Adaptativo**
    *   Basado en la intención, se elige la estrategia más eficiente:
        *   **Respuesta Directa (1 llamada al LLM):** Para preguntas simples y definiciones.
        *   **Análisis Estructurado (2 llamadas al LLM):** Para consultas que requieren explorar relaciones, jerarquías o propiedades.
        *   **Análisis Comparativo (3 llamadas al LLM):** Para comparar dos o más entidades de forma detallada.

*   **Paso 3: Proceso de Razonamiento en Múltiples Pasos (Chain-of-Thought)**
    *   Una vez seleccionada la estrategia, el sistema ejecuta una cadena de pensamiento guiada:
        *   Si la estrategia es **Respuesta Directa**, se realiza una única llamada al LLM con un prompt detallado que le instruye a responder de forma concisa y directa, basándose estrictamente en el contexto.
        *   Si la estrategia es **Análisis Estructurado**, el proceso se divide en dos roles:
            1.  **Rol de Analista:** En la primera llamada, el LLM extrae los hechos y relaciones relevantes del grafo en un formato técnico y estructurado (JSON), sin intentar aún dar una respuesta al usuario.
            2.  **Rol de Comunicador:** En la segunda llamada, el LLM recibe su propio análisis técnico y lo utiliza como base para sintetizar y redactar una respuesta final coherente y en lenguaje natural.
        *   Si la estrategia es **Análisis Comparativo**, el razonamiento se extiende a tres pasos:
            1.  **Análisis de Entidad A:** El LLM realiza un análisis estructurado solo de la primera entidad.
            2.  **Análisis de Entidad B:** Se repite el proceso, realizando un análisis estructurado solo de la segunda entidad.
            3.  **Rol de Comparador:** En la llamada final, el LLM recibe ambos análisis y tiene la única tarea de compararlos para generar una respuesta que resalte similitudes y diferencias.

*   **Paso 4: Contexto Enriquecido y Respuesta Final**
    *   La respuesta final se construye exclusivamente a partir de los hechos verificados en la ontología durante el proceso de razonamiento, garantizando que el modelo no invente información.

## 🔄 Arquitectura del sistema

### Componentes principales

- **`server.py`**: Servidor principal Flask con asistente conversacional
- **`main.py`**: API FastAPI para consultas SPARQL y procesamiento RDF
- **`kg_embedding.py`**: Motor de embeddings con estrategias adaptativas
- **`model_config.py`**: Configuración centralizada de todos los modelos
- **`annotation_enrichment.py`**: Sistema de enriquecimiento semántico automático
- **`adaptive_embedding_strategy.py`**: Estrategia de selección inteligente de modelos
- **`virtuoso_client.py`**: Cliente especializado para comunicación con Virtuoso
- **`index.js`**: Frontend de visualización con Cosmograph
- **`sparql.js`**: Manejador avanzado de consultas SPARQL
- **`interactive-startup.js`**: Sistema de pantalla de inicio interactiva

### ⚡ Proceso detallado de inicialización del servidor

Al ejecutar `python server.py`, el sistema realiza una **inicialización ligera** y queda en espera de selección de ontología:

#### **Fase 1: Arranque del servidor (inmediato)**
1. **Inicialización de Flask**: Configura rutas
2. **Configuración de endpoints**: `/chat`, `/reset`, `/clear_cache`, `/select-graph`
3. **Verificación de caché**: Comprueba si existe caché previo válido

#### **Fase 2: Selección de ontología (usuario)**
- **Usuario navega** a la pantalla de inicio interactiva
- **Selecciona ontología** desde Virtuoso o sube nueva ontología
- **Sistema recibe** petición `/select-graph` con URI del grafo
- **Inicia procesamiento** automático

#### **Fase 3: Procesamiento automático de Knowledge Graph**

Una vez seleccionada la ontología, el sistema ejecuta la inicialización completa:

##### 1. **Gestión inteligente de caché**
   - Verifica caché existente para el grafo específico seleccionado
   - Comprueba timestamps para detectar cambios en la ontología
   - Valida integridad de modelos y embeddings almacenados
   - Decide si reutilizar caché o regenerar desde cero

##### 2. **Extracción y análisis ontológico**
   - Se conecta al servidor Virtuoso con el grafo seleccionado
   - Extrae la estructura completa de clases y jerarquías
   - Analiza automáticamente las anotaciones presentes (descubrimiento de predicado automático)
   - Genera mapeo multilingüe entre términos equivalentes

##### 3. **Entrenamiento del modelo de grafos de conocimiento**
   - **Selecciona ComplEx** como modelo principal
   - **Conversión a formato PyKEEN**: Transforma tripletas RDF a tensores (matrices)
   - **Aprendizaje de representaciones**:
     - Convierte entidades y relaciones en vectores numéricos
     - Captura patrones mediante representaciones complejas (números complejos)
     - Optimiza representaciones para preservar relaciones semánticas asimétricas
   - **Entrenamiento iterativo**:
     - Procesa datos en lotes de 512 ejemplos (configurable)
     - Ejecuta 1500 épocas de entrenamiento (ajustable en `model_config.py`)
     - Aplica regularización para evitar sobreajuste
   - **Evaluación de calidad**: Mide precisión en predicción de enlaces

##### 4. **Generación de embeddings adaptativos** (en revisión)
   - **Carga del sistema adaptativo**: Inicializa múltiples modelos especializados
   - **Análisis de contenido ontológico**:
     - Clasifica entidades por longitud y complejidad
     - Detecta contenido técnico vs. descriptivo
     - Identifica idioma predominante de las anotaciones
   - **Generación vectorial especializada**:
     - **LaBSE**: Para labels cortos y términos multilingües
     - **all-mpnet-base-v2**: Para descripciones de longitud media
     - **all-MiniLM-L12-v2**: Para textos largos y contextos extensos

##### 5. **Construcción del sistema de conocimiento**
   - **Indexación semántica**: Crea índices invertidos para búsqueda rápida
   - **Mapeo de términos**: Construye diccionarios español↔inglés automáticos
   - **Jerarquías de clases**: Analiza relaciones `rdfs:subClassOf` recursivamente
   - **Sistema de sinónimos**: Detecta términos equivalentes automáticamente

##### 6. **Persistencia y optimización**
   - **Almacenamiento en caché**: Guarda todos los artefactos en `.cache/`
   - **Verificación de integridad**: Checksums para validar datos
   - **Logs detallados**: Registro de todo el proceso de inicialización

##### 7. **Finalización**
   - **Sistema KG activado**: Knowledge Graph embeddings listos
   - **Asistente habilitado**: Endpoint `/chat` operativo
   - **Visualización preparada**: Frontend puede consultar datos del grafo

## 📡 API REST

### Endpoints del servidor principal (Flask - Puerto 5000)

| Endpoint       | Método | Descripción                         | Parámetros |
|----------------|--------|-------------------------------------|------------|
| `/chat`        | POST   | Enviar pregunta al asistente        | `message`, `graph_data` |
| `/reset`       | POST   | Reiniciar conversación              | Ninguno |
| `/clear_cache` | POST   | Limpiar caché del sistema           | Ninguno |
| `/select-graph` | POST   | Seleccionar grafo para inicializar  | `graph_uri`, `is_temporary` |
| `/initialize-progress` | GET | Obtener progreso de inicialización | Ninguno |
| `/upload-ontology` | POST | Subir archivo de ontología       | `ontology` (FormData) |
| `/cleanup-ontology` | POST | Limpiar ontología temporal       | `graph_uri` |

### Endpoints de consultas SPARQL (FastAPI - Puerto 32323)

| Endpoint          | Método | Descripción                      | Parámetros |
|-------------------|--------|----------------------------------|------------|
| `/query_rdf`      | POST   | Consultar archivo RDF local      | `file_path`, `sparql_query` |
| `/query_virtuoso` | POST   | Consultar servidor Virtuoso      | `virtuoso_endpoint`, `virtuoso_database`, `virtuoso_username`, `virtuoso_password`, `query` |
| `/available-graphs` | GET  | Listar grafos disponibles en Virtuoso | Ninguno |
| `/select-graph`   | POST   | Seleccionar grafo específico (por migrar)     | `graph_uri` |
| `/upload-ontology` | POST  | Subir archivo de ontología (por migrar)       | `ontology` (FormData) |
| `/cleanup-ontology` | POST | Limpiar ontología temporal (por migrar)       | `graph_uri` |

## ⚙️ Configuración avanzada

### Sistema de caché inteligente multicapa

El sistema utiliza un caché sofisticado ubicado en `.cache/` con componentes especializados:

- **`ontology_structure.pkl`**: Estructura jerárquica de clases y metadatos
- **`all_triples.pkl`**: Conjunto completo de tripletas RDF
- **`kg_model_*.pkl`**: Modelos de grafos de conocimiento entrenados
- **`embeddings_*.pkl`**: Vectores semánticos por estrategia
- **`annotations_*.pkl`**: Sistema de anotaciones enriquecidas
- **Expiración automática**: 12 horas por defecto (configurable)

**Limpieza manual del caché:**

```bash
# Limpiar completamente
curl -X POST http://localhost:5000/clear_cache

# O eliminar directamente
rm -rf .cache/
```

### Optimización de rendimiento

#### Configuración de umbrales y límites:

```python
# En kg_embedding.py
SIMILARITY_THRESHOLD = 0.7      # Umbral mínimo de similitud
MAX_ENTITIES_PER_QUERY = 50     # Entidades máximas por consulta
BATCH_SIZE_EMBEDDINGS = 32      # Lote para cálculo de embeddings
CACHE_EXPIRATION_HOURS = 12     # Expiración de caché
```

## ⚠️ Solución de problemas

### Errores comunes y soluciones

#### 🔴 **"Knowledge Graph embeddings not initialized"**
```bash
# Solución: Limpiar caché y reiniciar
curl -X POST http://localhost:5000/clear_cache
rm -rf .cache/
python server.py
```

