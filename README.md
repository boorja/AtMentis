# AtMentis – Intelligent Assistant for Knowledge Graphs

<div align="center">
  <img src="src/assets/addlogo.png" alt="AtMentis Logo" width="200" height="200"/>
</div> 

<div align="center">

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-lightgrey)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)

</div>

## 📚 Table of Contents

- [📋 Description](#-description)
- [🌟 Key Features](#-key-features)
- [🔧 System Requirements](#-system-requirements)
- [💻 Installation](#-installation)
- [🚀 System Usage](#-system-usage)
  - [🎯 Interactive Startup Screen](#-new-interactive-startup-screen)
  - [📊 Graph Selection from Virtuoso](#-option-1-view-available-graphs-in-virtuoso)
  - [📤 Ontology Upload](#-option-2-upload-new-ontology)
  - [🧠 Knowledge Graph Initialization](#-knowledge-graph-initialization-screen)
- [🔄 System Architecture](#-system-architecture)
- [📡 REST API](#-rest-api)
- [⚙️ Advanced Configuration](#-advanced-configuration)
- [📂 Project Structure](#-project-structure)
- [⚠️ Troubleshooting](#-troubleshooting)

## 📋 Description

**AtMentis** is a platform for intelligent exploration and querying of knowledge graphs. It offers interactive visualization of ontologies stored in Virtuoso servers, along with a conversational assistant that answers questions using language models and embedding systems.

The system combines natural language processing techniques, vector representation, automatic semantic enrichment, and graphical visualization to provide a complete ontological navigation experience with advanced contextual understanding.

## 🌟 Key Features

- **Interactive visualization** with Cosmograph
- **Contextualized conversational assistant** with Markdown formatting
- **Adaptive embeddings system** that automatically selects the optimal model (to be implemented)
- **Automatic semantic enrichment** with automatic predicate discovery
- **Hierarchical navigation** through classes and instances with depth analysis
- **Automatic SPARQL queries** optimized for Virtuoso
- **Adaptive embedding model strategy** based on content type and length (currently under review)
- **Intelligent caching system** with automatic expiration
- **Contextual exploration** based on current graph visibility
- **Complete REST API** for integration with external systems
- **Deep semantic analysis** with multiple specialized models

## 🔧 System Requirements

### Main Dependencies

- **Python 3.10+**
- **Flask 2.0+** and Flask-CORS for the web server
- **FastAPI 0.68+** for SPARQL query API
- **PyKEEN** for knowledge graph models
- **SentenceTransformers** with multilingual models
- **RDFLib** for ontology processing
- **SPARQLWrapper** for Virtuoso queries
- **D3.js** and **Cosmograph** for visualization
- **Virtuoso Server** with loaded ontology

### Supported Embedding Models

- **LaBSE**: High-quality multilingual for short texts
- **all-mpnet-base-v2**: Excellent general semantic understanding
- **all-MiniLM-L12-v2**: Efficient for long texts
- **Adaptive strategy**: Automatic selection based on content (under review)

## 💻 Installation

```bash
git clone https://github.com/your-username/Graph_Visualizer.git
cd Graph_Visualizer
pip install -r requirements.txt
npm install
```

### System Configuration

#### Main server configuration (`server.py`):

```python
# LLM model configuration
MODEL_URL = "http://your-llm-server:port/v1/chat/completions"
MODEL_NAME = "your-model-name"

# Virtuoso configuration
VIRTUOSO_CONFIG = {
    "endpoint": "http://your-virtuoso-server:8890/sparql",
    "database": "http://your-base-ontology/",
    "username": "your-username",
    "password": "your-password"
}
```

#### Startup screen configuration (`interactive-startup.js`):

```javascript
// Endpoint configuration for startup screen
const CONFIG = {
  BACKEND_URL: 'http://your-server:5000',        // Main Flask server
  VIRTUOSO_URL: 'http://your-server:32323',      // Virtuoso server
  STATE_KEY: 'atmentis_app_state',               // Persistent state key
  STATE_MAX_AGE: 7 * 24 * 60 * 60 * 1000,      // 7 days persistence
  VALID_EXTENSIONS: ['.owl', '.ttl', '.rdf', '.n3'] // Supported formats
};
```

#### Model configuration (`model_config.py`):

```python
# Configure embedding strategy
EMBEDDING_MODELS = {
    "default": "paraphrase-multilingual-mpnet-base-v2",
    "adaptive": "adaptive_strategy",  # Recommended
    "high_quality": "sentence-transformers/LaBSE"
}

# Configure knowledge graph model
KG_MODELS = {
    "default": {
        "name": "ComplEx",  # Current main model
        "embedding_dim": 200,
        "num_epochs": 1500
    }
}
```

## 🚀 System Usage

### Initialize Services

```bash
# Terminal 1: SPARQL query API (FastAPI)
python main.py

# Terminal 2: Main assistant server (Flask)
python server.py

# Terminal 3: Interactive visualization frontend
npm start
```

### Access and Startup Screen

Visit: `http://localhost:1234`

#### 🎯 New Interactive Startup Screen

The system presents an **interactive startup screen** that allows selecting ontologies from different sources:

<div align="center">
  <img src="docs/startup-screen.png" alt="Interactive startup screen" width="600"/>
</div>

**Interface elements:**
- **AtMentis central node**: Main system logo
- **Option nodes**: Two main options for loading ontologies

##### 📊 Option 1: View Available Graphs in Virtuoso

**Functionality:**
1. **Click on "View Graphs"** - Opens modal with available graphs on Virtuoso server
2. **Automatic listing** - Connects to Virtuoso and displays all available ontologies
3. **Detailed information** - Shows complete URI and number of triples per graph
4. **Conditional selection** - Detects if a graph was previously loaded to reuse embeddings

**Selection process:**
```
📊 View Graphs → Modal with list → Selection → State verification → Load
```

##### 📤 Option 2: Upload New Ontology

**Functionality:**
1. **Click on "Upload Ontology"** - Opens file upload modal
2. **Drag & Drop** - Drag files directly to the upload area
3. **File explorer** - Click to select file from system
4. **Automatic validation** - Verifies format before processing

**Supported formats:**
- `.owl` - Web Ontology Language
- `.ttl` - Turtle syntax  
- `.rdf` - RDF/XML format
- `.n3` - Notation3

**Upload process:**
```
📁 Select file → Validation → Upload → Processing → Temporary load
```

**Upload features:**
- **Progress bar**: Visual indicator of upload process
- **Pre-validation**: Verifies that the file is a valid ontology
- **Temporary load**: Uploaded files are marked as temporary
- **Automatic cleanup**: Automatically deleted when changing graphs or closing session
- **Real-time processing**: Shows number of processed triples

##### ⚡ Persistent State System

**Automatic session management:**
- **Saved state**: Remembers the last used ontology
- **Automatic restoration**: When reopening the app, restores previous state
- **Availability check**: Verifies that the graph is still available in Virtuoso

**System advantages:**
- ✅ **Resource optimization**: Avoids unnecessary retraining
- ✅ **Temporary management**: Automatically cleans temporary files
- ✅ **Smooth experience**: Transparent transition between sessions

##### 🧠 Knowledge Graph Initialization Screen

Once an ontology is selected, the system displays an **initialization screen** that monitors the entire training process:

<div align="center">
  <img src="docs/kg-initialization.png" alt="Knowledge Graph initialization" width="600"/>
</div>

**Screen elements:**
- **Progress bar**: Visual indicator of completion percentage (0-100%)
- **Current step**: Detailed description of ongoing operation
- **Technical details**: Model configuration (ComplEx 256D, LaBSE Multilingual)
- **Real-time logs**: Detailed record of all operations

#### Graph Interaction

- **Click on nodes:** Expands classes and shows subclasses
- **Zoom:** Mouse wheel to zoom in/out
- **Drag:** Move and reposition the graph
- **Control buttons:**
  - **Pause/Resume:** Physics simulation control
  - **Go back:** Returns to previous graph state
  - **Return to menu:** Returns to startup screen

#### 🏠 Navigation Between Ontologies

**"Return to Menu" button:**
- **Functionality**: Returns to startup screen without closing the application
- **Intelligent management**: 
  - If a temporary ontology is loaded, automatically cleans it
  - Preserves permanent Virtuoso ontologies
  - Allows switching between different graphs without restarting the server

**Navigation flow:**
```
Startup screen → Select ontology → Visualization → Return to menu → New selection
```

#### Using the Conversational Assistant

1. **Type your question** in natural language (Spanish or English)
2. **Send the query** by clicking "Send" or pressing Enter
3. **Receive contextualized response** in Markdown format

#### 🏷️ Interactive Tag System

The assistant includes a tag system that appears automatically in responses to facilitate navigation and graph expansion:

| Tag | Status | Function | Description |
|-----|--------|----------|-------------|
| **@Browse** | ✅ **Functional** | Smart graph expansion | Expands only the most relevant entities based on LLM analysis with confidence threshold |
| **@Select** | 🚧 **In development** | Select node | Automatically selects and centers a specific node in the visualization |
| **@Create** | 🚧 **In development** | Create new node | Allows creating new nodes or relationships in the graph |

#### 🔍 @Browse Functionality - Smart Expansion

**@Browse** uses an advanced semantic analysis system to expand only the most relevant entities:

**Threshold system advantages:**
- ✅ **Performance**: Maintains visualization fluidity
- ✅ **Context**: Expands only semantically coherent entities

**Examples of @Browse system usage:**

The @Browse system works by adding the tag in **your question**, not in the assistant's response.

**Example 1: Basic query with automatic expansion**
```
👤 User: "What types of vehicles exist? @Browse"

🤖 Assistant: The main types of vehicles include:
- Motor vehicles: cars, motorcycles, trucks, buses
- Non-motor vehicles: bicycles, scooters, animal-drawn vehicles
- Watercraft: boats, submarines, kayaks
- Aircraft: airplanes, helicopters, hot air balloons
```
**Result:** Vehicle-related nodes are automatically expanded in the visualization.

**Example 2: Specific query with intelligent analysis**
```
👤 User: "Explain network protocols @Browse"

🤖 Assistant: Network protocols define communication rules:
- HTTP/HTTPS for web transfer
- TCP/UDP for data transport
- IP for routing between networks
- DNS for name resolution
```
**Result:** Only the most relevant protocols appear in the graph according to LLM analysis.

### 🧠 Advanced Query Processing System

When a user asks a question, the system executes an analysis and response process:

#### 1. **Initial Reception and Analysis**
   - Receives user query (e.g., "what types of [entity] are there?")
   - Identifies current visual context (nodes and links displayed in the interface)

#### 2. **Automated Semantic Enrichment**
   - **Annotation analysis**: Automatically extracts labels, descriptions, and metadata from the ontology
   - **Vocabulary detection**: Identifies present predicates

#### 3. **Adaptive Embedding Strategy** (under review for implementation)
   - **Content analysis**: Classifies text by length and technical complexity
   - **Model selection**: Automatically chooses the optimal embedding model:
     - **LaBSE**: For labels and short texts (≤100 characters)
     - **all-mpnet-base-v2**: For medium queries and general understanding
     - **all-MiniLM-L12-v2**: For long descriptions and extensive context
   - **Vector calculation**: Generates specialized semantic representations

#### 4. **Intelligent Scoring System**
   - **Visibility bonus**: For entities visible in the current graph
   - **Semantic similarity**: Scoring based on cosine distance of embeddings
   - **Exact matches**: Maximum score for direct matches

#### 5. **Specific Context Construction**
   - Selects the highest-scoring entities as the response core
   - Extracts RDF triples related to these key entities
   - Includes hierarchical relationships, properties, and relevant metadata in the final prompt

#### 6. **Response Generation with Adaptive Reasoning (Deep Thinking)**

The system abandons the single-query approach and adopts a multi-step adaptive reasoning process to maximize response accuracy and relevance, based exclusively on ontology knowledge.

*   **Step 1: Query Intent Analysis**
    *   First, the system classifies the user's question intent to determine its nature.

*   **Step 2: Adaptive Reasoning Strategy Selection**
    *   Based on intent, the most efficient strategy is chosen:
        *   **Direct Response (1 LLM call):** For simple questions and definitions.
        *   **Structured Analysis (2 LLM calls):** For queries requiring exploration of relationships, hierarchies, or properties.
        *   **Comparative Analysis (3 LLM calls):** For comparing two or more entities in detail.

*   **Step 3: Multi-Step Reasoning Process (Chain-of-Thought)**
    *   Once the strategy is selected, the system executes a guided thought chain:
        *   If the strategy is **Direct Response**, a single LLM call is made with a detailed prompt instructing it to respond concisely and directly, strictly based on context.
        *   If the strategy is **Structured Analysis**, the process is divided into two roles:
            1.  **Analyst Role:** In the first call, the LLM extracts relevant facts and relationships from the graph in a technical, structured format (JSON), without yet attempting to answer the user.
            2.  **Communicator Role:** In the second call, the LLM receives its own technical analysis and uses it as a basis to synthesize and write a coherent final response in natural language.
        *   If the strategy is **Comparative Analysis**, reasoning extends to three steps:
            1.  **Entity A Analysis:** The LLM performs structured analysis only on the first entity.
            2.  **Entity B Analysis:** The process repeats, performing structured analysis only on the second entity.
            3.  **Comparator Role:** In the final call, the LLM receives both analyses and has the sole task of comparing them to generate a response highlighting similarities and differences.

*   **Step 4: Enriched Context and Final Response**
    *   The final response is built exclusively from facts verified in the ontology during the reasoning process, ensuring the model doesn't fabricate information.

## 🔄 System Architecture

### Main Components

- **`server.py`**: Main Flask server with conversational assistant
- **`main.py`**: FastAPI API for SPARQL queries and RDF processing
- **`kg_embedding.py`**: Embedding engine with adaptive strategies
- **`model_config.py`**: Centralized configuration for all models
- **`annotation_enrichment.py`**: Automatic semantic enrichment system
- **`adaptive_embedding_strategy.py`**: Intelligent model selection strategy
- **`virtuoso_client.py`**: Specialized client for Virtuoso communication
- **`index.js`**: Visualization frontend with Cosmograph
- **`sparql.js`**: Advanced SPARQL query handler
- **`interactive-startup.js`**: Interactive startup screen system

### ⚡ Detailed Server Initialization Process

When running `python server.py`, the system performs a **lightweight initialization** and waits for ontology selection:

#### **Phase 1: Server Startup (immediate)**
1. **Flask initialization**: Configures routes
2. **Endpoint configuration**: `/chat`, `/reset`, `/clear_cache`, `/select-graph`
3. **Cache verification**: Checks if valid previous cache exists

#### **Phase 2: Ontology Selection (user)**
- **User navigates** to interactive startup screen
- **Selects ontology** from Virtuoso or uploads new ontology
- **System receives** `/select-graph` request with graph URI
- **Starts automatic processing**

#### **Phase 3: Automatic Knowledge Graph Processing**

Once the ontology is selected, the system executes complete initialization:

##### 1. **Intelligent Cache Management**
   - Verifies existing cache for the specific selected graph
   - Checks timestamps to detect ontology changes
   - Validates integrity of stored models and embeddings
   - Decides whether to reuse cache or regenerate from scratch

##### 2. **Ontological Extraction and Analysis**
   - Connects to Virtuoso server with selected graph
   - Extracts complete class structure and hierarchies
   - Automatically analyzes present annotations (automatic predicate discovery)
   - Generates multilingual mapping between equivalent terms

##### 3. **Knowledge Graph Model Training**
   - **Selects ComplEx** as main model
   - **PyKEEN format conversion**: Transforms RDF triples to tensors (matrices)
   - **Representation learning**:
     - Converts entities and relationships to numerical vectors
     - Captures patterns through complex representations (complex numbers)
     - Optimizes representations to preserve asymmetric semantic relationships
   - **Iterative training**:
     - Processes data in batches of 512 examples (configurable)
     - Runs 1500 training epochs (adjustable in `model_config.py`)
     - Applies regularization to avoid overfitting
   - **Quality evaluation**: Measures link prediction accuracy

##### 4. **Adaptive Embedding Generation** (under review)
   - **Adaptive system loading**: Initializes multiple specialized models
   - **Ontological content analysis**:
     - Classifies entities by length and complexity
     - Detects technical vs. descriptive content
     - Identifies predominant annotation language
   - **Specialized vector generation**:
     - **LaBSE**: For short labels and multilingual terms
     - **all-mpnet-base-v2**: For medium-length descriptions
     - **all-MiniLM-L12-v2**: For long texts and extensive contexts

##### 5. **Knowledge System Construction**
   - **Semantic indexing**: Creates inverted indexes for fast search
   - **Term mapping**: Builds automatic Spanish↔English dictionaries
   - **Class hierarchies**: Recursively analyzes `rdfs:subClassOf` relationships
   - **Synonym system**: Automatically detects equivalent terms

##### 6. **Persistence and Optimization**
   - **Cache storage**: Saves all artifacts in `.cache/`
   - **Integrity verification**: Checksums to validate data
   - **Detailed logs**: Record of entire initialization process

##### 7. **Finalization**
   - **KG system activated**: Knowledge Graph embeddings ready
   - **Assistant enabled**: `/chat` endpoint operational
   - **Visualization prepared**: Frontend can query graph data

## 📡 REST API

### Main Server Endpoints (Flask - Port 5000)

| Endpoint       | Method | Description                         | Parameters |
|----------------|--------|-------------------------------------|------------|
| `/chat`        | POST   | Send question to assistant          | `message`, `graph_data` |
| `/reset`       | POST   | Reset conversation                  | None |
| `/clear_cache` | POST   | Clear system cache                  | None |
| `/select-graph` | POST   | Select graph to initialize          | `graph_uri`, `is_temporary` |
| `/initialize-progress` | GET | Get initialization progress       | None |
| `/upload-ontology` | POST | Upload ontology file               | `ontology` (FormData) |
| `/cleanup-ontology` | POST | Clean up temporary ontology        | `graph_uri` |

### SPARQL Query Endpoints (FastAPI - Port 32323)

| Endpoint          | Method | Description                      | Parameters |
|-------------------|--------|----------------------------------|------------|
| `/query_rdf`      | POST   | Query local RDF file             | `file_path`, `sparql_query` |
| `/query_virtuoso` | POST   | Query Virtuoso server            | `virtuoso_endpoint`, `virtuoso_database`, `virtuoso_username`, `virtuoso_password`, `query` |
| `/available-graphs` | GET  | List available graphs in Virtuoso | None |
| `/select-graph`   | POST   | Select specific graph (to be migrated) | `graph_uri` |
| `/upload-ontology` | POST  | Upload ontology file (to be migrated) | `ontology` (FormData) |
| `/cleanup-ontology` | POST | Clean up temporary ontology (to be migrated) | `graph_uri` |

## ⚙️ Advanced Configuration

### Multi-layer Intelligent Cache System

The system uses a sophisticated cache located in `.cache/` with specialized components:

- **`ontology_structure.pkl`**: Hierarchical class structure and metadata
- **`all_triples.pkl`**: Complete set of RDF triples
- **`kg_model_*.pkl`**: Trained knowledge graph models
- **`embeddings_*.pkl`**: Semantic vectors by strategy
- **`annotations_*.pkl`**: Enriched annotation system
- **Automatic expiration**: 12 hours by default (configurable)

**Manual cache cleanup:**

```bash
# Complete cleanup
curl -X POST http://localhost:5000/clear_cache

# Or delete directly
rm -rf .cache/
```

### Performance Optimization

#### Threshold and limit configuration:

```python
# In kg_embedding.py
SIMILARITY_THRESHOLD = 0.7      # Minimum similarity threshold
MAX_ENTITIES_PER_QUERY = 50     # Maximum entities per query
BATCH_SIZE_EMBEDDINGS = 32      # Batch size for embedding calculation
CACHE_EXPIRATION_HOURS = 12     # Cache expiration
```

## ⚠️ Troubleshooting

### Common Errors and Solutions

#### 🔴 **"Knowledge Graph embeddings not initialized"**
```bash
# Solution: Clear cache and restart
curl -X POST http://localhost:5000/clear_cache
rm -rf .cache/
python server.py
```

