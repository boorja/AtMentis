// ====================================
//          IMPORTS
// ====================================
import "./styles.css";
import { fetchGraphData, fetchVirtuoso, fetchClassInstances, fetchSubclasses, virtuosoConfig, fetchInitialGraph, fetchInstanceDataProperties, fetchInstanceObjectAssertions, isInstanceNode, fetchNodeAnnotations, updateGraphUri } from './sparql.js';
import { Graph, GraphConfigInterface } from "@cosmograph/cosmos";
import { drawColorBar } from './colorbar.js';
import * as d3 from 'd3';
import { CosmosLabels } from "./labels"; // Importar CosmosLabels
import logoSrc from './assets/assistant-logo2.png';  // <-- importa tu logo PNG
import MarkdownIt from 'markdown-it'; // Import markdown-it

// ====================================
//     GLOBAL VARIABLES & STATE
// ====================================
let stateHistory = []; // Stack of previous states

// Global graph state management variables
let globalNodes = [];
let globalLinks = [];
let globalGraph = null;
let globalLabels = null;
let globalCurrentExpandedNode = null;
let globalIsRestoringState = false;
let globalIsPaused = false;
let globalPauseButton = null;

// Configuración de las etiquetas disponibles para autocompletado
const availableTags = [
  { 
    id: 'browse', 
    text: '@Browse', 
    icon: '🔍', 
    description: 'Expandir grafo con nodos relacionados',
    enabled: true
  },
  { 
    id: 'select', 
    text: '@Select', 
    icon: '🎯', 
    description: 'Seleccionar nodos específicos (próximamente)',
    enabled: false
  },
  { 
    id: 'create', 
    text: '@Create', 
    icon: '✨', 
    description: 'Crear nuevos nodos (próximamente)',
    enabled: false
  }
];

// Initialize Markdown parser with proper emphasis configuration
const md = new MarkdownIt({
  html: false, // Disable HTML tags for security
  xhtmlOut: true, // Use '/' to close single tags
  breaks: true, // Convert '\n' in paragraphs into <br>
  linkify: true, // Autoconvert URL-like text to links
  typographer: true // Enable some language-neutral replacement + quotes beautification
}).enable(['emphasis', 'strikethrough']); // Explicitly enable emphasis (bold/italic)

// Flag to prevent multiple simultaneous state restoration operations
let isRestoringState = false;

// Crear el botón de retorno si no existe y almacenarlo en una variable global
let returnButton = document.getElementById("return-button");
if (!returnButton) {
  returnButton = document.createElement("button");
  returnButton.id = "return-button";
  returnButton.textContent = "↩"; // Símbolo de retorno
  // CSS will handle most styles via #return-button
  returnButton.style.display = "none"; // Ocultar inicialmente, JS will toggle
  document.body.appendChild(returnButton);
}

// ====================================
//      HELPER FUNCTIONS (GRAPH)
// ====================================

/**
 * Calcula el grado de entrada para cada nodo.
 * @param {Array} nodes - Array de nodos.
 * @param {Array} edges - Array de enlaces.
 * @returns {Object} - Mapa de ID de nodo a grado de entrada.
 */
function calculateInDegree(nodes, edges) {
  const inDegreeMap = {};
  nodes.forEach(node => { inDegreeMap[node.id] = 0; });
  edges.forEach(edge => {
    const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
    if (inDegreeMap[targetId] !== undefined) {
      inDegreeMap[targetId]++;
    }
  });
  return inDegreeMap;
}

/**
 * Calcula el grado de salida para cada nodo.
 * @param {Array} nodes - Array de nodos.
 * @param {Array} edges - Array de enlaces.
 * @returns {Object} - Mapa de ID de nodo a grado de salida.
 */
function calculateOutDegree(nodes, edges) {
  const outDegreeMap = {};
  nodes.forEach(node => { outDegreeMap[node.id] = 0; });
  edges.forEach(edge => {
    const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
    if (outDegreeMap[sourceId] !== undefined) {
      outDegreeMap[sourceId]++;
    }
  });
  return outDegreeMap;
}

/**
 * Calcula el grado total (entrada + salida) para cada nodo.
 * @param {Array} nodes - Array de nodos.
 * @param {Array} edges - Array de enlaces.
 * @returns {Object} - Mapa de ID de nodo a grado total.
 */
function calculateTotalDegree(nodes, edges) {
  const totalDegreeMap = {};
  nodes.forEach(node => { totalDegreeMap[node.id] = 0; });
  edges.forEach(edge => {
    const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
    const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
    if (totalDegreeMap[sourceId] !== undefined) {
      totalDegreeMap[sourceId]++;
    }
    if (totalDegreeMap[targetId] !== undefined) {
      totalDegreeMap[targetId]++;
    }
  });
  return totalDegreeMap;
}

/**
 * Determina el color de un nodo basado en su grado y el grado máximo.
 * @param {number} degree - Grado del nodo.
 * @param {number} maxDegree - Grado máximo en el grafo.
 * @returns {string} - Color hexadecimal.
 */
function getNodeColorByDegree(degree, maxDegree) {
  // Escala de color logarítmica de morado a verde
  const colorScale = d3.scaleSequential(
    t => d3.interpolateRgb("#6640b3", "#8db340")(t)
  ).domain([Math.log(maxDegree + 1), Math.log(1)]); // Dominio logarítmico invertido

  // Handle degree 0 specifically, maybe assign the start color or a distinct color
  if (degree === 0) return "#6640b3"; // Or choose another color like grey: "#aaaaaa"
  // Handle maxDegree 0 case to avoid log(1) domain issues if graph has only isolated nodes
  if (maxDegree === 0) return "#6640b3";

  return colorScale(Math.log(degree + 1));
}


/**
 * Extrae una etiqueta legible de una URI.
 * @param {string} uri - La URI a procesar.
 * @returns {string} - La etiqueta extraída o la URI original.
 */
function getLabelFromUri(uri) {
    if (!uri) return '';
    try {
        // Intenta decodificar y parsear como URL
        const decodedUri = decodeURIComponent(String(uri));
        const url = new URL(decodedUri);
        const pathParts = url.pathname.split('/').filter(part => part.length > 0);
        // Prioriza el fragmento (#) si existe, sino la última parte de la ruta
        const label = url.hash ? url.hash.substring(1) : pathParts.pop();
        return label || uri; // Devuelve la etiqueta o la URI si no se extrajo nada
    } catch (e) {
        // Si falla el parseo de URL (ej. URN o URI mal formada), divide por / y #
        const parts = String(uri).split(/[/#]/);
        return parts.pop() || uri; // Devuelve la última parte o la URI original
    }
};

/**
 * Obtiene el estado actual del grafo para enviar al backend.
 * @param {Array} graphNodes - Los nodos del grafo para enviar.
 * @param {Array} graphLinks - Los enlaces del grafo para enviar.
 * @returns {object} - Objeto con nodos y enlaces simplificados.
 */
function getCurrentGraphState(graphNodes, graphLinks) {
  // Incluir solo propiedades esenciales para los nodos
  const currentNodes = graphNodes.map(node => ({
      id: node.id,
      label: node.label || getLabelFromUri(String(node.id))
  }));
  
  // Para enlaces, asegurar consistencia en la propiedad 'link' 
  // que es la esperada por graph_data_to_triples
  const currentLinks = graphLinks.map(link => ({
      source: typeof link.source === 'object' ? link.source.id : link.source,
      target: typeof link.target === 'object' ? link.target.id : link.target,
      link: link.label || link.link || 'connected_to' // Asegurar propiedad 'link'
  }));
  
  return { nodes: currentNodes, links: currentLinks };
}

/**
 * Filters out NamedIndividual and owl:Thing nodes and related edges from graph data
 * @param {Array} nodes - Array of nodes to filter
 * @param {Array} edges - Array of edges to filter
 * @returns {Object} - Filtered nodes and edges
 */
function filterOutNamedIndividual(nodes, edges) {
  // Filter out NamedIndividual and owl:Thing nodes
  const filteredNodes = nodes.filter(node => {
    if (!node || !node.id) return false;
    
    const nodeLabel = node.label || getLabelFromUri(node.id);
    const nodeId = String(node.id);
    
    // Check if this is a NamedIndividual node
    const isNamedIndividual = nodeLabel === 'NamedIndividual' || 
                              nodeId.includes('NamedIndividual') ||
                              nodeId.endsWith('#NamedIndividual') ||
                              nodeId.endsWith('/NamedIndividual');
    
    // Check if this is owl:Thing
    const isOwlThing = nodeLabel === 'Thing' || 
                       nodeId === 'http://www.w3.org/2002/07/owl#Thing' ||
                       nodeId.endsWith('#Thing') ||
                       nodeId.endsWith('/Thing');
    
    if (isNamedIndividual) {
      console.log(`Filtering out NamedIndividual node: ${nodeId}`);
      return false;
    }
    
    if (isOwlThing) {
      console.log(`Filtering out owl:Thing node: ${nodeId}`);
      return false;
    }
    
    return true;
  });
  
  // Get IDs of remaining nodes for edge filtering
  const remainingNodeIds = new Set(filteredNodes.map(n => n.id));
  
  // Filter out edges that involve NamedIndividual, owl:Thing, or connect to filtered nodes
  const filteredEdges = edges.filter(edge => {
    if (!edge) return false;
    
    const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
    const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
    const linkLabel = edge.label || edge.link || '';
    
    // Check if edge involves NamedIndividual
    const sourceIsNamedIndividual = String(sourceId).includes('NamedIndividual');
    const targetIsNamedIndividual = String(targetId).includes('NamedIndividual');
    const isTypeToNamedIndividual = linkLabel.includes('type') && targetIsNamedIndividual;
    
    // Check if edge involves owl:Thing
    const sourceIsOwlThing = String(sourceId) === 'http://www.w3.org/2002/07/owl#Thing' || String(sourceId).endsWith('#Thing');
    const targetIsOwlThing = String(targetId) === 'http://www.w3.org/2002/07/owl#Thing' || String(targetId).endsWith('#Thing');
    
    if (sourceIsNamedIndividual || targetIsNamedIndividual || isTypeToNamedIndividual) {
      console.log(`Filtering out edge involving NamedIndividual: ${sourceId} -> ${linkLabel} -> ${targetId}`);
      return false;
    }
    
    if (sourceIsOwlThing || targetIsOwlThing) {
      console.log(`Filtering out edge involving owl:Thing: ${sourceId} -> ${linkLabel} -> ${targetId}`);
      return false;
    }
    
    // Check if both source and target nodes still exist after node filtering
    return remainingNodeIds.has(sourceId) && remainingNodeIds.has(targetId);
  });
  
  console.log(`Filtered out ${nodes.length - filteredNodes.length} NamedIndividual/owl:Thing nodes and ${edges.length - filteredEdges.length} related edges`);
  
  return {
    nodes: filteredNodes,
    edges: filteredEdges
  };
}

// ====================================
//    GLOBAL STATE MANAGEMENT FUNCTIONS
// ====================================

/**
 * Force clear all label-related elements and processes
 */
function forceCleanAllLabels() {
  console.log('Starting force clean of all labels');
  
  // Stop any potential animation frames or timers that might be updating labels
  if (window.requestAnimationFrame) {
    // Cancel any pending animation frames (labels might use these)
    let id = window.requestAnimationFrame(() => {});
    while (id > 0) {
      window.cancelAnimationFrame(id);
      id--;
      if (id < window.requestAnimationFrame(() => {}) - 100) break; // Safety limit
    }
  }
  
  // Clear the main labels container first
  const labelsDiv = document.getElementById("labels");
  if (labelsDiv) {
    labelsDiv.innerHTML = '';
    // Remove all child nodes manually as a double-check
    while (labelsDiv.firstChild) {
      labelsDiv.removeChild(labelsDiv.firstChild);
    }
  }
  
  // Find and remove all CSS label elements
  const allLabelElements = document.querySelectorAll('.css-label, [class*="css-label"], [class*="label-"]');
  allLabelElements.forEach(element => {
    element.remove();
  });
  
  // Find any elements that might contain node text/labels
  const potentialLabels = document.querySelectorAll('div, span, text');
  potentialLabels.forEach(element => {
    // Skip if it's part of UI elements we want to keep
    if (element.closest('.action-container') || 
        element.closest('#assistant-container') || 
        element.closest('#context-menu') ||
        element.closest('.modal') ||
        element.id === 'labels' ||
        element.tagName === 'BUTTON' ||
        element.tagName === 'INPUT') {
      return;
    }
    
    // Check if element contains what looks like a node label
    const text = element.textContent?.trim();
    if (text && 
        text.length < 100 && // Reasonable label length
        !text.includes(' ') && // Single word (typical of node labels)
        (text.includes('#') || text.includes('/') || text.includes(':')) // URI-like
       ) {
      console.log('Removing potential orphaned label element:', text);
      element.remove();
    }
  });
  
  console.log('Force clean of all labels completed');
}

/**
 * Global function to save the current state of the graph
 */
function saveCurrentState() {
  if (!globalNodes || !globalLinks) {
    console.warn('Cannot save state: global nodes or links not available');
    return;
  }
  
  const newState = {
    nodes: JSON.parse(JSON.stringify(globalNodes)), 
    links: JSON.parse(JSON.stringify(globalLinks)), 
  };
  
  stateHistory.push(newState);
  
  const returnButton = document.getElementById("return-button");
  if (returnButton) {
    returnButton.style.display = "flex"; // Show button
    console.log("Return button shown. History size:", stateHistory.length);
  } else {
    console.warn("Return button not found when trying to show it");
  }
  console.log("Current state saved to history. History size:", stateHistory.length);
}

/**
 * Global function to restore the graph to the previous state
 */
function returnToPreviousState() {
  console.log(`[returnToPreviousState] Called. Current history size: ${stateHistory.length}`);
  console.trace('[returnToPreviousState] Call stack trace');
  
  // Prevent multiple simultaneous calls
  if (globalIsRestoringState) {
    console.log('[returnToPreviousState] Already restoring state, ignoring duplicate call');
    return;
  }
  
  if (stateHistory.length > 0 && globalGraph && globalNodes && globalLinks) {
    globalIsRestoringState = true;
    console.log("Restoring to previous state...");
    
    const previousState = stateHistory.pop();
    
    globalNodes = [...previousState.nodes];
    globalLinks = [...previousState.links];

    // Clear the currently expanded node when returning to previous state
    globalCurrentExpandedNode = null;

    console.log("State restored. Remaining history size:", stateHistory.length);

    const returnButton = document.getElementById("return-button");
    if (stateHistory.length === 0 && returnButton) {
      returnButton.style.display = "none"; // Hide button if no more history
      console.log("Return button hidden - no more history available");
    }

    // Update graph properties and display
    updateGlobalGraphProperties();
    
    // Update the actual graph
    const wasPaused = globalIsPaused; 
    globalGraph.pause();

    setTimeout(() => {
      try {
        const current_node_ids = new Set(globalNodes.map(n => n.id));
        const validLinks = globalLinks.filter(link => {
          const sourceId = typeof link.source === 'object' ? link.source.id : link.source;
          const targetId = typeof link.target === 'object' ? link.target.id : link.target;
          const sourceExists = current_node_ids.has(sourceId);
          const targetExists = current_node_ids.has(targetId);
          return sourceExists && targetExists;
        });
        
        console.log("Setting restored data:", globalNodes.length, "nodes,", validLinks.length, "valid links");

        globalGraph.setData(globalNodes, validLinks);

        if (globalLabels) globalLabels.updateNodeMap(globalNodes);

        const nodeIds = globalNodes.map(n => n.id);
        globalGraph.trackNodePositionsByIds(nodeIds);

        globalGraph.unselectNodes();
        
        // Call fitView from the global scope
        if (window.globalFitView) {
          window.globalFitView(300, 0.3);
        }

        // Restore pause state
        const pauseButton = globalPauseButton || document.getElementById("pause");
        if (wasPaused) {
          globalGraph.pause();
          globalIsPaused = true;
          if (pauseButton) pauseButton.textContent = "Resume";
          console.log("Simulation kept paused after state restoration.");
        } else {
          globalGraph.start();
          globalIsPaused = false;
          if (pauseButton) pauseButton.textContent = "Pause";
          console.log("Simulation resumed after state restoration.");
        }

        console.log("Restored to previous state successfully.");
        globalIsRestoringState = false; // Reset flag
      } catch (err) {
        console.error("Error during state restoration:", err);
        globalIsRestoringState = false; // Reset flag even on error
        
        // Restore pause state on error too
        const pauseButton = globalPauseButton || document.getElementById("pause");
        if (wasPaused) {
          globalGraph.pause();
          globalIsPaused = true;
          if (pauseButton) pauseButton.textContent = "Resume";
        } else {
          globalGraph.start();
          globalIsPaused = false;
          if (pauseButton) pauseButton.textContent = "Pause";
        }
      }
    }, 100); 

  } else {
    console.warn("No previous state saved to restore or graph not available.");
    const returnButton = document.getElementById("return-button");
    if (returnButton) {
      returnButton.style.display = "none";
      console.log("Return button hidden - no previous state available");
    }
  }
}

/**
 * Global function to update graph properties
 */
function updateGlobalGraphProperties() {
  if (!globalNodes || !globalLinks || !globalGraph) {
    console.warn('Cannot update graph properties: required globals not available');
    return;
  }
  
  // Calculate degrees
  const inDegreeMap = calculateInDegree(globalNodes, globalLinks);
  const outDegreeMap = calculateOutDegree(globalNodes, globalLinks);
  const totalDegreeMap = calculateTotalDegree(globalNodes, globalLinks);
  const maxTotalDegree = Math.max(...Object.values(totalDegreeMap));

  // Update node properties with colors and labels
  globalNodes.forEach(node => {
    const degree = totalDegreeMap[node.id] || 0;
    node.color = getNodeColorByDegree(degree, maxTotalDegree);
    
    if (!node.label) {
      node.label = getLabelFromUri(String(node.id));
    }
  });

  console.log("Global graph properties updated.");
}

/**
 * Reset global state for new ontology
 */
function resetGlobalState() {
  console.log('Resetting global graph state');
  stateHistory = [];
  globalNodes = [];
  globalLinks = [];
  globalCurrentExpandedNode = null;
  globalIsRestoringState = false;
  globalIsPaused = false;
  globalPauseButton = document.getElementById("pause");
  
  // Enhanced label cleanup
  console.log('Performing enhanced label cleanup');
  
  // Clear globalLabels
  if (globalLabels) {
    try {
      globalLabels.cleanup();
    } catch (error) {
      console.warn('Error during globalLabels cleanup:', error);
    }
    globalLabels = null;
  }
  
  // Clear the main labels container
  const labelsDiv = document.getElementById("labels");
  if (labelsDiv) {
    console.log('Clearing labels container and all child elements');
    labelsDiv.innerHTML = '';
    
    // Force a style recalculation to clear any cached styles
    labelsDiv.style.display = 'none';
    labelsDiv.offsetHeight; // Trigger reflow
    labelsDiv.style.display = '';
  }
  
  // Remove any orphaned label elements throughout the document
  const orphanedLabels = document.querySelectorAll('.css-label, [class*="label-"], [data-label]');
  if (orphanedLabels.length > 0) {
    console.log(`Found ${orphanedLabels.length} orphaned label elements, removing them`);
    orphanedLabels.forEach((label, index) => {
      console.log(`Removing orphaned label ${index + 1}:`, label.textContent?.substring(0, 30) || label.className);
      label.remove();
    });
  }
  
  // Clear any potential label-related timers or intervals
  // This helps prevent delayed label creation after cleanup
  if (window.labelUpdateTimer) {
    clearTimeout(window.labelUpdateTimer);
    window.labelUpdateTimer = null;
  }
  
  if (window.labelUpdateInterval) {
    clearInterval(window.labelUpdateInterval);
    window.labelUpdateInterval = null;
  }
  
  const returnButton = document.getElementById("return-button");
  if (returnButton) {
    returnButton.style.display = "none";
    console.log('Return button hidden for new session');
  }
  
  console.log('Global state reset and label cleanup complete');
}

// ====================================
//     MAIN APPLICATION LOGIC
// ====================================

/**
 * Main application initialization function
 * @param {string} source - The source of the graph data (graph URI or file name)
 */
// Global flag to prevent multiple simultaneous initializations
let isInitializing = false;

async function initializeMainApplication(source) {
  // Prevent multiple simultaneous initializations
  if (isInitializing) {
    console.warn('Application is already initializing, ignoring duplicate call');
    return;
  }
  
  isInitializing = true;
  console.log('Initializing main application with source:', source);
  
  // Update the graph URI configuration if source is provided
  if (source && typeof source === 'string' && source.startsWith('http')) {
    console.log('Updating SPARQL configuration to use graph URI:', source);
    updateGraphUri(source);
  }
  
  // Clear state history when loading a new ontology to prevent confusion with previous states
  console.log('Resetting global state for new ontology session');
  
  // Force clean all labels before resetting state
  forceCleanAllLabels();
  
  resetGlobalState();
  
  // Ensure main app is visible and ready
  const mainApp = document.getElementById('main-app');
  if (mainApp) {
    console.log('Main app element found:', mainApp);
    mainApp.classList.remove('hidden');
    // Ensure it's displayed properly
    if (mainApp.style.display === 'none') {
      mainApp.style.display = '';
    }
  } else {
    console.warn('Main app element not found!');
  }
  
  console.log('Starting fetchInitialGraph...');
  
  try {
    // --- 1. Inicialización y Carga de Datos ---
    const initialData = await fetchInitialGraph(source);
    console.log('fetchInitialGraph completed successfully');

    let nodes = initialData.nodes || [];
    let links = initialData.edges || [];
    let literals = initialData.literals || {}; // Almacena literales asociados a nodos

    console.log("Datos iniciales cargados - Nodos:", nodes.length, "Enlaces:", links.length);

  // --- Identificar clases expandibles dinámicamente ---
  // Asumiendo que fetchVirtuoso devuelve las clases que queremos expandir
  const expandableClassIds = new Set();
  if (nodes && Array.isArray(nodes)) {
      nodes.forEach(node => expandableClassIds.add(node.id));
      console.log(`Identificadas ${expandableClassIds.size} clases expandibles dinámicamente.`);
  }
  // ----------------------------------------------------

  let canvas = document.querySelector("canvas");
  
  // Ensure canvas exists
  if (!canvas) {
    console.warn('Canvas not found, creating it...');
    canvas = document.createElement('canvas');
    
    // Insert it into the main app container
    const mainApp = document.getElementById('main-app');
    if (mainApp) {
      mainApp.appendChild(canvas);
      console.log('Canvas created and added to main-app');
    } else {
      document.body.appendChild(canvas);
      console.log('Canvas created and added to body');
    }
  }
  
  console.log('Canvas element:', canvas);
  let labelsDiv = document.getElementById("labels"); // Contenedor para etiquetas
  
  // Ensure labels container exists
  if (!labelsDiv) {
    console.warn('Labels container not found, creating it...');
    labelsDiv = document.createElement('div');
    labelsDiv.id = 'labels';
    labelsDiv.style.position = 'absolute';
    labelsDiv.style.top = '0';
    labelsDiv.style.left = '0';
    labelsDiv.style.width = '100%';
    labelsDiv.style.height = '100%';
    labelsDiv.style.pointerEvents = 'none';
    labelsDiv.style.overflow = 'hidden';
    
    // Insert it into the main app container
    const mainApp = document.getElementById('main-app');
    if (mainApp) {
      mainApp.appendChild(labelsDiv);
      console.log('Labels container created and added to main-app');
    } else {
      document.body.appendChild(labelsDiv);
      console.log('Labels container created and added to body');
    }
  }
  
  console.log('Labels div element:', labelsDiv);

  // Clean up any existing instances to prevent overlapping
  if (window.globalGraph) {
    console.log('Cleaning up existing graph instance');
    try {
      // Force clean all labels first
      forceCleanAllLabels();
      
      // Clear the canvas
      const context = canvas.getContext('2d');
      if (context) {
        context.clearRect(0, 0, canvas.width, canvas.height);
      }
      window.globalGraph = null;
    } catch (error) {
      console.warn('Error clearing previous graph:', error);
    }
  }
  
  if (window.globalLabels) {
    console.log('Cleaning up existing labels instance');
    try {
      // Use the cleanup method to properly reset the labels system
      window.globalLabels.cleanup();
      
      // Force clear any remaining labels in the DOM
      const labelsDiv = document.getElementById("labels");
      if (labelsDiv) {
        console.log('Force clearing all label elements from DOM');
        labelsDiv.innerHTML = '';
        
        // Also clear any css-label elements that might be outside the container
        const existingLabels = document.querySelectorAll('.css-label');
        existingLabels.forEach(label => {
          console.log('Removing orphaned label element:', label.textContent);
          label.remove();
        });
        
        // Clear any potential label renderers or overlays
        const labelOverlays = document.querySelectorAll('[class*="label"], [id*="label"]');
        labelOverlays.forEach(overlay => {
          if (overlay.id !== 'labels' && overlay.textContent && overlay.textContent.trim()) {
            console.log('Removing potential label overlay:', overlay.id, overlay.className, overlay.textContent.substring(0, 50));
            // Only remove if it's not a main UI element
            if (!overlay.closest('.action-container') && 
                !overlay.closest('#assistant-container') && 
                !overlay.closest('#context-menu') &&
                !overlay.id.includes('button') &&
                !overlay.id.includes('input')) {
              overlay.remove();
            }
          }
        });
      }
      
      window.globalLabels = null;
      globalLabels = null;
    } catch (error) {
      console.warn('Error clearing previous labels:', error);
    }
  }
  
  // Clear the labels container as additional safety measure
  if (labelsDiv) {
    console.log('Clearing labels container');
    labelsDiv.innerHTML = '';
  }
  
  // Clear the canvas completely
  if (canvas) {
    const context = canvas.getContext('2d');
    if (context) {
      context.clearRect(0, 0, canvas.width, canvas.height);
    }
    // Remove any existing event listeners by cloning the canvas
    const newCanvas = canvas.cloneNode(true);
    canvas.parentNode.replaceChild(newCanvas, canvas);
    // Update canvas reference
    canvas = newCanvas;
  }

  let graph; // Instancia del grafo Cosmograph
  let labels; // Instancia de CosmosLabels
  let inDegreeMap, outDegreeMap, totalDegreeMap, maxTotalDegree; // Variables para grados
  
  // Update global references
  globalNodes = nodes;
  globalLinks = links;
  globalCurrentExpandedNode = null;
  
  // Reset any global state that might interfere
  console.log('Resetting global application state');
  
  console.log('Reset currentExpandedNode for new session');

  // --- 2. Funciones de Actualización del Grafo ---

  /**
   * Shows a banner message to inform the user about the current state
   * @param {string} message - Message to display in the banner
   * @param {string} type - Type of banner ('info', 'warning', 'success')
   */
  function showBanner(message, type = 'info') {
    // Remove any existing banner with explosion animation
    const existingBanner = document.getElementById('graph-banner');
    if (existingBanner) {
      explodeBanner(existingBanner);
    }

    // Create new banner after a short delay to avoid conflicts
    setTimeout(() => {
      const banner = document.createElement('div');
      banner.id = 'graph-banner';
      banner.className = `graph-banner graph-banner-${type}`;
      banner.textContent = message;

      // Add close button
      const closeButton = document.createElement('button');
      closeButton.className = 'graph-banner-close';
      closeButton.innerHTML = '×';
      closeButton.onclick = () => explodeBanner(banner);

      banner.appendChild(closeButton);
      document.body.appendChild(banner);

      // Auto-hide after 3 seconds with explosion
      setTimeout(() => {
        if (banner.parentNode) {
          explodeBanner(banner);
        }
      }, 3000);
    }, existingBanner ? 200 : 0);
  }

  /**
   * Triggers explosion animation and removes banner
   * @param {HTMLElement} banner - Banner element to explode
   */
  function explodeBanner(banner) {
    if (!banner || !banner.parentNode) return;
    
    banner.classList.add('graph-banner-exploding');
    setTimeout(() => {
      if (banner.parentNode) {
        banner.remove();
      }
    }, 500); // Match the new deflate animation duration
  }

  /**
   * Calcula grados, asigna colores y etiquetas a los nodos.
   */
  function updateGraphProperties() {
    // --- FILTER OUT NULL NODES FIRST ---
    if (globalNodes && Array.isArray(globalNodes)) {
      const originalNodeCount = globalNodes.length;
      globalNodes = globalNodes.filter(node => node && node.id && String(node.id).toLowerCase() !== 'null');
      if (globalNodes.length < originalNodeCount) {
        console.warn(`[updateGraphProperties] Filtered out ${originalNodeCount - globalNodes.length} nodes with "null" ID.`);
      }
    }
    // Update local reference
    nodes = globalNodes;
    // ------------------------------------

    // --- FILTRAR ENLACES PARA EVITAR ERRORES ---
    const nodeIdSet = new Set(globalNodes.map(n => n.id));
    // Solo mantener enlaces donde source y target existen en nodes y no son null/undefined
    const filteredLinks = globalLinks.filter(edge => {
      // Verifica que source y target existan
      if (!edge || edge.source == null || edge.target == null) return false;
      const sourceId = typeof edge.source === 'object' && edge.source !== null ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' && edge.target !== null ? edge.target.id : edge.target;
      // Si sourceId o targetId es null/undefined, descarta el enlace
      if (sourceId == null || targetId == null) return false;
      return nodeIdSet.has(sourceId) && nodeIdSet.has(targetId);
    });
    if (filteredLinks.length !== globalLinks.length) {
      console.warn(`[updateGraphProperties] Filtered out ${globalLinks.length - filteredLinks.length} links with missing or invalid nodes.`);
    }
    globalLinks = filteredLinks;
    // Update local reference
    links = globalLinks;
    // -------------------------------------------

    inDegreeMap = calculateInDegree(globalNodes, globalLinks);
    outDegreeMap = calculateOutDegree(globalNodes, globalLinks);
    totalDegreeMap = calculateTotalDegree(globalNodes, globalLinks);
    const totalDegrees = Object.values(totalDegreeMap);
    maxTotalDegree = totalDegrees.length > 0 ? Math.max(...totalDegrees) : 0;
    console.log("Max total degree:", maxTotalDegree); // Log max degree

    nodes.forEach(node => {
      const totalDegree = totalDegreeMap[node.id] || 0;
      // Asigna color basado en grado si no tiene o necesita actualización
      // Force color update if degree calculation might change it
      node.color = getNodeColorByDegree(totalDegree, maxTotalDegree);
      delete node.needsColorUpdate; // Marca como actualizado

      // Asigna etiqueta si no tiene
      if (!node.label) {
          // Ensure getLabelFromUri doesn't return "null" for a "null" id, though node.id itself is already filtered
          node.label = getLabelFromUri(node.id);
      }
    });

    // drawColorBar('#color-scale', maxTotalDegree); // Opcional: Dibujar barra de colores
  }

  /**
   * Ajusta la vista para encuadrar todos los nodos visibles.
   * @param {number} duration - Duración de la animación (ms).
   * @param {number} padding - Espaciado alrededor de los nodos (0-1).
   */
   function fitView(duration = 250, padding = 0.2) {
    const activeGraph = globalGraph || graph;
    if (activeGraph) { // Check if graph is initialized
        activeGraph.fitView(duration, padding);
    } else {
        console.warn("FitView called before graph initialization.");
    }
   }
   
   // Make fitView globally available
   window.globalFitView = fitView;

  // --- 3. Configuración del Grafo (Cosmograph) ---
  const config = {
    backgroundColor: "#f4f4f4", // This is for canvas, assistant has its own background
    nodeSize: 4, // Tamaño base del nodo
    nodeSizeScale: 1, // Escala de tamaño del nodo (relativo a nodeSize)
    nodeColor: node => node.color || "#151515", // Color por defecto si no está asignado
    nodeGreyoutOpacity: 0.1, // Opacidad de nodos no seleccionados/resaltados
    linkWidth: 1.8, // Ancho del enlace
    linkColor: "#5F74C2", // Color del enlace
    linkGreyoutOpacity: 0, // Opacidad de enlaces no resaltados (0 = invisibles)
    curvedLinks: true, // Usar enlaces curvos
    showFPSMonitor: true, // Mostrar monitor de FPS
    renderHoveredNodeRing: true, // Dibujar anillo al pasar el ratón sobre un nodo
    hoveredNodeRingColor: node => node.color || "#cccccc", // Color del anillo (igual al nodo)
    linkVisibilityDistanceRange: [10, 800], // Rango de distancia para visibilidad de enlaces
    linkArrows: true, // Mostrar flechas en los enlaces
    linkArrowsSizeScale: 1.5, // Tamaño de las flechas
    initialZoomLevel: 0.7, // Nivel de zoom inicial
    forceCollide: true, // Activar detección de colisiones para evitar superposiciones
    collideRadius: 6, // Radio de colisión (mayor que nodeSize para espacio extra)
    simulation: {
      gravity: 0.1, // Gravedad moderada
      linkDistance: 30, // Mayor distancia entre nodos conectados
      repulsion: 2, // Incrementada repulsión para evitar superposiciones
      friction: 0.5, // Fricción reducida para más movimiento
      onTick: () => { // Función ejecutada en cada paso de la simulación
        try {
          if (labels) labels.update(graph); // Actualizar posición de etiquetas
        } catch (error) {
          console.warn('Error updating labels on tick:', error);
        }
      },
      forces: { // Configuración fina de las fuerzas de simulación
        center: { strength: 5 }, // Fuerza central moderada
        repulsion: { strength: 25, distanceMax: 2500 }, // Mayor repulsión para evitar superposiciones
        link: { strength: 0.3 } // Añadida fuerza de enlace moderada
      }
    },
    events: {
      /**
       * Manejador de evento al hacer clic en un nodo.
       */
      onClick: async (node, i, pos, event) => {
        if (!node) { // Si se hace clic fuera de un nodo
            graph.unselectNodes(); // Deseleccionar todo
            return;
        }

        graph.selectNodeById(node.id, true); // Seleccionar el nodo clicado

        // Check if this node is already the currently expanded node
        if (globalCurrentExpandedNode && globalCurrentExpandedNode === node.id) {
          const nodeLabel = node.label || getLabelFromUri(node.id);
          showBanner(`El nodo "${nodeLabel}" ya está expandido actualmente`, 'info');
          return; // Exit early without expanding again
        }

        // Si el nodo clicado es una de las clases identificadas dinámicamente
        if (expandableClassIds.has(node.id)) {
          try {
            // Guardar el estado de pausa antes de la expansión
            const wasPausedBeforeExpansion = globalIsPaused;
            
            // Primero intentamos cargar subclases
            console.log(`[onClick] Node ${node.id} IS expandable. Fetching subclasses first...`);
            const subclassData = await fetchSubclasses(node.id);
            
            // Filter out NamedIndividual from subclass data
            const filteredSubclassData = filterOutNamedIndividual(
              subclassData.nodes || [], 
              subclassData.edges || []
            );
            
            // Verificamos si hay subclases (más de un nodo, ya que el nodo padre siempre está incluido)
            if (filteredSubclassData.nodes.length > 1) {
              console.log(`[onClick] Found ${filteredSubclassData.nodes.length-1} subclasses for ${node.id}. Processing...`);
              
              const existingNodeIds = new Set(nodes.map(n => n.id));
              const newNodes = filteredSubclassData.nodes.filter(n => 
                n && n.id && String(n.id).toLowerCase() !== 'null' && !existingNodeIds.has(n.id)
              );
              console.log(`[onClick] Filtered ${newNodes.length} NEW valid subclass nodes.`);
              
              // Añadir todas las nuevas subclases al set de clases expandibles
              newNodes.forEach(node => {
                expandableClassIds.add(node.id);
                console.log(`[onClick] Added new subclass ${node.id} to expandable classes`);
              });
              
              let newLinks = filteredSubclassData.edges.filter(l => {
                const sourceId = typeof l.source === 'object' ? l.source.id : l.source;
                const targetId = typeof l.target === 'object' ? l.target.id : l.target;
                // Solo incluir enlaces relevantes entre nodos actuales
                return sourceId && targetId;
              });
              console.log(`[onClick] Filtered ${newLinks.length} NEW relevant subclass links.`);
              
              // Fusionar literales nuevos si hay
              if (subclassData.literals) {
                for (const sourceId in subclassData.literals) {
                  if (!literals[sourceId]) literals[sourceId] = {};
                  for (const linkId in subclassData.literals[sourceId]) {
                    if (!literals[sourceId][linkId]) literals[sourceId][linkId] = [];
                    // Usar Set para evitar duplicados al fusionar
                    literals[sourceId][linkId] = [...new Set([...literals[sourceId][linkId], ...subclassData.literals[sourceId][linkId]])];
                  }
                }
              }
              
              // Actualizar el grafo con subclases
              if (newNodes.length > 0 || newLinks.length > 0) {
                console.log("[onClick] Updating graph with new subclass nodes/links.");
                
                saveCurrentState(); // Guardar estado actual antes de modificar el grafo
                
                globalNodes = [node, ...newNodes]; // Mantener nodo padre y añadir hijos
                globalLinks = [...newLinks]; // Solo enlaces que conectan estos nodos
                nodes = globalNodes;
                links = globalLinks;
                
                // Set this node as the currently expanded one
                globalCurrentExpandedNode = node.id;
                
                console.log("After subclass expansion - Nodes:", nodes.length, "Links:", links.length);
                
                updateGraphProperties(); // Recalcular propiedades (colores, etc.)
                graph.setData(nodes, links); // Actualizar datos en Cosmograph
                
                // Resto del código para rastrear y ajustar la vista...
                const nodeIds = nodes.map(n => n.id);
                graph.trackNodePositionsByIds(nodeIds);
                
                // Actualizar el mapa de nodos en el gestor de etiquetas
                if (labels) labels.updateNodeMap(nodes);
                
                // Seleccionar todos los nodos actualmente visibles
                graph.selectNodesByIds(nodeIds, false); 
                
                // Mantener el estado de pausa después de la expansión
                if (wasPausedBeforeExpansion) {
                  graph.pause();
                  if (pauseButton) pauseButton.textContent = "Resume";
                  console.log("Simulation kept paused after subclass expansion.");
                } else {
                  graph.start();
                  if (pauseButton) pauseButton.textContent = "Pause";
                }
                
                // Ajustar la vista después de la expansión
                setTimeout(() => {
                  fitView(500, 0.2); // Ajustar vista con animación y padding
                }, 700);
                
                return; // Terminar aquí, ya hemos expandido subclases
              } else {
                // No subclasses found to expand
                console.log(`[onClick] No new subclass nodes or links found for ${node.id}.`);
                // En lugar de mostrar banner, continuar con la verificación de instancias
                // Esto permitirá que el código siga el flujo normal hacia la consulta de instancias
              }
            } else {
              // No subclasses found at all
              console.log(`[onClick] No subclasses found for ${node.id}.`);
            }
            
            // Si llegamos aquí, no hay subclases o no se pudieron cargar
            // Entonces cargamos las instancias como antes
            console.log(`[onClick] No subclasses found for ${node.id}. Fetching instances instead...`);
            const childData = await fetchClassInstances(node.id); // Consultar instancias
            
            // Filter out NamedIndividual from instance data
            const filteredChildData = filterOutNamedIndividual(
              childData.nodes || [], 
              childData.edges || []
            );
            
            // *** LOG DETALLADO DE LA RESPUESTA ***
            console.log(`[onClick] Received and filtered childData for ${node.id}:`, {
              originalNodes: childData.nodes?.length || 0,
              filteredNodes: filteredChildData.nodes.length,
              originalEdges: childData.edges?.length || 0,
              filteredEdges: filteredChildData.edges.length
            });

            // Check if we have any meaningful expansion data
            const hasNewContent = filteredChildData.nodes.length > 0 || filteredChildData.edges.length > 0;
            
            if (hasNewContent) {
                console.log(`[onClick] filteredChildData has ${filteredChildData.nodes.length} nodes. Processing...`);

                const existingNodeIds = new Set(nodes.map(n => n.id));
                const newNodes = filteredChildData.nodes.filter(n => 
                    n && n.id && String(n.id).toLowerCase() !== 'null' && !existingNodeIds.has(n.id)
                );
                console.log(`[onClick] Filtered ${newNodes.length} NEW valid nodes.`);

                // Añadir todas las nuevas instancias al set de clases expandibles
                newNodes.forEach(node => {
                  expandableClassIds.add(node.id);
                  console.log(`[onClick] Added instance ${node.id} to expandable classes for further exploration`);
                });

                // --- CAMBIO: en vez de filtrar los enlaces, usar todos los edges retornados ---
                let newLinks = filteredChildData.edges; // Use filtered edges
                console.log(`[onClick] Using all ${newLinks.length} filtered links from childData.`);

                // Fusionar literales nuevos con los existentes
                if (childData.literals) {
                    for (const sourceId in childData.literals) {
                        if (!literals[sourceId]) literals[sourceId] = {};
                        for (const linkId in childData.literals[sourceId]) {
                            if (!literals[sourceId][linkId]) literals[sourceId][linkId] = [];
                            // Usar Set para evitar duplicados al fusionar
                            literals[sourceId][linkId] = [...new Set([...literals[sourceId][linkId], ...childData.literals[sourceId][linkId]])];
                        }
                    }
                }

                // Only proceed if we actually have new meaningful content
                if (newNodes.length > 0 || newLinks.length > 0) {
                    console.log("[onClick] Updating graph with new instance nodes/links.");
                    
                    saveCurrentState(); // Guardar estado actual antes de modificar el grafo
                    
                    globalNodes = [node, ...newNodes]; // Keep parent and add all new children
                    globalLinks = [...newLinks]; // Usar todos los enlaces retornados
                    nodes = globalNodes;
                    links = globalLinks;

                    // Set this node as the currently expanded one
                    globalCurrentExpandedNode = node.id;

                    console.log("After instance expansion - Nodes:", nodes.length, "Links:", links.length);

                    updateGraphProperties(); // Recalcular propiedades (colores, etc.)
                    graph.setData(nodes, links); // Actualizar datos en Cosmograph

                    // Rastrear posiciones de los nodos actuales
                    const nodeIds = nodes.map(n => n.id);
                    graph.trackNodePositionsByIds(nodeIds);

                    // Actualizar el mapa de nodos en el gestor de etiquetas
                    if (labels) labels.updateNodeMap(nodes);

                    // Seleccionar todos los nodos actualmente visibles
                    console.log("Selecting all nodes after expansion:", nodeIds);
                    graph.selectNodesByIds(nodeIds, false); // Select only these nodes, replacing previous selection

                    // Mantener el estado de pausa después de la expansión
                    if (wasPausedBeforeExpansion) {
                      graph.pause();
                      if (pauseButton) pauseButton.textContent = "Resume";
                      console.log("Simulation kept paused after instance expansion.");
                    } else {
                      graph.start();
                      if (pauseButton) pauseButton.textContent = "Pause";
                    }

                } else {
                    console.log("[onClick] No new meaningful nodes or links found after filtering.");
                    // Verificar si el nodo es una instancia para mostrar el menú contextual
                    try {
                      const isInstance = await isInstanceNode(node.id);
                      console.log(`[onClick] Node ${node.id} isInstance result (no expansion): ${isInstance}`);
                      
                      if (isInstance) {
                        // Mostrar el menú contextual para instancias
                        console.log(`[onClick] Showing context menu for instance node (no expansion): ${node.id}`);
                        console.log(`[onClick] About to call showContextMenu with event.clientX=${event.clientX}, event.clientY=${event.clientY}`);
                        showContextMenu(event.clientX, event.clientY, node);
                        console.log(`[onClick] showContextMenu call completed for instance node`);
                      } else {
                        // Es una clase sin más contenido - también mostrar menú contextual
                        console.log(`[onClick] Showing context menu for class node (no expansion): ${node.id}`);
                        console.log(`[onClick] About to call showContextMenu with event.clientX=${event.clientX}, event.clientY=${event.clientY}`);
                        showContextMenu(event.clientX, event.clientY, node);
                        console.log(`[onClick] showContextMenu call completed for class node`);
                      }
                    } catch (error) {
                      console.error(`Error checking if node ${node.id} is an instance (no expansion):`, error);
                      // Fallback: mostrar banner normal
                      const nodeLabel = node.label || getLabelFromUri(node.id);
                      showBanner(`El nodo "${nodeLabel}" no tiene más elementos para expandir`, 'warning');
                    }
                    graph.selectNodeById(node.id, true); // Still select the node
                }
      } else {
        // No se encontró contenido para expandir
        console.log(`[onClick] No expansion data found for ${node.id} - empty response from backend.`);
        // Verificar si el nodo es una instancia para mostrar el menú contextual
        try {
          const isInstance = await isInstanceNode(node.id);
          console.log(`[onClick] Node ${node.id} isInstance result (empty response): ${isInstance}`);
          
          if (isInstance) {
            // Mostrar el menú contextual para instancias
            console.log(`[onClick] Showing context menu for instance node (empty response): ${node.id}`);
            showContextMenu(event.clientX, event.clientY, node);
          } else {
            // Es una clase sin más contenido - también mostrar menú contextual
            console.log(`[onClick] Showing context menu for class node (empty response): ${node.id}`);
            showContextMenu(event.clientX, event.clientY, node);
          }
        } catch (error) {
          console.error(`Error checking if node ${node.id} is an instance (empty response):`, error);
          // Fallback: mostrar banner normal
          const nodeLabel = node.label || getLabelFromUri(node.id);
          showBanner(`El nodo "${nodeLabel}" no tiene más elementos para expandir`, 'warning');
        }
        graph.selectNodeById(node.id, true); // Still select the node
      }
    } catch (error) {
      console.error(`[onClick] Error fetching/processing data for ${node.id}:`, error);
      const nodeLabel = node.label || getLabelFromUri(node.id);
      showBanner(`Error al intentar expandir el nodo "${nodeLabel}"`, 'warning');
    }
  } else {
      // Verificar si el nodo es una instancia para mostrar el menú contextual
      console.log(`[onClick] Node ${node.id} was NOT in expandableClassIds. Checking if it's an instance...`);
      try {
        const isInstance = await isInstanceNode(node.id);
        console.log(`[onClick] Node ${node.id} isInstance result: ${isInstance}`);
        
        if (isInstance) {
          // Mostrar el menú contextual para instancias
          console.log(`[onClick] Showing context menu for instance node: ${node.id}`);
          showContextMenu(event.clientX, event.clientY, node);
        } else {
          // Es una clase - también mostrar menú contextual
          console.log(`[onClick] Showing context menu for class node: ${node.id}`);
          showContextMenu(event.clientX, event.clientY, node);
        }
      } catch (error) {
        console.error(`Error checking if node ${node.id} is an instance:`, error);
        // Fallback: añadir a expandableClassIds
        expandableClassIds.add(node.id);
        graph.selectNodeById(node.id, true);
      }
  }

  // Ajustar la vista después de un breve retardo para permitir la estabilización
  setTimeout(() => {
    fitView(500, 0.2); // Ajustar vista con animación y padding
  }, 700);
      },
      /**
       * Manejador de evento al hacer zoom.
       */
      onZoom: () => {
        try {
          if (labels) labels.update(graph); // Actualizar etiquetas
        } catch (error) {
          console.warn('Error updating labels on zoom:', error);
        }
      },
      /**
       * Manejador de evento al pasar el mouse sobre un nodo.
       */
      onHover: (node, i, pos, event) => {
        // Removed interactive mode logic since we're removing it
      },
      /**
       * Manejador de evento en cada frame de animación.
       */
      onFrame: () => {
        try {
          if (labels) labels.update(graph); // Actualizar etiquetas continuamente
        } catch (error) {
          console.warn('Error updating labels on frame:', error);
        }
        // Context menu position is now handled by our own requestAnimationFrame loop
      },

    }
  };

  // --- 4. Instanciación e Inicialización del Grafo y Etiquetas ---
  try {
      // Verify elements exist before creating instances
      if (!canvas) {
        throw new Error('Canvas element is null or not found');
      }
      if (!labelsDiv) {
        throw new Error('Labels div element is null or not found');
      }
      
      console.log('Creating graph and labels instances...');
      console.log('Canvas:', canvas, 'Labels div:', labelsDiv);
      console.log('Global nodes count:', globalNodes.length);
      
      graph = new Graph(canvas, config); // Crear instancia de Cosmograph
      labels = new CosmosLabels(labelsDiv, globalNodes); // Crear instancia de CosmosLabels
      
      // Store instances globally for cleanup and access
      window.globalGraph = graph;
      window.globalLabels = labels;
      globalGraph = graph;
      globalLabels = labels;
      
      console.log('New graph and labels instances created');

      // Prevenir zoom con doble clic en el canvas
      canvas.addEventListener('dblclick', function(event) {
        event.preventDefault();
        event.stopPropagation();
        return false;
      }, true);

      updateGraphProperties(); // Calcular propiedades iniciales
      graph.setData(globalNodes, globalLinks); // Cargar datos iniciales en el grafo

      // Rastrear posiciones de nodos iniciales y ajustar vista
      const initialNodeIds = globalNodes.map(n => n.id);
      graph.trackNodePositionsByIds(initialNodeIds);
      graph.zoom(config.initialZoomLevel); // Aplicar zoom inicial
      fitView(0, 0.3); // Ajustar vista inicial sin animación

  } catch (error) {
      console.error("Error initializing Cosmograph or Labels:", error);
      // Display error to user?
      const errorDiv = document.createElement('div');
      errorDiv.classList.add('graph-init-error'); // Apply CSS class
      errorDiv.textContent = `Error initializing graph: ${error.message}`;
      // errorDiv.style.color = 'red'; // Removed
      // errorDiv.style.padding = '20px'; // Removed
      document.body.prepend(errorDiv); // Add error message at the top
      return; // Stop execution if graph fails to initialize
  }

  // --- 4.5. Context Menu System for Instance Nodes ---
  
  let contextMenuNode = null; // Store the node for which context menu is shown
  let contextMenuVisible = false; // Track if context menu is currently visible
  let contextMenuAnimationId = null; // Store animation frame ID for context menu updates
  
  /**
   * Animation loop to continuously update context menu position
   */
  function animateContextMenu() {
    if (contextMenuVisible && contextMenuNode) {
      updateContextMenuPosition();
      contextMenuAnimationId = requestAnimationFrame(animateContextMenu);
    } else {
      contextMenuAnimationId = null;
    }
  }

  /**
   * Shows the context menu at the specified position for the given node
   */
  async function showContextMenu(x, y, node) {
    const contextMenu = document.getElementById('context-menu');
    if (!contextMenu) {
      console.error('[showContextMenu] Context menu element not found!');
      return;
    }
    
    // Hide all labels except for selected nodes when context menu is shown
    if (labels) {
      const selectedNodeIds = graph.getSelectedNodeIds?.() || [];
      const selectedNodeIdsSet = new Set(selectedNodeIds);
      
      // Always include the context menu node itself as visible
      selectedNodeIdsSet.add(node.id);
      
      console.log(`[showContextMenu] Keeping labels visible for ${selectedNodeIdsSet.size} selected nodes:`, Array.from(selectedNodeIdsSet));
      labels.hideExcept(selectedNodeIdsSet);
    }
    
    // Reset any existing animations/classes first
    contextMenu.classList.remove('show');
    contextMenu.style.display = 'none';
    
    contextMenuNode = node;
    contextMenuVisible = true;
    
    // Check if node is an instance to show appropriate options
    try {
      const isInstance = await isInstanceNode(node.id);
      
      // Update menu visibility based on node type
      if (isInstance) {
        // Show instance options, hide class options
        contextMenu.className = 'context-menu show-instance';
        console.log(`[showContextMenu] Showing instance options for node: ${node.id}`);
      } else {
        // Show class options, hide instance options
        contextMenu.className = 'context-menu show-class';
        console.log(`[showContextMenu] Showing class options for node: ${node.id}`);
      }
      
      // Debug: Log which menu items are visible
      const visibleItems = contextMenu.querySelectorAll('.context-menu-item:not([style*="display: none"])');
      console.log(`[showContextMenu] Visible menu items:`, Array.from(visibleItems).map(item => item.getAttribute('data-action')));
      
    } catch (error) {
      console.error('Error determining node type for context menu:', error);
      // Default to class options
      contextMenu.className = 'context-menu show-class';
    }
    
    // Show the menu with animation first
    contextMenu.style.display = 'block';
    
    // Initialize position near the click coordinates
    contextMenu.style.left = `${x}px`;
    contextMenu.style.top = `${y}px`;
    
    // Force a reflow to ensure display:block is applied before animation
    contextMenu.offsetHeight;
    
    // Add show class to trigger animation
    setTimeout(() => {
      contextMenu.classList.add('show');
    }, 10);
    
    // Position the menu relative to the node after it's visible and has dimensions
    setTimeout(() => {
      updateContextMenuPosition();
      // Start the animation loop to continuously update menu position
      if (!contextMenuAnimationId) {
        animateContextMenu();
      }
    }, 50); // Small delay to ensure menu is fully rendered
    
    // Close menu when clicking outside
    setTimeout(() => {
      const handleClickOutside = (event) => {
        const contextMenu = document.getElementById('context-menu');
        if (contextMenu && !contextMenu.contains(event.target)) {
          hideContextMenu();
          document.removeEventListener('click', handleClickOutside);
        }
      };
      document.addEventListener('click', handleClickOutside, false);
    }, 100);
  }
  
  /**
   * Updates the context menu position based on the associated node's location
   */
  function updateContextMenuPosition() {
    if (!contextMenuNode || !contextMenuVisible) {
      return;
    }
    
    const contextMenu = document.getElementById('context-menu');
    if (!contextMenu) {
      return;
    }
    
    try {
      // Get the tracked node positions from the graph
      const trackedPositions = graph.getTrackedNodePositionsMap();
      
      const nodePosition = trackedPositions.get(contextMenuNode.id);
      
      if (!nodePosition) {
        // Node is not tracked - this shouldn't happen since all nodes should be 
        // tracked from initialization, but handle gracefully
        console.warn(`[updateContextMenuPosition] Node ${contextMenuNode.id} is not being tracked.`);
        return;
      }
      
      // Convert from graph space to screen coordinates
      const screenPosition = graph.spaceToScreenPosition([nodePosition[0], nodePosition[1]]);
      const nodeRadius = graph.spaceToScreenRadius(graph.getNodeRadiusById(contextMenuNode.id) || 4);
      
      // console.log(`[updateContextMenuPosition] Node screen position:`, screenPosition, `Node radius:`, nodeRadius);
      
      // Calculate menu position with offset to avoid overlapping the node
      let menuX = screenPosition[0] + nodeRadius + 10; // 10px padding from node edge
      let menuY = screenPosition[1] - nodeRadius;
      
      // Get menu dimensions for boundary checking
      const menuRect = contextMenu.getBoundingClientRect();
      const menuWidth = menuRect.width || 200; // fallback width
      const menuHeight = menuRect.height || 100; // fallback height
      
      //console.log(`[updateContextMenuPosition] Menu dimensions:`, { width: menuWidth, height: menuHeight });
      //console.log(`[updateContextMenuPosition] Initial position:`, { x: menuX, y: menuY });
      
      // Ensure menu stays within viewport boundaries
      const viewport = {
        width: window.innerWidth,
        height: window.innerHeight
      };
      
      // Adjust horizontal position if menu would go off-screen
      if (menuX + menuWidth > viewport.width) {
        // Position menu to the left of the node instead
        menuX = screenPosition[0] - nodeRadius - menuWidth - 10;
        // console.log(`[updateContextMenuPosition] Adjusted X to left:`, menuX);
      }
      
      // Ensure menu doesn't go off the left edge
      if (menuX < 0) {
        menuX = 10; // Minimum padding from left edge
        //console.log(`[updateContextMenuPosition] Adjusted X to min:`, menuX);
      }
      
      // Adjust vertical position if menu would go off-screen
      if (menuY + menuHeight > viewport.height) {
        menuY = viewport.height - menuHeight - 10; // Position near bottom with padding
        //console.log(`[updateContextMenuPosition] Adjusted Y to bottom:`, menuY);
      }
      
      // Ensure menu doesn't go off the top edge
      if (menuY < 0) {
        menuY = 10; // Minimum padding from top edge
        //console.log(`[updateContextMenuPosition] Adjusted Y to top:`, menuY);
      }
      
      // Apply the calculated position
      const newLeft = Math.round(menuX);
      const newTop = Math.round(menuY);
      
      // console.log(`[updateContextMenuPosition] Final position:`, { left: newLeft, top: newTop });
      
      // Only update if position actually changed to avoid unnecessary DOM updates
      const currentLeft = parseInt(contextMenu.style.left) || 0;
      const currentTop = parseInt(contextMenu.style.top) || 0;
      
      if (Math.abs(currentLeft - newLeft) > 1 || Math.abs(currentTop - newTop) > 1) {
        contextMenu.style.left = `${newLeft}px`;
        contextMenu.style.top = `${newTop}px`;
        //console.log(`[updateContextMenuPosition] Position updated to:`, { left: newLeft, top: newTop });
      }
      
    } catch (error) {
      console.error('[updateContextMenuPosition] Error updating menu position:', error);
    }
  }

  /**
   * Hides the context menu
   */
  function hideContextMenu() {
    const contextMenu = document.getElementById('context-menu');
    if (!contextMenu) return;
    
    contextMenu.classList.remove('show');
    
    // Stop the animation loop
    if (contextMenuAnimationId) {
      cancelAnimationFrame(contextMenuAnimationId);
      contextMenuAnimationId = null;
    }
    
    // Reset menu state - hide all tables and show options
    const objectTableContainer = document.getElementById('object-assertions-table-container');
    const dataTableContainer = document.getElementById('data-properties-table-container');
    const annotationsTableContainer = document.getElementById('annotations-table-container');
    const menuOptions = document.getElementById('context-menu-options');
    const backButton = document.getElementById('context-menu-back');
    const tableTitle = document.getElementById('context-menu-table-title');
    const dataTitle = document.getElementById('context-menu-data-title');
    const annotationsTitle = document.getElementById('context-menu-annotations-title');
    
    if (objectTableContainer) objectTableContainer.style.display = 'none';
    if (dataTableContainer) dataTableContainer.style.display = 'none';
    if (annotationsTableContainer) annotationsTableContainer.style.display = 'none';
    if (menuOptions) menuOptions.style.display = 'block';
    if (backButton) backButton.style.display = 'none';
    if (tableTitle) tableTitle.style.display = 'none';
    if (dataTitle) dataTitle.style.display = 'none';
    if (annotationsTitle) annotationsTitle.style.display = 'none';
    
    // Wait for animation to complete before hiding
    setTimeout(() => {
      contextMenu.style.display = 'none';
      // Reset context menu class to default
      contextMenu.className = 'context-menu';
    }, 250);
    
    // Show labels again when context menu is hidden
    if (labels) {
      labels.show();
    }
    
    contextMenuNode = null;
    contextMenuVisible = false;
  }
  
  /**
   * Expands data properties for the given instance node - shows table instead of expanding graph
   */
  async function expandInstanceDataProperties(instanceNode) {
    try {
      console.log(`[expandInstanceDataProperties] Showing data properties table for instance: ${instanceNode.id}`);
      
      // Get the table container and elements
      const tableContainer = document.getElementById('data-properties-table-container');
      const tableBody = document.getElementById('data-properties-tbody');
      
      if (!tableContainer || !tableBody) {
        console.error('[expandInstanceDataProperties] Table elements not found!');
        return;
      }
      
      // Show loading indicator
      tableBody.innerHTML = '<tr><td colspan="2" style="text-align: center;">Loading data properties...</td></tr>';
      
      // Hide menu options and show table with back button and title
      const menuOptions = document.getElementById('context-menu-options');
      const backButton = document.getElementById('context-menu-back');
      const dataTitle = document.getElementById('context-menu-data-title');
      
      if (menuOptions) menuOptions.style.display = 'none';
      if (backButton) backButton.style.display = 'flex';
      if (dataTitle) dataTitle.style.display = 'block';
      tableContainer.style.display = 'block';
      
      // Fetch data properties for this instance
      console.log('[expandInstanceDataProperties] About to call fetchInstanceDataProperties...');
      const dataPropertiesData = await fetchInstanceDataProperties(instanceNode.id);
      
      console.log(`[expandInstanceDataProperties] Received data properties:`, dataPropertiesData);
      
      // Clear loading indicator
      tableBody.innerHTML = '';
      
      const properties = [];
      
      // Process literals (data properties)
      if (dataPropertiesData.literals && dataPropertiesData.literals[instanceNode.id]) {
        const instanceLiterals = dataPropertiesData.literals[instanceNode.id];
        for (const propertyUri in instanceLiterals) {
          const propertyName = getLabelFromUri(propertyUri);
          const values = instanceLiterals[propertyUri];
          
          values.forEach(value => {
            properties.push({
              property: propertyName,
              value: value
            });
          });
        }
      }
      
      console.log(`[expandInstanceDataProperties] Processed properties:`, properties);
      
      if (properties.length > 0) {
        // Populate table with data properties
        properties.forEach(prop => {
          const row = document.createElement('tr');
          row.innerHTML = `
            <td>${prop.property}</td>
            <td>${prop.value}</td>
          `;
          tableBody.appendChild(row);
        });
      } else {
        // No data properties found
        tableBody.innerHTML = '<tr><td colspan="2" class="no-assertions-message">No data properties found for this entity</td></tr>';
      }

    } catch (error) {
      console.error(`[expandInstanceDataProperties] Error showing data properties for ${instanceNode.id}:`, error);
      const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
      showBanner(`Error loading data properties for "${nodeLabel}"`, 'warning');
      
      // Hide table container and title in case of error
      const tableContainer = document.getElementById('data-properties-table-container');
      const dataTitle = document.getElementById('context-menu-data-title');
      if (tableContainer) {
        tableContainer.style.display = 'none';
      }
      if (dataTitle) {
        dataTitle.style.display = 'none';
      }
    }
  }
  
  /**
   * Shows annotations for an instance in a table format inside the context menu
   */
  async function expandInstanceAnnotations(instanceNode) {
    try {
      console.log(`[expandInstanceAnnotations] Showing annotations for instance: ${instanceNode.id}`);
      
      // Get elements
      const menuOptions = document.getElementById('context-menu-options');
      const tableContainer = document.getElementById('annotations-table-container');
      const tableBody = document.getElementById('annotations-tbody');
      const backButton = document.getElementById('context-menu-back');
      const annotationsTitle = document.getElementById('context-menu-annotations-title');
      
      if (!tableContainer || !tableBody || !backButton || !annotationsTitle) {
        console.error('[expandInstanceAnnotations] Required DOM elements not found');
        return;
      }
      
      // Show loading state
      tableBody.innerHTML = '<tr><td colspan="2" style="text-align: center; padding: 20px; color: #888;">Loading annotations...</td></tr>';
      
      // Hide menu options and show table with back button and title
      if (menuOptions) menuOptions.style.display = 'none';
      if (backButton) backButton.style.display = 'flex';
      if (annotationsTitle) annotationsTitle.style.display = 'block';
      tableContainer.style.display = 'block';
      
      // Fetch annotations for this instance
      console.log('[expandInstanceAnnotations] About to call fetchNodeAnnotations...');
      const annotationData = await fetchNodeAnnotations(instanceNode.id);
      
      console.log(`[expandInstanceAnnotations] Received annotations:`, annotationData);
      console.log(`[expandInstanceAnnotations] Annotation count:`, annotationData.annotations ? annotationData.annotations.length : 0);
      
      // Clear loading indicator
      tableBody.innerHTML = '';
      
      const annotations = [];
      
      // Process annotations
      if (annotationData.annotations && annotationData.annotations.length > 0) {
        annotationData.annotations.forEach(annotation => {
          const propertyName = annotation.propertyName || getLabelFromUri(annotation.property);
          
          annotations.push({
            property: propertyName,
            value: annotation.value,
            valueType: annotation.valueType,
            language: annotation.language
          });
        });
      }
      
      console.log(`[expandInstanceAnnotations] Processed annotations:`, annotations);
      
      if (annotations.length > 0) {
        // Populate table with annotations
        annotations.forEach(annotation => {
          const row = document.createElement('tr');
          
          // Property name cell
          const propertyCell = document.createElement('td');
          propertyCell.innerHTML = `<span class="annotation-property-name">${annotation.property}</span>`;
          row.appendChild(propertyCell);
          
          // Value cell - handle different types and long text
          const valueCell = document.createElement('td');
          let valueContent = '';
          
          if (annotation.valueType === 'uri') {
            // For URI values, show as clickable link
            const shortUri = annotation.value.split(/[#/]/).pop() || annotation.value;
            valueContent = `<span class="annotation-uri-value" title="${annotation.value}">${shortUri}</span>`;
          } else {
            // For literal values, handle as potentially long text
            let value = annotation.value;
            
            // Check if it's a long description (more than 80 characters or contains line breaks)
            const isLongText = value.length > 80 || value.includes('\n') || value.includes('\\n');
            
            if (isLongText) {
              // Format as paragraph with proper line breaks
              value = value.replace(/\\n/g, '\n').trim();
              valueContent = `<div class="annotation-value long-text">${value}</div>`;
            } else {
              valueContent = `<span class="annotation-value">${value}</span>`;
            }
            
            // Add language indicator if present
            if (annotation.language) {
              valueContent += `<span class="annotation-language">(${annotation.language})</span>`;
            }
          }
          
          valueCell.innerHTML = valueContent;
          row.appendChild(valueCell);
          
          tableBody.appendChild(row);
        });
      } else {
        // No annotations found
        tableBody.innerHTML = '<tr><td colspan="2" class="no-assertions-message">No annotations found for this entity</td></tr>';
      }
      
    } catch (error) {
      console.error(`[expandInstanceAnnotations] Error showing annotations for ${instanceNode.id}:`, error);
      const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
      showBanner(`Error loading annotations for "${nodeLabel}"`, 'warning');
      
      // Hide table container and title in case of error
      const tableContainer = document.getElementById('annotations-table-container');
      const annotationsTitle = document.getElementById('context-menu-annotations-title');
      if (tableContainer) {
        tableContainer.style.display = 'none';
      }
      if (annotationsTitle) {
        annotationsTitle.style.display = 'none';
      }
    }
  }
  
  /**
   * Expands data properties in the graph (original functionality)
   */
  async function expandDataPropertiesInGraph(instanceNode, dataPropertiesData) {
    try {
      console.log(`[expandDataPropertiesInGraph] Expanding data properties in graph for instance: ${instanceNode.id}`);
      
      // Store current pause state
      const wasPausedBeforeExpansion = isPaused;
      
      // Filter out any invalid data
      const filteredData = filterOutNamedIndividual(
        dataPropertiesData.nodes || [], 
        dataPropertiesData.edges || []
      );
      
      console.log(`[expandDataPropertiesInGraph] Received data properties:`, {
        nodes: filteredData.nodes.length,
        edges: filteredData.edges.length,
        literals: Object.keys(dataPropertiesData.literals || {}).length
      });
      
      if (filteredData.nodes.length > 0 || filteredData.edges.length > 0 || 
          Object.keys(dataPropertiesData.literals || {}).length > 0) {
        
        // Save current state before expansion
        saveCurrentState();
        
        // Get existing node IDs to avoid duplicates
        const existingNodeIds = new Set(globalNodes.map(n => n.id));
        const newNodes = filteredData.nodes.filter(n => 
          n && n.id && String(n.id).toLowerCase() !== 'null' && !existingNodeIds.has(n.id)
        );
        
        // Add new nodes to the graph
        if (newNodes.length > 0) {
          newNodes.forEach(node => {
            if (!node.label) {
              node.label = getLabelFromUri(node.id);
            }
            expandableClassIds.add(node.id);
          });
          
          globalNodes = [...globalNodes, ...newNodes];
          nodes = globalNodes;
        }
        
        // Add new edges
        if (filteredData.edges.length > 0) {
          globalLinks = [...globalLinks, ...filteredData.edges];
          links = globalLinks;
        }
        
        // Merge literals
        if (dataPropertiesData.literals) {
          for (const sourceId in dataPropertiesData.literals) {
            if (!literals[sourceId]) literals[sourceId] = {};
            for (const linkId in dataPropertiesData.literals[sourceId]) {
              if (!literals[sourceId][linkId]) literals[sourceId][linkId] = [];
              literals[sourceId][linkId] = [...new Set([...literals[sourceId][linkId], ...dataPropertiesData.literals[sourceId][linkId]])];
            }
          }
        }
        
        // Update graph
        updateGraphProperties();
        graph.setData(nodes, links);
        
        const nodeIds = nodes.map(n => n.id);
        graph.trackNodePositionsByIds(nodeIds);
        
        if (labels) labels.updateNodeMap(nodes);
        graph.selectNodesByIds(nodeIds, false);
        
        // Maintain pause state
        if (wasPausedBeforeExpansion) {
          graph.pause();
          if (pauseButton) pauseButton.textContent = "Resume";
        } else {
          graph.start();
          if (pauseButton) pauseButton.textContent = "Pause";
        }
        
        // Show success message
        const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
        const totalNewItems = newNodes.length + filteredData.edges.length + Object.keys(dataPropertiesData.literals || {}).length;
        showBanner(`Expanded ${totalNewItems} data properties for "${nodeLabel}"`, 'success');
        
      } else {
        const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
        showBanner(`No data properties found for "${nodeLabel}"`, 'info');
      }
      
    } catch (error) {
      console.error(`[expandDataPropertiesInGraph] Error expanding data properties for ${instanceNode.id}:`, error);
      const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
      showBanner(`Error expanding data properties for "${nodeLabel}"`, 'warning');
    }
  }
  
  /**
   * Expands object assertions for the given instance node
   */
  async function expandInstanceObjectAssertions(instanceNode) {
    try {
      console.log(`[expandInstanceObjectAssertions] Starting expansion for instance: ${instanceNode.id}`);
      console.log(`[expandInstanceObjectAssertions] Instance node:`, instanceNode);
      
      // Get the context menu and table container
      const contextMenu = document.getElementById('context-menu');
      const tableContainer = document.getElementById('object-assertions-table-container');
      const tableBody = document.getElementById('object-assertions-tbody');
      
      console.log('[expandInstanceObjectAssertions] DOM elements found:', {
        contextMenu: !!contextMenu,
        tableContainer: !!tableContainer,
        tableBody: !!tableBody
      });
      
      if (!tableContainer || !tableBody) {
        console.error('[expandInstanceObjectAssertions] Required DOM elements not found');
        showBanner('Error: Table elements not found', 'error');
        return;
      }
      
      // Clear previous content
      tableBody.innerHTML = '';
      
      // Show loading indicator
      tableBody.innerHTML = '<tr><td colspan="2" style="text-align: center;">Loading assertions...</td></tr>';
      
      // Hide menu options and show table with back button and title
      const menuOptions = document.getElementById('context-menu-options');
      const backButton = document.getElementById('context-menu-back');
      const tableTitle = document.getElementById('context-menu-table-title');
      
      if (menuOptions) menuOptions.style.display = 'none';
      if (backButton) backButton.style.display = 'flex';
      if (tableTitle) tableTitle.style.display = 'block';
      tableContainer.style.display = 'block';
      
      // Fetch object assertions for this instance
      console.log('[expandInstanceObjectAssertions] About to call fetchInstanceObjectAssertions...');
      const objectAssertionsData = await fetchInstanceObjectAssertions(instanceNode.id);
      
      console.log(`[expandInstanceObjectAssertions] Received object assertions:`, objectAssertionsData);
      
      // Clear loading indicator
      tableBody.innerHTML = '';
      
      // Extract object assertions from the response
      const assertions = [];
      
      // Process edges to get property-target pairs
      if (objectAssertionsData.edges && objectAssertionsData.edges.length > 0) {
        objectAssertionsData.edges.forEach(edge => {
          // Skip rdf:type relationships and label properties
          if (edge.link !== 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' && 
              edge.link !== 'http://www.w3.org/2000/01/rdf-schema#label') {
            
            // Find target node to get its label
            let targetLabel = '';
            if (objectAssertionsData.nodes) {
              const targetNode = objectAssertionsData.nodes.find(node => node.id === edge.target);
              if (targetNode && targetNode.label) {
                targetLabel = targetNode.label;
              }
            }
            
            // Extract property name from URI
            const propertyName = edge.link.split(/[#/]/).pop();
            
            // Ensure we always get a clean name for the target
            let targetName = targetLabel;
            
            // If targetLabel is empty OR it looks like a URI, extract the meaningful part
            if (!targetName || targetName.startsWith('http://') || targetName.startsWith('https://')) {
              // Extract the last meaningful part of the URI
              const uriToProcess = targetName || edge.target;
              const uriParts = uriToProcess.split(/[#/]/);
              // Get the last non-empty part
              for (let i = uriParts.length - 1; i >= 0; i--) {
                if (uriParts[i] && uriParts[i].trim() !== '') {
                  targetName = uriParts[i];
                  break;
                }
              }
              // Fallback if no meaningful part found
              if (!targetName) {
                targetName = 'Unknown';
              }
            }
            
            console.log(`[expandInstanceObjectAssertions] Target name extraction:`, {
              originalTarget: edge.target,
              foundLabel: targetLabel,
              finalName: targetName
            });
            
            // Add to assertions array
            assertions.push({
              property: {
                uri: edge.link,
                name: propertyName
              },
              target: {
                uri: edge.target,
                name: targetName
              }
            });
          }
        });
      }
      
      // If we have assertions, display them in the table
      if (assertions.length > 0) {
        assertions.forEach(assertion => {
          const row = document.createElement('tr');
          
          // Property cell
          const propertyCell = document.createElement('td');
          const propertySpan = document.createElement('span');
          propertySpan.className = 'property-name';
          propertySpan.title = assertion.property.uri;
          propertySpan.textContent = assertion.property.name;
          propertyCell.appendChild(propertySpan);
          
          // Target cell
          const targetCell = document.createElement('td');
          const targetSpan = document.createElement('span');
          targetSpan.className = 'target-name';
          targetSpan.title = assertion.target.uri;
          // Only show the name, not the URI
          targetSpan.textContent = assertion.target.name;
          targetCell.appendChild(targetSpan);
          
          // Add cells to row
          row.appendChild(propertyCell);
          row.appendChild(targetCell);
          
          // Add row to table
          tableBody.appendChild(row);
        });
      } else {
        // No assertions found
        tableBody.innerHTML = '<tr><td colspan="2" class="no-assertions-message">No object assertions found for this entity</td></tr>';
      }
      
      // Setup event listeners for the table buttons
      const expandButton = document.getElementById('expand-assertions-graph');
      
      // Remove previous event listeners (if any)
      const newExpandButton = expandButton.cloneNode(true);
      expandButton.parentNode.replaceChild(newExpandButton, expandButton);
      
      // Add new event listeners
      newExpandButton.addEventListener('click', function() {
        // Only perform graph expansion if we have assertions
        if (assertions.length > 0) {
          expandNodeInGraph(instanceNode, objectAssertionsData);
          // Hide table and title, go back to menu options
          tableContainer.style.display = 'none';
          const menuOptions = document.getElementById('context-menu-options');
          const backButton = document.getElementById('context-menu-back');
          const tableTitle = document.getElementById('context-menu-table-title');
          
          if (menuOptions) menuOptions.style.display = 'block';
          if (backButton) backButton.style.display = 'none';
          if (tableTitle) tableTitle.style.display = 'none';
        } else {
          showBanner(`No object assertions found for "${instanceNode.label || getLabelFromUri(instanceNode.id)}"`, 'info');
        }
      });
      
    } catch (error) {
      console.error(`[expandInstanceObjectAssertions] Error expanding object assertions for ${instanceNode.id}:`, error);
      const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
      showBanner(`Error expanding object assertions for "${nodeLabel}"`, 'warning');
      
      // Hide table container and title in case of error
      const tableContainer = document.getElementById('object-assertions-table-container');
      const tableTitle = document.getElementById('context-menu-table-title');
      if (tableContainer) {
        tableContainer.style.display = 'none';
      }
      if (tableTitle) {
        tableTitle.style.display = 'none';
      }
    }
  }
  
  // Helper function to expand nodes in the graph (called from the table's "Expand in Graph" button)
  function expandNodeInGraph(instanceNode, objectAssertionsData) {
    try {
      // Store current pause state
      const wasPausedBeforeExpansion = isPaused;
      
      // Filter out any invalid data
      const filteredData = filterOutNamedIndividual(
        objectAssertionsData.nodes || [], 
        objectAssertionsData.edges || []
      );
      
      if (filteredData.nodes.length > 0 || filteredData.edges.length > 0) {
        
        // Save current state before expansion
        saveCurrentState();
        
        // Get existing node IDs to avoid duplicates
        const existingNodeIds = new Set(globalNodes.map(n => n.id));
        const newNodes = filteredData.nodes.filter(n => 
          n && n.id && String(n.id).toLowerCase() !== 'null' && !existingNodeIds.has(n.id)
        );
        
        // Add new nodes to the graph
        if (newNodes.length > 0) {
          newNodes.forEach(node => {
            if (!node.label) {
              node.label = getLabelFromUri(node.id);
            }
            expandableClassIds.add(node.id);
          });
          
          globalNodes = [...globalNodes, ...newNodes];
          nodes = globalNodes;
        }
        
        // Add new edges
        if (filteredData.edges.length > 0) {
          globalLinks = [...globalLinks, ...filteredData.edges];
          links = globalLinks;
        }
        
        // Merge literals
        if (objectAssertionsData.literals) {
          for (const sourceId in objectAssertionsData.literals) {
            if (!literals[sourceId]) literals[sourceId] = {};
            for (const linkId in objectAssertionsData.literals[sourceId]) {
              if (!literals[sourceId][linkId]) literals[sourceId][linkId] = [];
              literals[sourceId][linkId] = [...new Set([...literals[sourceId][linkId], ...objectAssertionsData.literals[sourceId][linkId]])];
            }
          }
        }
        
        // Update graph
        updateGraphProperties();
        graph.setData(globalNodes, globalLinks);
        
        const nodeIds = globalNodes.map(n => n.id);
        graph.trackNodePositionsByIds(nodeIds);
        
        if (labels) labels.updateNodeMap(globalNodes);
        graph.selectNodesByIds(nodeIds, false);
        
        // Maintain pause state
        if (wasPausedBeforeExpansion) {
          graph.pause();
          if (pauseButton) pauseButton.textContent = "Resume";
        } else {
          graph.start();
          if (pauseButton) pauseButton.textContent = "Pause";
        }
        
        // Adjust view
        setTimeout(() => {
          fitView(500, 0.2);
        }, 700);
        
        const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
        showBanner(`Expanded object assertions for "${nodeLabel}"`, 'success');
        
      } else {
        const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
        showBanner(`No object assertions found for "${nodeLabel}"`, 'info');
      }
    } catch (error) {
      console.error(`[expandNodeInGraph] Error expanding node in graph for ${instanceNode.id}:`, error);
      const nodeLabel = instanceNode.label || getLabelFromUri(instanceNode.id);
      showBanner(`Error expanding object assertions for "${nodeLabel}"`, 'warning');
    }
  }
  
  /**
   * Shows detailed information about a node
   */
  function showNodeInfo(node) {
    try {
      const nodeLabel = node.label || getLabelFromUri(node.id);
      const nodeType = expandableClassIds.has(node.id) ? 'Class' : 'Node';
      
      // Create info message with node details
      let infoMessage = `**${nodeLabel}**\n\n`;
      infoMessage += `**Type:** ${nodeType}\n`;
      infoMessage += `**URI:** ${node.id}\n`;
      
      // Add degree information if available
      if (totalDegreeMap && totalDegreeMap[node.id]) {
        infoMessage += `**Connections:** ${totalDegreeMap[node.id]}\n`;
      }
      
      // Check if it has subclasses or instances
      const relatedNodes = nodes.filter(n => 
        links.some(link => 
          (link.source === node.id && link.target === n.id) || 
          (link.target === node.id && link.source === n.id)
        )
      );
      
      if (relatedNodes.length > 0) {
        infoMessage += `**Related nodes:** ${relatedNodes.length}\n`;
      }
      
      showBanner(`Node Information: ${nodeLabel}`, 'info');
      
      // Also show in assistant if available
      if (typeof addAssistantMessage === 'function') {
        addAssistantMessage('Sistema', infoMessage);
      }
      
      console.log(`[showNodeInfo] Displayed information for node: ${node.id}`);
      
    } catch (error) {
      console.error(`[showNodeInfo] Error showing node info:`, error);
      showBanner('Error showing node information', 'error');
    }
  }
  
  /**
   * Reintenta la expansión para un nodo.
   * Obtiene los datos de expansión (subclases o instancias) y los muestra.
   */
  async function retryNodeExpansion(node) {
    try {
      console.log(`[retryNodeExpansion] Retrying expansion for node: ${node.id}`);
      
      let expansionData = await fetchSubclasses(node.id);
      if (!expansionData || !expansionData.nodes || expansionData.nodes.length <= 1) {
        expansionData = await fetchClassInstances(node.id);
      }
      
      const nodeLabel = node.label || getLabelFromUri(node.id);
      const successMessage = `Se expandió exitosamente "${nodeLabel}"`;
      
      // Llamar a la función núcleo para actualizar el grafo
      expandAndReplaceGraph(expansionData, successMessage);
      
    } catch (error) {
      console.error(`[retryNodeExpansion] Error retrying expansion:`, error);
      showBanner('Error al reintentar la expansión del nodo.', 'error');
    }
  }
  
  // Setup context menu event listeners (removed DOMContentLoaded wrapper)
  console.log('[Setup] Setting up context menu event listeners...');
  const contextMenu = document.getElementById('context-menu');
  console.log('[Setup] Context menu element found:', !!contextMenu);
  
  if (contextMenu) {
    console.log('[Setup] Adding click event listener to context menu');
    contextMenu.addEventListener('click', (event) => {
      console.log('[Context Menu] Click event triggered:', event.target);
      const menuItem = event.target.closest('.context-menu-item');
      console.log('[Context Menu] Menu item found:', menuItem);
      console.log('[Context Menu] Context menu node:', contextMenuNode);
      
      if (menuItem && contextMenuNode) {
        const action = menuItem.getAttribute('data-action');
        console.log('[Context Menu] Action detected:', action);
        
        switch (action) {
          case 'expand-data-properties':
            console.log('[Context Menu] Expand data properties clicked for node:', contextMenuNode);
            // Don't hide context menu immediately - let the function handle showing the table
            expandInstanceDataProperties(contextMenuNode);
            break;
          case 'expand-object-assertions':
            console.log('[Context Menu] Expand object assertions clicked for node:', contextMenuNode);
            // Don't hide context menu immediately - let the function handle showing the table
            expandInstanceObjectAssertions(contextMenuNode);
            break;
          case 'show-info':
            showNodeInfo(contextMenuNode);
            hideContextMenu();
            break;
          case 'retry-expansion':
            retryNodeExpansion(contextMenuNode);
            hideContextMenu();
            break;
          case 'show-annotations':
            expandInstanceAnnotations(contextMenuNode);
            break;
          default:
            console.warn(`[contextMenu] Unknown action: ${action}`);
            hideContextMenu();
        }
        
        // Only hide context menu for non-table actions
        if (action !== 'expand-object-assertions') {
          // hideContextMenu() is already called above for each specific case
        }
      }
    });
  } else {
    console.error('[Setup] Context menu element not found!');
  }
  
  // Debug: Check if table elements exist
  console.log('[Setup] Checking for table elements...');
  const tableContainer = document.getElementById('object-assertions-table-container');
  const tableBody = document.getElementById('object-assertions-tbody');
  const expandButton = document.getElementById('expand-assertions-graph');
  const backButton = document.getElementById('context-menu-back');
  
  console.log('[Setup] Table elements found:', {
    tableContainer: !!tableContainer,
    tableBody: !!tableBody,
    expandButton: !!expandButton,
    backButton: !!backButton
  });
  
  // Setup back button event listener
  if (backButton) {
    backButton.addEventListener('click', function() {
      console.log('[Back Button] Clicked - returning to menu options');
      // Hide all tables and titles, show menu options
      const objectTableContainer = document.getElementById('object-assertions-table-container');
      const dataTableContainer = document.getElementById('data-properties-table-container');
      const annotationsTableContainer = document.getElementById('annotations-table-container');
      if (objectTableContainer) objectTableContainer.style.display = 'none';
      if (dataTableContainer) dataTableContainer.style.display = 'none';
      if (annotationsTableContainer) annotationsTableContainer.style.display = 'none';
      
      const menuOptions = document.getElementById('context-menu-options');
      const tableTitle = document.getElementById('context-menu-table-title');
      const dataTitle = document.getElementById('context-menu-data-title');
      const annotationsTitle = document.getElementById('context-menu-annotations-title');
      if (menuOptions) menuOptions.style.display = 'block';
      if (tableTitle) tableTitle.style.display = 'none';
      if (dataTitle) dataTitle.style.display = 'none';
      if (annotationsTitle) annotationsTitle.style.display = 'none';
      backButton.style.display = 'none';
    });
  }
  
  // Setup annotations popup close button
  const annotationsCloseButton = document.getElementById('annotations-close');
  if (annotationsCloseButton) {
    annotationsCloseButton.addEventListener('click', hideAnnotationsPopup);
  }
  
  // Close annotations popup when clicking outside
  const annotationsPopup = document.getElementById('annotations-popup');
  if (annotationsPopup) {
    annotationsPopup.addEventListener('click', (event) => {
      if (event.target === annotationsPopup) {
        hideAnnotationsPopup();
      }
    });
  }


  // --- 5. Implementación del Asistente de Chat ---
  
  // Active tags management (globally available)
  let activeTags = new Set();
  
  // Abort controller for canceling requests
  let currentAbortController = null;
  
  /**
   * Manages active tags as chips above the input
   */
  function addActiveTag(tagText) {
    if (activeTags.has(tagText)) return; // Already added
    
    activeTags.add(tagText);
    updateActiveTagsUI();
  }
  
  function removeActiveTag(tagText) {
    activeTags.delete(tagText);
    updateActiveTagsUI();
  }
  
  // Make removeActiveTag globally available
  window.removeActiveTag = removeActiveTag;
  
  function updateActiveTagsUI() {
    const activeTagsArea = document.getElementById('active-tags-area');
    if (!activeTagsArea) return;
    
    // Show/hide the area based on whether there are active tags
    if (activeTags.size === 0) {
      activeTagsArea.style.display = 'none';
      activeTagsArea.innerHTML = '';
      return;
    }
    
    activeTagsArea.style.display = 'flex';
    activeTagsArea.innerHTML = '';
    
    // Create chip for each active tag
    activeTags.forEach(tagText => {
      const chip = document.createElement('div');
      chip.classList.add('tag-chip');
      chip.innerHTML = `
        <span class="tag-text">${tagText}</span>
        <span class="tag-remove" onclick="removeActiveTag('${tagText}')" title="Eliminar etiqueta">×</span>
      `;
      activeTagsArea.appendChild(chip);
    });
  }
  
  function clearActiveTags() {
    activeTags.clear();
    updateActiveTagsUI();
  }
  
  function getActiveTagsArray() {
    return Array.from(activeTags);
  }
  
  function detectAndExtractTags(text) {
    const validTags = availableTags.map(tag => tag.text);
    let cleanText = text;
    let foundTags = [];
    
    validTags.forEach(tagText => {
      // Check if the tag exists in the text
      const regex = new RegExp(`\\b${tagText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'g');
      const matches = cleanText.match(regex);
      if (matches) {
        foundTags.push(tagText);
        // Remove the tag from the text
        cleanText = cleanText.replace(regex, '').replace(/\s+/g, ' ').trim();
      }
    });
    
    return { cleanText, foundTags };
  }
  
  let assistantContainer = document.getElementById("assistant-container");
  if (assistantContainer) {
    // Get references to existing HTML elements
    const titleBar = document.getElementById("assistant-title-bar");
    const content = document.getElementById("assistant-content");
    const messagesArea = document.getElementById("assistant-output");
    const input = document.getElementById("assistant-input");
    const sendButton = document.getElementById("assistant-send");
    const toggleButton = document.getElementById("assistant-toggle");

    // Variables para gestionar el arrastre
    let isDragging = false;
    let offsetX, offsetY;
    let wasDragged = false; // Variable para detectar si realmente hubo movimiento
    let lastValidPosition = { left: 0, bottom: 0 }; // Para recordar posición entre estados
    let dragStartTime = 0; // Para medir el tiempo desde que empezó el arrastre
    let currentX = 0, currentY = 0; // Variables para tracking suave de la posición
    let rafId = null; // Para controlar el requestAnimationFrame
    
    // NEW: Variables for relative positioning during window resize
    let relativePosition = { x: 0, y: 0 }; // Store position as percentage of window size
    let lastWindowSize = { width: window.innerWidth, height: window.innerHeight };

    // NEW: Function to calculate relative position
    function updateRelativePosition() {
      const rect = assistantContainer.getBoundingClientRect();
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      
      // Calculate position as percentage of window size
      relativePosition.x = rect.left / (windowWidth - rect.width);
      relativePosition.y = rect.top / (windowHeight - rect.height);
      
      // Clamp values between 0 and 1
      relativePosition.x = Math.max(0, Math.min(1, relativePosition.x));
      relativePosition.y = Math.max(0, Math.min(1, relativePosition.y));
      
      lastWindowSize.width = windowWidth;
      lastWindowSize.height = windowHeight;
    }

    // NEW: Function to apply relative position after window resize
    function applyRelativePosition() {
      const windowWidth = window.innerWidth;
      const windowHeight = window.innerHeight;
      const chatWidth = assistantContainer.offsetWidth;
      const chatHeight = assistantContainer.offsetHeight;
      
      // Calculate new position based on relative position
      const maxX = windowWidth - chatWidth;
      const maxY = windowHeight - chatHeight;
      
      const newX = Math.max(0, Math.min(maxX, relativePosition.x * maxX));
      const newY = Math.max(0, Math.min(maxY, relativePosition.y * maxY));
      
      // Update position
      currentX = newX;
      currentY = newY;
      assistantContainer.style.left = `${newX}px`;
      assistantContainer.style.top = `${newY}px`;
      assistantContainer.style.bottom = 'auto';
      assistantContainer.style.transform = 'none';
      
      // Update last valid position
      lastValidPosition = {
        left: newX,
        top: newY
      };
    }

    // NEW: Window resize event listener
    function handleWindowResize() {
      // Don't adjust position if currently dragging
      if (isDragging) return;
      
      // Apply the relative position to the new window size
      applyRelativePosition();
    }

    // Initialize relative position on first load
    setTimeout(() => {
      updateRelativePosition();
    }, 100);

    // Add resize event listener
    window.addEventListener('resize', handleWindowResize);

    // Función para iniciar el arrastre
    function startDrag(e) {
      // Evitar arrastrar cuando se hace clic en el botón de minimizar
      if (e.target === toggleButton) return;
      
      isDragging = true;
      wasDragged = false;
      dragStartTime = Date.now();
      
      const rect = assistantContainer.getBoundingClientRect();
      
      // Almacenar la posición actual para animaciones suaves
      currentX = rect.left;
      currentY = rect.top;
      
      // Add dragging class to disable transitions
      assistantContainer.classList.add('dragging');
      
      // Simplificamos manejo de posición, siempre usamos top/left
      if (assistantContainer.style.bottom && assistantContainer.style.bottom !== 'auto') {
        const currentBottom = parseInt(assistantContainer.style.bottom, 10) || 0;
        currentY = window.innerHeight - rect.height - currentBottom;
        assistantContainer.style.bottom = 'auto';
      }
      
      offsetX = e.clientX - currentX;
      offsetY = e.clientY - currentY;
      
      titleBar.style.cursor = 'grabbing';
      document.body.classList.add('dragging-assistant');
      
      // Cancelar cualquier animación anterior
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      
      // Habilitar manejo de eventos de arrastre
      document.addEventListener('mousemove', drag, { passive: false });
      document.addEventListener('mouseup', stopDrag);
      
      e.preventDefault();
    }

    // Función para animar el movimiento suavemente
    function updatePosition() {
      // Use direct positioning instead of transform for more predictable behavior
      assistantContainer.style.left = `${currentX}px`;
      assistantContainer.style.top = `${currentY}px`;
      assistantContainer.style.transform = 'none';
      assistantContainer.style.bottom = 'auto';
      
      // Guardar posición para el historial
      lastValidPosition = {
        left: currentX,
        top: currentY
      };
      
      rafId = null;
    }

    // Función para realizar el arrastre
    function drag(e) {
      if (!isDragging) return;
      
      let newX = e.clientX - offsetX;
      let newY = e.clientY - offsetY;
      
      const chatWidth = assistantContainer.offsetWidth;
      const chatHeight = assistantContainer.offsetHeight;

      // Clamp to window boundaries FIRST
      const maxX = window.innerWidth - chatWidth;
      const maxY = window.innerHeight - chatHeight;
      newX = Math.max(0, Math.min(newX, maxX));
      newY = Math.max(0, Math.min(newY, maxY));

      // Define forbidden zones and padding - only apply if not already clamped to edges
      const forbiddenZones = [];
      const PADDING = 20; // Pixels of space to keep from forbidden zones

      const actionContainerElement = document.querySelector('.action-container');
      if (actionContainerElement && getComputedStyle(actionContainerElement).display !== 'none') {
        forbiddenZones.push(actionContainerElement.getBoundingClientRect());
      }

      // returnButton is already a global reference to the DOM element
      if (returnButton && getComputedStyle(returnButton).display !== 'none') {
        forbiddenZones.push(returnButton.getBoundingClientRect());
      }

      // Only apply collision detection if we're not at window edges
      if (newX > 0 && newX < maxX && newY > 0 && newY < maxY) {
        for (const zone of forbiddenZones) {
          // Proposed chat window boundaries for collision check
          const chatRect = {
            left: newX,
            top: newY,
            right: newX + chatWidth,
            bottom: newY + chatHeight,
            width: chatWidth,
            height: chatHeight
          };

          // Check for overlap between chatRect and zone
          const overlaps = !(chatRect.right <= zone.left || 
                             chatRect.left >= zone.right || 
                             chatRect.bottom <= zone.top || 
                             chatRect.top >= zone.bottom);

          if (overlaps) {
            // Calculate overlap on X and Y axes
            const overlapX = Math.min(chatRect.right, zone.right) - Math.max(chatRect.left, zone.left);
            const overlapY = Math.min(chatRect.bottom, zone.bottom) - Math.max(chatRect.top, zone.top);

            // Resolve collision along the axis of least penetration
            if (overlapX < overlapY) { // Resolve horizontally
              if (chatRect.left + chatRect.width / 2 < zone.left + zone.width / 2) {
                // Chat center is to the left of zone center, push chat to be fully left of zone
                newX = Math.max(0, zone.left - chatWidth - PADDING);
              } else {
                // Chat center is to the right, push chat to be fully right of zone
                newX = Math.min(maxX, zone.right + PADDING);
              }
            } else { // Resolve vertically
              if (chatRect.top + chatRect.height / 2 < zone.top + zone.height / 2) {
                // Chat center is above zone center, push chat to be fully above zone
                newY = Math.max(0, zone.top - chatHeight - PADDING);
              } else {
                // Chat center is below, push chat to be fully below zone
                newY = Math.min(maxY, zone.bottom + PADDING);
              }
            }
            
            // Re-clamp after collision adjustment
            newX = Math.max(0, Math.min(newX, maxX));
            newY = Math.max(0, Math.min(newY, maxY));
          }
        }
      }
      
      // Detect if hubo movimiento significativo
      if (Math.abs(newX - currentX) > 2 || Math.abs(newY - currentY) > 2) {
        wasDragged = true;
      }
      
      // Only update if position actually changed to avoid unnecessary updates
      if (Math.abs(newX - currentX) > 0.5 || Math.abs(newY - currentY) > 0.5) {
        // Actualizar posición para animación suave
        currentX = newX;
        currentY = newY;
        
        // Programar actualización visual en próximo cuadro de animación
        if (!rafId) {
          rafId = requestAnimationFrame(updatePosition);
        }
      }
      
      e.preventDefault();
    }

    // Función para finalizar el arrastre
    function stopDrag(e) {
      if (!isDragging) return;
      
      isDragging = false;
      titleBar.style.cursor = 'grab';
      document.body.classList.remove('dragging-assistant');
      
      // Remove dragging class to re-enable transitions
      assistantContainer.classList.remove('dragging');
      
      // Realizar la última actualización de posición
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
      }
      
      // Ensure final position is set correctly
      assistantContainer.style.left = `${currentX}px`;
      assistantContainer.style.top = `${currentY}px`;
      assistantContainer.style.transform = 'none';
      assistantContainer.style.bottom = 'auto';
      
      // UPDATE: Calculate new relative position after drag ends
      updateRelativePosition();
      
      const dragDuration = Date.now() - dragStartTime;
      
      if (dragDuration > 150) { // Reduced threshold
        wasDragged = true;
      }
      
      document.removeEventListener('mousemove', drag);
      document.removeEventListener('mouseup', stopDrag);
      
      e.stopPropagation();
    }

    titleBar.addEventListener('mousedown', startDrag);

    // Setup toggle functionality using existing button from HTML
    toggleButton.onclick = () => {
      const content = document.getElementById("assistant-content");
      content.style.display = "none"; // Hide content
      
      // Get current window position and calculate bottom-left corner
      const currentRect = assistantContainer.getBoundingClientRect();
      const bubbleLeft = currentRect.left; // Keep same left position
      const bubbleTop = currentRect.bottom - 62; // Position at bottom of window (60px is bubble height)
      
      // Minimize to bubble at bottom-left corner of current window
      assistantContainer.style.width = "60px";
      assistantContainer.style.height = "60px";
      assistantContainer.style.borderRadius = "50%";
      assistantContainer.style.overflow = "hidden";
      
      // Position at bottom-left corner of the original window
      assistantContainer.style.left = `${bubbleLeft}px`;
      assistantContainer.style.top = `${bubbleTop}px`;
      assistantContainer.style.bottom = "auto"; // Reset bottom positioning
      assistantContainer.style.transform = "none"; // Reset any transforms
      
      assistantContainer.style.cursor = "pointer"; 

      toggleButton.style.display = "none";

      // Update tracking variables for the new bubble position
      currentX = bubbleLeft;
      currentY = bubbleTop;
      lastValidPosition = {
        left: bubbleLeft,
        top: bubbleTop
      };

      // Update relative position for the current bubble location
      updateRelativePosition();

      const currentTitleBar = document.querySelector('#assistant-container .assistant-title-bar');
      if (currentTitleBar) {
        currentTitleBar.style.justifyContent = "center";
        currentTitleBar.style.padding = "0";
        currentTitleBar.style.height = "100%";
        currentTitleBar.style.cursor = "pointer";
        
        let logoImg = document.getElementById("assistant-logo");
        if (!logoImg) {
          logoImg = document.createElement("img");
          logoImg.id = "assistant-logo"; // CSS will style it
          logoImg.src = logoSrc;
          currentTitleBar.insertBefore(logoImg, currentTitleBar.firstChild);
        }
        logoImg.style.display = "block"; // Show logo

        // Hide the title text (span element)
        const titleSpan = currentTitleBar.querySelector('span');
        if (titleSpan) {
          titleSpan.style.display = "none";
        }

        // Also hide any text nodes for backwards compatibility
        Array.from(currentTitleBar.childNodes).forEach(node => {
          if (node.nodeType === Node.TEXT_NODE) {
            node._originalData = node.data;
            node.data = "";
          }
        });
      }
    };
    
    assistantContainer.addEventListener("click", (event) => {
      const content = document.getElementById("assistant-content");
      const isBubble = content.style.display === "none";
      const currentTitleBar = document.querySelector('#assistant-container .assistant-title-bar');
      
      const validClickTarget = isBubble || 
                              (event.target !== currentTitleBar && !currentTitleBar.contains(event.target));
      
      if (isBubble && !wasDragged && event.target !== toggleButton && validClickTarget) {
        maximizeAssistant();
        event.stopPropagation();
      }
      
      setTimeout(() => {
        wasDragged = false;
      }, 10);
    });

    // Función para maximizar el asistente (Modificamos para mantener coherencia con el nuevo sistema)
    function maximizeAssistant() {
      const content = document.getElementById("assistant-content");
      if (content.style.display === "none") {
        // Add expanding class for smooth animation
        assistantContainer.classList.add('expanding');
        
        // Get current bubble position instead of hardcoded values
        const bubbleRect = assistantContainer.getBoundingClientRect();
        const bubbleX = bubbleRect.left;
        const bubbleY = bubbleRect.top;
        
        // Calculate a good position for the expanded window
        const expandedWidth = 380; // Match CSS width
        const expandedHeight = 450; // Match CSS height
        
        // Try to position the expanded window near the bubble but ensure it fits on screen
        let newX = bubbleX;
        let newY = Math.max(0, bubbleY - expandedHeight + 60); // Position so it doesn't go off top
        
        // Ensure it doesn't go off screen
        const maxX = window.innerWidth - expandedWidth;
        const maxY = window.innerHeight - expandedHeight;
        newX = Math.max(0, Math.min(newX, maxX));
        newY = Math.max(0, Math.min(newY, maxY));
        
        // Start the animation from current bubble position and size
        assistantContainer.style.width = "60px";
        assistantContainer.style.height = "60px";
        assistantContainer.style.left = `${bubbleX}px`;
        assistantContainer.style.top = `${bubbleY}px`;
        assistantContainer.style.bottom = "auto";
        assistantContainer.style.borderRadius = "50%";
        
        // Force a reflow to ensure the starting state is applied
        assistantContainer.offsetHeight;
        
        // Now animate to the expanded state
        setTimeout(() => {
          assistantContainer.style.width = `${expandedWidth}px`;
          assistantContainer.style.height = `${expandedHeight}px`;
          assistantContainer.style.borderRadius = "12px";
          assistantContainer.style.left = `${newX}px`;
          assistantContainer.style.top = `${newY}px`;
          assistantContainer.style.transform = 'none';
          
          // Update tracking variables to match the final position
          currentX = newX;
          currentY = newY;
          lastValidPosition = {
            left: newX,
            top: newY
          };
        }, 50);
        
        assistantContainer.style.cursor = "default";
        content.style.display = "flex"; // Show content

        toggleButton.style.display = "block";

        // Remove expanding class after animation completes
        setTimeout(() => {
          assistantContainer.classList.remove('expanding');
          // UPDATE: Calculate relative position after maximizing
          updateRelativePosition();
        }, 450); // Slightly longer than the CSS transition

        const currentTitleBar = document.querySelector('#assistant-container .assistant-title-bar');
        if (currentTitleBar) {
          currentTitleBar.style.justifyContent = "space-between"; // Restore style
          currentTitleBar.style.padding = "16px 20px"; // Restore original padding
          currentTitleBar.style.height = "auto"; // Restore style
          currentTitleBar.style.cursor = "grab";
          
          const logoImg = document.getElementById("assistant-logo");
          if (logoImg) logoImg.style.display = "none"; // Hide logo

          // Show the title text (span element)
          const titleSpan = currentTitleBar.querySelector('span');
          if (titleSpan) {
            titleSpan.style.display = "inline";
          }

          // Also restore any text nodes for backwards compatibility
          Array.from(currentTitleBar.childNodes).forEach(node => {
            if (node.nodeType === Node.TEXT_NODE && node._originalData) {
              node.data = node._originalData;
            }
          });
        }
      }
    }    // Initialize the assistant UI
    displayMessage("Asistente", `### 👋 ¡Hola! Bienvenido

  Puedes **preguntarme cualquier cosa** sobre el grafo actual o usar \`@\` para **comandos especiales**.

  `);
    
    // Setup tag system event handlers
    setupTagSystem();
    
    // Setup autocomplete system (define function first, then call)
    initializeAutocomplete();
  }

  /**
   * Formatea el texto destacando las etiquetas @xxxx.
   * @param {string} text - El texto a formatear.
   * @param {boolean} onlyValidTags - Si true, solo formatea etiquetas válidas completas.
   * @returns {string} - El texto con las etiquetas formateadas.
   */
  function formatTagsInText(text, onlyValidTags = false) {
    if (onlyValidTags) {
      // Solo formatear etiquetas válidas completas
      const validTags = availableTags.map(tag => tag.text);
      let formattedText = text;
      
      validTags.forEach(tagText => {
        // Simple case-sensitive replacement
        const regex = new RegExp(`(${tagText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'g');
        const replacement = `<span class="tag-highlight">${tagText}</span>`;
        formattedText = formattedText.replace(regex, replacement);
      });
      
      return formattedText;
    } else {
      // Regex para detectar etiquetas @xxxx (letras, números, guiones)
      const tagRegex = /@([a-zA-Z][a-zA-Z0-9\-_]*)/g;
      return text.replace(tagRegex, '<span class="tag-highlight">@$1</span>');
    }
  }

  /**
   * Configura el sistema de autocompletado para el input del chat.
   */
  function initializeAutocomplete() {
    const input = document.getElementById("assistant-input");
    const inputWrapper = input.parentElement; // .input-wrapper
    
    // Crear contenedor de autocompletado si no existe
    let autocompleteContainer = inputWrapper.querySelector('.autocomplete-container');
    if (!autocompleteContainer) {
      autocompleteContainer = document.createElement('div');
      autocompleteContainer.classList.add('autocomplete-container');
      
      // Wrap the input in the autocomplete container
      inputWrapper.insertBefore(autocompleteContainer, input);
      autocompleteContainer.appendChild(input);
    }
    
    // Crear dropdown de autocompletado
    let dropdown = autocompleteContainer.querySelector('.autocomplete-dropdown');
    if (!dropdown) {
      dropdown = document.createElement('div');
      dropdown.classList.add('autocomplete-dropdown');
      autocompleteContainer.appendChild(dropdown);
    }
    
    let selectedIndex = -1;
    let filteredTags = [];
    
    /**
     * Muestra las opciones de autocompletado filtradas.
     */
    function showAutocomplete(filter = '') {
      filteredTags = availableTags.filter(tag => 
        tag.text.toLowerCase().includes(filter.toLowerCase())
      );
      
      dropdown.innerHTML = '';
      
      if (filteredTags.length === 0) {
        dropdown.classList.remove('show');
        return;
      }
      
      filteredTags.forEach((tag, index) => {
        const item = document.createElement('div');
        item.classList.add('autocomplete-item');
        if (!tag.enabled) {
          item.classList.add('disabled');
        }
        
        item.innerHTML = `
          <span class="autocomplete-icon">${tag.icon}</span>
          <div class="autocomplete-title">${tag.text}</div>
        `;
        
        item.addEventListener('click', () => {
          if (tag.enabled) {
            insertTag(tag.text);
          }
        });
        
        dropdown.appendChild(item);
      });
      
      dropdown.classList.add('show');
      selectedIndex = -1;
    }
    
    /**
     * Oculta el dropdown de autocompletado.
     */
    function hideAutocomplete() {
      dropdown.classList.remove('show');
      selectedIndex = -1;
    }
    
    /**
     * Actualiza la selección visual en el dropdown.
     */
    function updateSelection() {
      const items = dropdown.querySelectorAll('.autocomplete-item');
      const enabledItems = dropdown.querySelectorAll('.autocomplete-item:not(.disabled)');
      
      items.forEach((item, index) => {
        item.classList.remove('selected');
      });
      
      if (selectedIndex >= 0 && selectedIndex < enabledItems.length) {
        enabledItems[selectedIndex].classList.add('selected');
      }
    }
    
    /**
     * Inserta una etiqueta como chip activo.
     */
    function insertTag(tagText) {
      const cursorPos = input.selectionStart;
      const textBefore = input.value.substring(0, cursorPos);
      const textAfter = input.value.substring(cursorPos);
      
      // Buscar el último @ antes del cursor
      const lastAtIndex = textBefore.lastIndexOf('@');
      
      if (lastAtIndex !== -1) {
        // Eliminar desde el @ hasta el cursor
        const newTextBefore = textBefore.substring(0, lastAtIndex);
        input.value = (newTextBefore + textAfter).trim();
        
        // Posicionar el cursor donde estaba el @
        const newCursorPos = newTextBefore.length;
        input.setSelectionRange(newCursorPos, newCursorPos);
      }
      
      // Agregar etiqueta como chip activo
      addActiveTag(tagText);
      
      hideAutocomplete();
      input.focus();
    }
    
    // Event listeners
    input.addEventListener('input', (e) => {
      const cursorPos = input.selectionStart;
      const textBefore = input.value.substring(0, cursorPos);
      
      // Verificar si se escribió una etiqueta completa seguida de espacio
      const words = textBefore.split(' ');
      const lastWord = words[words.length - 1];
      
      // Si la última palabra termina con espacio, verificar la palabra anterior
      if (textBefore.endsWith(' ') && words.length >= 2) {
        const previousWord = words[words.length - 2];
        if (previousWord.startsWith('@')) {
          // Buscar si la etiqueta existe en nuestras opciones disponibles
          const matchingTag = availableTags.find(tag => 
            tag.text.toLowerCase() === previousWord.toLowerCase()
          );
          
          if (matchingTag) {
            if (matchingTag.enabled) {
              // Etiqueta válida y habilitada - agregar como chip activo
              const textWithoutTag = words.slice(0, -2).concat([words[words.length - 1]]).join(' ');
              input.value = textWithoutTag;
              
              addActiveTag(matchingTag.text);
              
              // Posicionar el cursor al final
              input.setSelectionRange(textWithoutTag.length, textWithoutTag.length);
              
              hideAutocomplete();
              return;
            } else {
              // Etiqueta existe pero está en desarrollo
              showBanner(`${matchingTag.text} está en desarrollo`, 'dev');
              
              // Eliminar la etiqueta del texto
              const textWithoutTag = words.slice(0, -2).concat([words[words.length - 1]]).join(' ');
              input.value = textWithoutTag;
              input.setSelectionRange(textWithoutTag.length, textWithoutTag.length);
              
              hideAutocomplete();
              return;
            }
          } else {
            // Etiqueta no existe - mostrar error
            showBanner(`${previousWord} no es una etiqueta válida`, 'error');
            
            // Eliminar la etiqueta del texto
            const textWithoutTag = words.slice(0, -2).concat([words[words.length - 1]]).join(' ');
            input.value = textWithoutTag;
            input.setSelectionRange(textWithoutTag.length, textWithoutTag.length);
            
            hideAutocomplete();
            return;
          }
        }
      }
      
      // Lógica existente para autocompletado
      const lastAtIndex = textBefore.lastIndexOf('@');
      
      if (lastAtIndex !== -1) {
        // Verificar que no hay espacios entre @ y el cursor
        const textAfterAt = textBefore.substring(lastAtIndex + 1);
        
        if (!textAfterAt.includes(' ')) {
          // Mostrar autocompletado con filtro
          showAutocomplete(textAfterAt);
          return;
        }
      }
      
      hideAutocomplete();
    });
    
    input.addEventListener('keydown', (e) => {
      if (!dropdown.classList.contains('show')) return;
      
      const enabledTags = filteredTags.filter(tag => tag.enabled);
      
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        selectedIndex = Math.min(selectedIndex + 1, enabledTags.length - 1);
        updateSelection();
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        selectedIndex = Math.max(selectedIndex - 1, -1);
        updateSelection();
      } else if ((e.key === 'Enter' || e.key === 'Tab') && selectedIndex >= 0) {
        e.preventDefault();
        const selectedTag = enabledTags[selectedIndex];
        if (selectedTag) {
          insertTag(selectedTag.text);
        }
      } else if (e.key === 'Tab' && selectedIndex === -1 && enabledTags.length > 0) {
        // Si no hay selección pero hay opciones, seleccionar la primera habilitada
        e.preventDefault();
        insertTag(enabledTags[0].text);
      } else if (e.key === 'Escape') {
        e.preventDefault();
        hideAutocomplete();
      }
    });
    
    // Inicializar el sistema
    hideAutocomplete();
    
    // Ocultar autocompletado al hacer click fuera
    document.addEventListener('click', (e) => {
      if (!autocompleteContainer.contains(e.target)) {
        hideAutocomplete();
      }
    });
  }

  /**
   * Configura los manejadores de eventos para el sistema de etiquetas.
   */
  function setupTagSystem() {
    const tagButton = document.getElementById("tag-button");
    const tagMenu = document.getElementById("tag-menu");
    const assistantInput = document.getElementById("assistant-input");

    if (!tagButton || !tagMenu || !assistantInput) return;

    // Toggle tag menu
    tagButton.addEventListener("click", (e) => {
      e.stopPropagation();
      tagMenu.classList.toggle("show");
    });

    // Handle tag selection
    tagMenu.addEventListener("click", (e) => {
      const tagItem = e.target.closest(".tag-menu-item");
      if (tagItem && !tagItem.disabled) {
        const tag = tagItem.getAttribute("data-tag");
        insertTagIntoInput(tag, assistantInput);
        tagMenu.classList.remove("show");
      }
    });

    // Close menu when clicking outside
    document.addEventListener("click", (e) => {
      if (!tagButton.contains(e.target) && !tagMenu.contains(e.target)) {
        tagMenu.classList.remove("show");
      }
    });

    // Close menu when pressing Escape
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        tagMenu.classList.remove("show");
      }
    });
  }

  /**
   * Inserta una etiqueta como chip activo desde el menú.
   */
  function insertTagIntoInput(tag, inputElement) {
    // Simplemente agregar la etiqueta como chip activo
    addActiveTag(tag);
    inputElement.focus();
  }

  /**
   * Muestra un indicador de escritura en el chat.
   */
  function showTypingIndicator() {
    const messagesArea = document.getElementById("assistant-output");
    if (!messagesArea) return;

    // Remove any existing typing indicator first
    const existingTyping = messagesArea.querySelector('.message.typing');
    if (existingTyping) {
      existingTyping.remove();
    }

    const typingElement = document.createElement("div");
    typingElement.classList.add("message", "typing", "asistente");
    typingElement.innerHTML = `
      <strong>Asistente:</strong> 
      <div class="typing-content">
        <span class="typing-indicator">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </span>
        <button class="cancel-request-btn" onclick="cancelCurrentRequest()" title="Cancelar petición">
          <span class="cancel-icon">✕</span>
        </button>
      </div>
    `;
    
    messagesArea.appendChild(typingElement);
    messagesArea.scrollTop = messagesArea.scrollHeight;
    
    return typingElement;
  }

  /**
   * Oculta el indicador de escritura del chat.
   */
  function hideTypingIndicator() {
    const messagesArea = document.getElementById("assistant-output");
    if (!messagesArea) return;

    const typingElement = messagesArea.querySelector('.message.typing');
    if (typingElement) {
      typingElement.remove();
    }
  }

  /**
   * Cancela la petición actual al asistente.
   */
  function cancelCurrentRequest() {
    if (currentAbortController) {
      currentAbortController.abort();
      currentAbortController = null;
      
      hideTypingIndicator();
      showBanner("Petición cancelada por el usuario", 'error');
      
      // Re-enable send button
      const sendButton = document.getElementById("assistant-send");
      if (sendButton) sendButton.disabled = false;
      
      // Focus back to input
      const assistantInput = document.getElementById("assistant-input");
      if (assistantInput) assistantInput.focus();
      
      console.log("[Frontend] Request cancelled by user");
    }
  }

  // Make cancelCurrentRequest globally accessible
  window.cancelCurrentRequest = cancelCurrentRequest;

  /**
   * Muestra un mensaje en el área de chat del asistente.
   * @param {string} sender - Quién envía el mensaje ("Usuario", "Asistente", "Sistema").
   * @param {string} message - El contenido del mensaje.
   * @param {Array} tags - Etiquetas activas para el mensaje (opcional).
   */
  function displayMessage(sender, message, tags = []) {
    const messagesArea = document.getElementById("assistant-output");
    if (!messagesArea) return;

    // Hide typing indicator when displaying a new message
    hideTypingIndicator();

    const messageElement = document.createElement("div");
    messageElement.classList.add("message"); // Base class
    messageElement.classList.add(sender.toLowerCase()); 
    
    let messageContent = '';
    
    // For user messages, show tags outside the bubble
    if (sender.toLowerCase() === 'usuario' && tags && tags.length > 0) {
      // Create tags container outside the message bubble
      const tagsContainer = document.createElement("div");
      tagsContainer.classList.add("message-tags-external");
      const tagsHTML = tags.map(tag => `<span class="message-tag-chip">${tag}</span>`).join('');
      tagsContainer.innerHTML = tagsHTML;
      
      // Insert tags before the message
      messagesArea.appendChild(tagsContainer);
    } else if (tags && tags.length > 0) {
      // For assistant and system messages, keep tags above inside the bubble
      const tagsHTML = tags.map(tag => `<span class="message-tag-chip">${tag}</span>`).join('');
      messageContent += `<div class="message-tags">${tagsHTML}</div>`;
    }
    
    // Add sender name for all message types with line break for user messages
    if (sender.toLowerCase() === 'usuario') {
      messageContent += `<strong>${sender}:</strong>`;
    } else {
      messageContent += `<strong>${sender}:</strong> `;
    }
    
    // Parse markdown for assistant and system messages, but not for user messages
    let processedMessage;
    if (sender.toLowerCase() === 'usuario') {
      // For user messages, escape HTML first
      processedMessage = message
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, '<br>');
    } else {
      // For assistant and system messages, format tags first, then parse markdown
      try {
        // Apply markdown processing
        processedMessage = md.render(message);
        
        // Apply tag formatting after markdown
        const validTags = availableTags.map(tag => tag.text);
        validTags.forEach(tagText => {
          const regex = new RegExp(`(${tagText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'g');
          processedMessage = processedMessage.replace(regex, `<span class="tag-highlight">${tagText}</span>`);
        });
      } catch (error) {
        console.warn('Markdown parsing failed, falling back to plain text:', error);
        processedMessage = message.replace(/\n/g, '<br>');
        processedMessage = formatTagsInText(processedMessage);
      }
    }
    
    messageContent += processedMessage;
    messageElement.innerHTML = messageContent;
    
    messagesArea.appendChild(messageElement);
    messagesArea.scrollTop = messagesArea.scrollHeight;
    
    // Prevent default behavior for mouse events only when actually dragging
    messageElement.addEventListener('mousedown', function(event) {
      // Allow normal text selection unless we're dragging the window
      if (!isDragging) {
        event.stopPropagation(); // Don't let it bubble to container drag handlers
      }
    });
  }

  /**
   * Envía una pregunta al backend del asistente y muestra la respuesta.
   * @param {string} question - La pregunta del usuario.
   */
  async function askAssistant(question) {
    if (!question.trim()) return; 

    // No mostramos el mensaje del usuario aquí ya que se muestra desde los event listeners
    const sendButton = document.getElementById("assistant-send"); 
    if (sendButton) sendButton.disabled = true; 

    // Create new AbortController for this request
    currentAbortController = new AbortController();

    // Show typing indicator with cancel button
    showTypingIndicator();

    // Check if the message contains @Browse tag
    const containsBrowseTag = question.includes("@Browse");

    let requestBody = {
        message: question,
        containsBrowseTag: containsBrowseTag
    };

    console.log("[Frontend] Fetching current graph state to send with message...");
    const graphState = getCurrentGraphState(nodes, links);
    if (graphState) { 
        console.log(`[Frontend] Graph state obtained: ${graphState.nodes?.length || 0} nodes, ${graphState.links?.length || 0} links.`);
        requestBody.graph_data = graphState; 
    } else {
        console.warn("[Frontend] Failed to get graph state, sending message without graph_data.");
    }

    console.log("[Frontend] Sending request body:", JSON.stringify(requestBody, null, 2)); 

    try {
      const response = await fetch('http://192.168.216.102:5000/chat', { 
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: currentAbortController.signal // Add abort signal
      });

      if (!response.ok) {
         let errorMessage = `Error HTTP: ${response.status}`;
         try {
            const errorData = await response.json();
            errorMessage = errorData.error || errorData.detail || errorMessage; 
         } catch (e) {
            // Ignore if response is not JSON
         }
         throw new Error(errorMessage);
      }

      const data = await response.json();
      console.log("[Frontend] Received response data:", data);

      if (data.reply) {
          displayMessage("Asistente", data.reply);
      }

      // Handle @Browse targets if present
      if (containsBrowseTag && data.browse_targets && data.browse_targets.length > 0) {
          console.log("[Frontend] Received browse targets from backend:", data.browse_targets);
          // La nueva función `handleBrowseExpansion` ahora recibirá la lista de URIs
          await handleBrowseExpansion(data.browse_targets);
      } else if (containsBrowseTag) {
          console.log("[Frontend] @Browse tag detected but no relevant targets were found by the backend.");
          showBanner("No se encontraron entidades relevantes para expandir con @Browse.", 'info');
      }

      if (data.error) {
         showBanner(`Error from backend: ${data.error}`, 'info');
      }

    } catch (error) { 
      // Don't show error message if request was aborted by user
      if (error.name === 'AbortError') {
        console.log("[Frontend] Request was aborted");
        return; // Exit early, cancelCurrentRequest already handled the UI
      }
      
      console.error("Error asking assistant:", error);
      hideTypingIndicator(); // Hide typing indicator on error
      showBanner(`Error communicating with assistant: ${error.message}`, 'info');
    } finally {
        // Clean up abort controller
        currentAbortController = null;
        
        if (sendButton) sendButton.disabled = false;
        const assistantInput = document.getElementById("assistant-input");
        if (assistantInput) assistantInput.focus(); 
    }
  }

  /**
   * Maneja la expansión del grafo para los objetivos de @Browse.
   * Recolecta datos de expansión para cada URI y los combina antes de actualizar el grafo.
   * @param {Array<Object>} targets - Un array de objetos, ej: [{uri: '...', score: 0.9}]
   */
  async function handleBrowseExpansion(targets) {
    console.log(`[handleBrowseExpansion] Expanding ${targets.length} target(s).`);

    try {
      const finalNodes = new Map();
      const finalEdges = [];
      const finalLiterals = {};

      for (const target of targets) {
        let expansionData = await fetchSubclasses(target.uri);
        if (!expansionData || !expansionData.nodes || expansionData.nodes.length <= 1) {
          expansionData = await fetchClassInstances(target.uri);
        }

        if (expansionData) {
          expansionData.nodes?.forEach(node => finalNodes.set(node.id, node));
          finalEdges.push(...(expansionData.edges || []));
          Object.assign(finalLiterals, expansionData.literals || {});
        }
      }

      const combinedData = {
        nodes: Array.from(finalNodes.values()),
        edges: Array.from(new Map(finalEdges.map(e => [`${e.source}-${e.target}-${e.link}`, e])).values()),
        literals: finalLiterals
      };

      const targetLabels = targets.map(t => getLabelFromUri(t.uri));
      const successMessage = `Mostrando expansión para: ${targetLabels.join(', ')}`;
      
      // Llamar a la función núcleo para actualizar el grafo
      expandAndReplaceGraph(combinedData, successMessage);

    } catch (error) {
      console.error("[handleBrowseExpansion] Error processing @Browse expansion:", error);
      showBanner("Ocurrió un error al expandir el grafo con @Browse.", 'error');
    }
  }

  /**
   * Handles expansion of multiple main nodes from backend expansion data
   * This function works with pre-expanded data from the backend that includes multiple entities above threshold
   * @param {Object} expansionData - Complete expansion data from backend including nodes and edges
   * @param {boolean} shouldReplaceGraph - Whether to replace the entire graph or add to existing
   * @returns {boolean} - True if expansion was successful
   */
  async function handleMultipleNodeExpansion(expansionData, shouldReplaceGraph) {
    console.log(`[handleMultipleNodeExpansion] Processing expansion with ${expansionData.nodes.length} nodes and ${expansionData.edges.length} edges`);
    
    // Log detailed information about main nodes being expanded
    const mainNodes = expansionData.nodes.filter(n => n.main_query_node);
    if (mainNodes.length > 0) {
      console.log(`[handleMultipleNodeExpansion] Expanding ${mainNodes.length} main nodes:`);
      mainNodes.forEach(node => {
        const label = node.label || getLabelFromUri(node.id);
        const confidence = node.confidence_score ? ` (confianza: ${(node.confidence_score * 100).toFixed(1)}%)` : '';
        console.log(`  - ${label}${confidence}`);
      });
    }
    
    try {
      const wasPausedBeforeExpansion = isPaused;
      
      // Guardar el estado actual ANTES de cualquier modificación
      saveCurrentState();
      
      if (shouldReplaceGraph) {
        // Replace entire graph with expansion data
        console.log("[handleMultipleNodeExpansion] Replacing entire graph with expansion data");
        
        // Set new nodes and links
        globalNodes = [...expansionData.nodes];
        globalLinks = [...expansionData.edges];
        nodes = globalNodes;
        links = globalLinks;
        
        // Clear and rebuild expandable classes
        expandableClassIds.clear();
        nodes.forEach(node => {
          if (node.id) {
            expandableClassIds.add(node.id);
          }
        });
        
        // Set literals if provided
        if (expansionData.literals) {
          literals = { ...expansionData.literals };
        } else {
          literals = {};
        }
        
      } else {
        // Add to existing graph
        console.log("[handleMultipleNodeExpansion] Adding expansion data to existing graph");
        
        const existingNodeIds = new Set(globalNodes.map(n => n.id));
        const newNodes = expansionData.nodes.filter(n => 
          n && n.id && !existingNodeIds.has(n.id)
        );
        
        // Add new nodes
        if (newNodes.length > 0) {
          newNodes.forEach(node => {
            if (!node.label) {
              node.label = getLabelFromUri(node.id);
            }
            expandableClassIds.add(node.id);
          });
          
          globalNodes = [...globalNodes, ...newNodes];
          nodes = globalNodes;
        }
        
        // Add new edges
        if (expansionData.edges.length > 0) {
          globalLinks = [...globalLinks, ...expansionData.edges];
          links = globalLinks;
        }
        
        // Merge literals
        if (expansionData.literals) {
          for (const sourceId in expansionData.literals) {
            if (!literals[sourceId]) literals[sourceId] = {};
            for (const linkId in expansionData.literals[sourceId]) {
              if (!literals[sourceId][linkId]) literals[sourceId][linkId] = [];
              literals[sourceId][linkId] = [...new Set([...literals[sourceId][linkId], ...expansionData.literals[sourceId][linkId]])];
            }
          }
        }
      }
      
      // Ensure all nodes have labels
      globalNodes.forEach(node => {
        if (!node.label) {
          node.label = getLabelFromUri(node.id);
        }
      });
      
      // Update graph properties and render
      updateGraphProperties();
      graph.setData(globalNodes, globalLinks);
      
      const nodeIds = globalNodes.map(n => n.id);
      graph.trackNodePositionsByIds(nodeIds);
      
      if (labels) labels.updateNodeMap(globalNodes);
      
      // Select all main query nodes to highlight them
      const mainNodeIds = expansionData.nodes
        .filter(n => n.main_query_node)
        .map(n => n.id);
      
      if (mainNodeIds.length > 0) {
        graph.selectNodesByIds(mainNodeIds, false);
        console.log(`[handleMultipleNodeExpansion] Selected ${mainNodeIds.length} main query nodes`);
      } else {
        // Fallback: select all nodes if no main nodes are marked
        graph.selectNodesByIds(nodeIds, false);
      }
      
      // Maintain pause state
      if (wasPausedBeforeExpansion) {
        graph.pause();
        if (pauseButton) pauseButton.textContent = "Resume";
        console.log("[handleMultipleNodeExpansion] Simulation kept paused after expansion");
      } else {
        graph.start();
        if (pauseButton) pauseButton.textContent = "Pause";
      }
      
      // Adjust view
      setTimeout(() => {
        fitView(500, 0.2);
      }, 700);
      
      console.log(`[handleMultipleNodeExpansion] Successfully processed expansion: ${nodes.length} total nodes, ${links.length} total edges`);
      return true;
      
    } catch (error) {
      console.error("[handleMultipleNodeExpansion] Error processing multiple node expansion:", error);
      return false;
    }
  }

  // --- 6. Configuración de Eventos del Asistente ---
  const assistantSendButton = document.getElementById("assistant-send");
  const assistantInput = document.getElementById("assistant-input");

  if (assistantSendButton && assistantInput) {
    assistantSendButton.addEventListener("click", () => {
      const question = assistantInput.value.trim();
      
      // Verificar si hay etiquetas no procesadas en el texto
      const tagRegex = /@[a-zA-Z][a-zA-Z0-9\-_]*/g;
      const unprocessedTags = question.match(tagRegex);
      
      if (unprocessedTags && unprocessedTags.length > 0) {
        // Verificar si alguna de las etiquetas encontradas es válida
        const validTagsFound = unprocessedTags.filter(tag => 
          availableTags.some(availableTag => availableTag.text.toLowerCase() === tag.toLowerCase())
        );
        
        if (validTagsFound.length > 0) {
          showBanner(`Debes presionar espacio después de escribir la etiqueta: ${validTagsFound.join(', ')}`, 'error');
          return;
        } else {
          // Si hay etiquetas inválidas, mostrar error por etiqueta no válida
          showBanner(`${unprocessedTags[0]} no es una etiqueta válida`, 'error');
          return;
        }
      }
      
      if (question || activeTags.size > 0) {
        if (!assistantSendButton.disabled) { 
          // Add send animation
          assistantSendButton.classList.add('send-animation');
          setTimeout(() => {
            assistantSendButton.classList.remove('send-animation');
          }, 200);
          
          const currentTags = getActiveTagsArray();
          // Display the message with active tags
          displayMessage("Usuario", question, currentTags);
          
          // Send to assistant with tags included
          const fullMessage = currentTags.length > 0 ? 
            `${currentTags.join(' ')} ${question}`.trim() : question;
          
          if (fullMessage.trim()) {
            askAssistant(fullMessage); 
          }
          
          // Clear input and tags
          assistantInput.value = ""; 
          clearActiveTags();
        }
      }
    });

    assistantInput.addEventListener("keypress", (event) => {
      if (event.key === "Enter" && !event.shiftKey) { 
        event.preventDefault(); 
        const question = assistantInput.value.trim();
        
        // Verificar si hay etiquetas no procesadas en el texto
        const tagRegex = /@[a-zA-Z][a-zA-Z0-9\-_]*/g;
        const unprocessedTags = question.match(tagRegex);
        
        if (unprocessedTags && unprocessedTags.length > 0) {
          // Verificar si alguna de las etiquetas encontradas es válida
          const validTagsFound = unprocessedTags.filter(tag => 
            availableTags.some(availableTag => availableTag.text.toLowerCase() === tag.toLowerCase())
          );
          
          if (validTagsFound.length > 0) {
            showBanner(`Debes presionar espacio después de escribir la etiqueta: ${validTagsFound.join(', ')}`, 'error');
            return;
          } else {
            // Si hay etiquetas inválidas, mostrar error por etiqueta no válida
            showBanner(`${unprocessedTags[0]} no es una etiqueta válida`, 'error');
            return;
          }
        }
        
        if (question || activeTags.size > 0) {
          if (!assistantSendButton.disabled) { 
            const currentTags = getActiveTagsArray();
            // Display the message with active tags
            displayMessage("Usuario", question, currentTags);
            
            // Send to assistant with tags included
            const fullMessage = currentTags.length > 0 ? 
              `${currentTags.join(' ')} ${question}`.trim() : question;
            
            if (fullMessage.trim()) {
              askAssistant(fullMessage);
            }
            
            // Clear input and tags
            assistantInput.value = ""; 
            clearActiveTags();
          }
        }
      }
    });
  } else {
       console.error("Assistant input elements not found!");
  }

  // --- 7. Controles de UI Adicionales (Pausa, Zoom, Selección) ---
  let isPaused = globalIsPaused;
  const pauseButton = globalPauseButton || document.getElementById("pause");
  
  // Update global reference
  globalPauseButton = pauseButton;

  function pauseSimulation() {
    if (graph && !isPaused) {
      isPaused = true;
      globalIsPaused = true;
      if (pauseButton) pauseButton.textContent = "Resume"; 
      graph.pause();
      console.log("Simulation paused.");
    }
  }

  function startSimulation() {
    if (graph && isPaused) {
      isPaused = false;
      globalIsPaused = false;
      if (pauseButton) pauseButton.textContent = "Pause"; 
      graph.start();
      console.log("Simulation resumed.");
    }
  }

  function togglePause() {
    if (isPaused) startSimulation();
    else pauseSimulation();
  }

  if (pauseButton) {
    pauseButton.addEventListener("click", togglePause);
    pauseButton.textContent = isPaused ? "Resume" : "Pause";
  }

  /**
   * Obtiene el ID de un nodo aleatorio del conjunto actual.
   * @returns {string|null} - ID del nodo o null si no hay nodos.
   */
  function getRandomNodeId() {
    return nodes.length > 0 ? nodes[Math.floor(Math.random() * nodes.length)].id : null;
  }

  /**
   * Genera un número aleatorio en un rango.
   * @param {number} min - Mínimo.
   * @param {number} max - Máximo.
   * @returns {number} - Número aleatorio.
   */
  function getRandomInRange(min, max) {
    return Math.random() * (max - min) + min;
  }

  // Acciones mejoradas para botones
  function zoomInOnRandomNode() {
    const nodeId = getRandomNodeId();
    if (nodeId && graph) {
        // Pause la simulación para mejor visualización
        if (!isPaused) {
          pauseSimulation();
        }
        
        // Primero deseleccionar todos los nodos
        graph.unselectNodes();
        
        // Seleccionar solo el nodo objetivo
        graph.selectNodeById(nodeId, true);
        
        // Centrar y hacer zoom al nodo seleccionado
        setTimeout(() => {
          if (graph) {
            try {
              // Usar fitView con padding muy pequeño para centrar en el nodo seleccionado
              // Como solo hay un nodo seleccionado, fitView centrará la vista en él
              graph.fitView(600, 0.01); // padding muy pequeño para acercamiento máximo
              
              // Aplicar zoom adicional después del centrado
              setTimeout(() => {
                // Obtener el nivel de zoom actual y aumentarlo
                const currentZoom = graph.getZoomLevel();
                if (currentZoom < 5) { // Evitar zoom excesivo
                  graph.zoom(Math.min(currentZoom * 2, 5), 400); // Duplicar zoom con límite
                }
              }, 200); // Esperar a que complete el fitView
              
              const nodeLabel = nodes.find(n => n.id === nodeId)?.label || getLabelFromUri(nodeId);
              showBanner(`Nodo seleccionado: "${nodeLabel}"`, 'info');
              
            } catch (error) {
              console.warn("Error applying zoom to node:", error);
              // Fallback más simple
              graph.fitView(800, 0.1);
              const nodeLabel = nodes.find(n => n.id === nodeId)?.label || getLabelFromUri(nodeId);
              showBanner(`Vista centrada en nodo: "${nodeLabel}"`, 'info');
            }
          }
        }, 100); // Tiempo reducido para mejor respuesta
    } else {
        console.warn("No nodes available or graph not ready for zoom.");
        showBanner("No hay nodos disponibles para hacer zoom", 'warning');
    }
  }

  function selectRandomNode() {
    const nodeId = getRandomNodeId();
     if (nodeId && graph) {
        // Deseleccionar nodos anteriores primero
        graph.unselectNodes();
        
        // Seleccionar el nuevo nodo
        graph.selectNodeById(nodeId, true);
        
        // Ajustar vista para mostrar el nodo seleccionado
        setTimeout(() => {
          fitView(500, 0.3);
          const nodeLabel = nodes.find(n => n.id === nodeId)?.label || getLabelFromUri(nodeId);
          showBanner(`Nodo seleccionado: "${nodeLabel}"`, 'success');
        }, 200);
     } else {
        console.warn("No nodes available or graph not ready for selection.");
        showBanner("No hay nodos disponibles para seleccionar", 'warning');
     }
  }

  function selectNodesInRandomArea() {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (w > 0 && h > 0 && graph) {
        // Deseleccionar nodos anteriores
        graph.unselectNodes();
        
        // Crear un área más centrada y de tamaño variable
        const areaWidth = getRandomInRange(w * 0.15, w * 0.35); // Entre 15% y 35% del ancho
        const areaHeight = getRandomInRange(h * 0.15, h * 0.35); // Entre 15% y 35% del alto
        
        // Centrar el área aleatoriamente pero asegurar que esté completamente dentro del canvas
        const centerX = getRandomInRange(areaWidth / 2, w - areaWidth / 2);
        const centerY = getRandomInRange(areaHeight / 2, h - areaHeight / 2);
        
        const left = centerX - areaWidth / 2;
               const right = centerX + areaWidth / 2;
        const top = centerY - areaHeight / 2;
        const bottom = centerY + areaHeight / 2;
        
        // Seleccionar nodos en el área
        const selectedNodes = graph.selectNodesInRange([
          [left, top], 
          [right, bottom] 
        ]);
        
        // Feedback visual mejorado
        if (selectedNodes && selectedNodes.length > 0) {
          showBanner(`${selectedNodes.length} nodos seleccionados en área`, 'success');
          
          // Ajustar vista para mostrar el área seleccionada
          setTimeout(() => {
            fitView(500, 0.2);
          }, 300);
        } else {
          showBanner("No se encontraron nodos en el área seleccionada", 'info');
        }
        
        console.log(`Area seleccionada: [${Math.round(left)}, ${Math.round(top)}] a [${Math.round(right)}, ${Math.round(bottom)}] - ${selectedNodes?.length || 0} nodos`);
    } else {
        console.warn("Invalid canvas dimensions or graph not ready for area selection.");
        showBanner("No se puede seleccionar área: canvas no disponible", 'error');
    }
  }

  // Variables para modos interactivos (simplified)
  let interactiveMode = null; // Remove area selection functionality
  let areaSelectionStart = null;

  // Función para activar modo de selección de área interactiva (development mode)
  function enableInteractiveAreaSelection() {
    showBanner("Esta funcionalidad está en desarrollo", 'dev');
  }

  // Función mejorada para Fit View
  function enhancedFitView() {
    if (!graph) {
      showBanner("Grafo no disponible", 'error');
      return;
    }

    // Verificar si hay nodos seleccionados
    const selectedNodes = graph.getSelectedNodeIds?.() || [];
    
    if (selectedNodes.length > 0) {
      // Si hay nodos seleccionados, ajustar vista para mostrarlos mejor
      fitView(600, 0.15);
      showBanner(`Vista ajustada para ${selectedNodes.length} nodo(s) seleccionado(s)`, 'info');
    } else {
      // Vista general de todos los nodos
      fitView(500, 0.2);
      showBanner("Vista ajustada para mostrar todos los nodos", 'success');
    }
  }

  document.getElementById("fit-view")?.addEventListener("click", enhancedFitView);
  document.getElementById("zoom")?.addEventListener("click", zoomInOnRandomNode);
  document.getElementById("select-points-in-area")?.addEventListener("click", enableInteractiveAreaSelection);
  
  // Back to main menu button
  document.getElementById("back-to-menu")?.addEventListener("click", () => {
    if (window.showBackToMenuConfirmation) {
      window.showBackToMenuConfirmation();
    }
  });

  // Confirmation banner event listeners
  document.getElementById("confirm-back-to-menu")?.addEventListener("click", () => {
    if (window.hideBackToMenuConfirmation) {
      window.hideBackToMenuConfirmation();
    }
    // Go back to main menu
    setTimeout(() => {
      if (window.showMainMenu) {
        window.showMainMenu();
      }
    }, 300);
  });

  document.getElementById("cancel-back-to-menu")?.addEventListener("click", () => {
    if (window.hideBackToMenuConfirmation) {
      window.hideBackToMenuConfirmation();
    }
  });

  // --- 8. Funcionalidad del Botón de Retorno ---

  // --- 8. Configuración del Botón de Retorno (usando funciones globales) ---
  
  if (returnButton) {
    // Remove any existing event listener to prevent duplicates
    returnButton.removeEventListener("click", returnToPreviousState);
    returnButton.addEventListener("click", returnToPreviousState);
  } else {
      console.error("Return button not found!");
  }

  // --- 9. Efectos Visuales Adicionales (Botones) ---
  const actionButtons = document.querySelectorAll('.action-container button');
  actionButtons.forEach(button => {
    button.addEventListener('click', function() {
      this.classList.add('pulse-effect');
      setTimeout(() => {
        this.classList.remove('pulse-effect');
      }, 600); 
    });
  });

  console.log("Initialization complete. Application ready.");

  // Save the initial state as the first state in history for the new ontology
  console.log('Saving initial state for new ontology session');
  saveCurrentState();

  /**
   * Función núcleo para actualizar el grafo con nuevos datos de expansión.
   * Reemplaza el estado actual del grafo y actualiza la vista.
   * @param {Object} expansionData - Objeto con { nodes, edges, literals }
   * @param {string} successMessage - Mensaje a mostrar si la expansión tiene éxito.
   */
  function expandAndReplaceGraph(expansionData, successMessage) {
    if (!expansionData || !expansionData.nodes || expansionData.nodes.length === 0) {
      showBanner("La expansión no produjo nuevos nodos para mostrar.", 'info');
      return;
    }
    
    // Guardar el estado actual ANTES de reemplazarlo.
    saveCurrentState();
    
    const wasPausedBeforeExpansion = isPaused;
    
    // Reemplazar el grafo actual con los nuevos datos
    globalNodes = expansionData.nodes;
    globalLinks = expansionData.edges || [];
    nodes = globalNodes;
    links = globalLinks;
    literals = expansionData.literals || {};
    globalCurrentExpandedNode = null; // Reiniciar el nodo expandido

    updateGraphProperties();
    graph.setData(nodes, links);
    
    // Lógica de UI: seleccionar nodos, ajustar vista, etc.
    const nodeIds = nodes.map(n => n.id);
    graph.trackNodePositionsByIds(nodeIds);
    if (labels) labels.updateNodeMap(nodes);
    graph.selectNodesByIds(nodeIds, false);
    
    // Mantener el estado de pausa
    if (wasPausedBeforeExpansion) {
        graph.pause();
        if (pauseButton) pauseButton.textContent = "Resume";
    } else {
        graph.start();
        if (pauseButton) pauseButton.textContent = "Pause";
    }
    
    setTimeout(() => fitView(500, 0.2), 700);

    showBanner(successMessage, 'success');
  }
  
  /**
   * Shows annotations (comments, labels, descriptions) for a node
   */
  async function showNodeAnnotations(node) {
    try {
      console.log(`[showNodeAnnotations] Fetching annotations for node: ${node.id}`);
      
      const nodeLabel = node.label || getLabelFromUri(node.id);
      
      // Show loading in popup
      showAnnotationsPopup();
      const contentDiv = document.getElementById('annotations-content');
      const titleSpan = document.querySelector('.annotations-popup-title');
      
      titleSpan.textContent = `Annotations - ${nodeLabel}`;
      contentDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #888;">Loading annotations...</div>';
      
      // Fetch annotations for the node
      const annotationData = await fetchNodeAnnotations(node.id);
      
      if (annotationData.annotations && annotationData.annotations.length > 0) {
        // Create annotations HTML
        let annotationsHTML = `<div class="node-uri">${node.id}</div>`;
        
        // Group annotations by property type
        const groupedAnnotations = {};
        annotationData.annotations.forEach(annotation => {
          if (!groupedAnnotations[annotation.propertyName]) {
            groupedAnnotations[annotation.propertyName] = [];
          }
          groupedAnnotations[annotation.propertyName].push(annotation);
        });
        
        // Display annotations in HTML format
        Object.keys(groupedAnnotations).forEach(propertyName => {
          annotationsHTML += `<div class="annotation-item">`;
          annotationsHTML += `<div class="annotation-property">${propertyName}</div>`;
          
          groupedAnnotations[propertyName].forEach(annotation => {
            if (annotation.valueType === 'uri') {
              // For URI values, show a shortened version with full URI
              const shortUri = annotation.value.split(/[#/]/).pop() || annotation.value;
              annotationsHTML += `<div class="annotation-value uri">
                <strong>${shortUri}</strong><br>
                <small>${annotation.value}</small>
              </div>`;
            } else {
              // For literal values, show the text
              let value = annotation.value;
              annotationsHTML += `<div class="annotation-value literal">${value}`;
              if (annotation.language) {
                annotationsHTML += ` <span class="annotation-language">(${annotation.language})</span>`;
              }
              annotationsHTML += `</div>`;
            }
          });
          
          annotationsHTML += `</div>`;
        });
        
        contentDiv.innerHTML = annotationsHTML;
        console.log(`[showNodeAnnotations] Displayed ${annotationData.annotations.length} annotations for node: ${node.id}`);
        
      } else {
        // No annotations found
        contentDiv.innerHTML = `
          <div class="node-uri">${node.id}</div>
          <div class="no-annotations">
            No annotations found for this node.<br>
            This node doesn't have any standard annotation properties like comments, labels, or descriptions.
          </div>
        `;
        console.log(`[showNodeAnnotations] No annotations found for node: ${node.id}`);
      }
      
    } catch (error) {
      console.error(`[showNodeAnnotations] Error fetching annotations for ${node.id}:`, error);
      
      // Show error in popup
      const contentDiv = document.getElementById('annotations-content');
      if (contentDiv) {
        contentDiv.innerHTML = `
          <div class="no-annotations">
            Error loading annotations for this node.<br>
            Please try again later.
          </div>
        `;
      }
    }
  }

  /**
   * Shows the annotations popup window
   */
  function showAnnotationsPopup() {
    const popup = document.getElementById('annotations-popup');
    if (popup) {
      popup.style.display = 'block';
      // Force reflow
      popup.offsetHeight;
      // Add show class for animation
      setTimeout(() => {
        popup.classList.add('show');
      }, 10);
    }
  }

  /**
   * Hides the annotations popup window
   */
  function hideAnnotationsPopup() {
    const popup = document.getElementById('annotations-popup');
    if (popup) {
      popup.classList.remove('show');
      setTimeout(() => {
        popup.style.display = 'none';
      }, 300);
    }
  }
  } catch (error) {
    console.error('Error initializing main application:', error);
    console.error('Error stack trace:', error.stack);
    console.error('Error occurred in initializeMainApplication with source:', source);
    // Show error to user
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      background: linear-gradient(135deg, #ff6b6b, #ff5252);
      color: white;
      padding: 30px;
      border-radius: 15px;
      box-shadow: 0 20px 40px rgba(255, 107, 107, 0.3);
      z-index: 12000;
      font-family: 'Space Grotesk', sans-serif;
      text-align: center;
      max-width: 400px;
    `;
    errorDiv.innerHTML = `
      <h3 style="margin-top: 0;">Error al cargar la aplicación</h3>
      <p>${error.message}</p>
      <button onclick="location.reload()" style="
        background: rgba(255, 255, 255, 0.2);
        border: none;
        color: white;
        padding: 10px 20px;
        border-radius: 25px;
        cursor: pointer;
        margin-top: 15px;
        font-family: inherit;
      ">Reintentar</button>
    `;
    document.body.appendChild(errorDiv);
  } finally {
    // Show all main app elements when initializing (success or error)
    showMainAppElements();
    
    // Release the initialization lock
    isInitializing = false;
    console.log('Application initialization completed');
  }
  
  // Create a global cleanup function for future use
  window.cleanupGraphApplication = function() {
    console.log('Global cleanup function called');
    
    if (window.globalGraph) {
      window.globalGraph = null;
    }
    
    if (window.globalLabels) {
      try {
        window.globalLabels.cleanup();
      } catch (error) {
        console.warn('Error during labels cleanup:', error);
      }
      window.globalLabels = null;
    }
    
    const labelsDiv = document.getElementById("labels");
    if (labelsDiv) {
      labelsDiv.innerHTML = '';
    }
    
    const canvas = document.querySelector("canvas");
    if (canvas) {
      const context = canvas.getContext('2d');
      if (context) {
        context.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
    
    console.log('Global cleanup completed');
  };
}

/**
 * Show all main application elements
 */
function showMainAppElements() {
  // Remove the loading class from body to show main app elements
  document.body.classList.remove('loading');
}

// Listen for the app ready event from startup manager
window.addEventListener('appReady', (event) => {
  const { source } = event.detail;
  console.log('App ready event received with source:', source);
  console.log('About to call initializeMainApplication');
  initializeMainApplication(source);
});

// Check if we should initialize directly (only if no startup manager and no saved state)
// Use a longer timeout and check multiple times
let checkCount = 0;
const maxChecks = 10;

const checkForStartupManager = () => {
  checkCount++;
  
  if (window.interactiveStartupManager) {
    console.log('Startup manager found, letting it handle initialization');
    return;
  }
  
  if (checkCount >= maxChecks) {
    // Check if there's a saved state that would indicate the startup manager should handle this
    const savedState = localStorage.getItem('atmentis_app_state');
    if (!savedState) {
      console.warn('Startup manager not found after multiple checks and no saved state, initializing with default data');
      initializeMainApplication('default');
    } else {
      console.warn('Startup manager not found but saved state exists - this might indicate a loading issue');
      // Clear potentially corrupted state and reload
      localStorage.removeItem('atmentis_app_state');
      location.reload();
    }
  } else {
    // Check again after a short delay
    setTimeout(checkForStartupManager, 200);
  }
};

// Start checking after the DOM is fully loaded
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    setTimeout(checkForStartupManager, 500);
  });
} else {
  setTimeout(checkForStartupManager, 500);
}