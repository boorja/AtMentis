/**
 * Interactive Graph Startup Manager
 * Handles the interactive node-based startup screen with draggable elements and SVG connections
 */

// Configuration constants
const CONFIG = {
  BACKEND_URL: 'http://192.168.216.102:5000',
  VIRTUOSO_URL: 'http://192.168.216.102:32323',
  STATE_KEY: 'atmentis_app_state',
  STATE_MAX_AGE: 7 * 24 * 60 * 60 * 1000, // 7 days
  EMERGENCY_TIMEOUT: 15000, // 15 seconds
  POLL_INTERVAL: 1000, // 1 second
  MAX_POLLS: 300, // 5 minutes
  VALID_EXTENSIONS: ['.owl', '.ttl', '.rdf', '.n3']
};

class InteractiveStartupManager {
  constructor() {
    this.loadingScreen = document.getElementById('loading-screen');
    this.startScreen = document.getElementById('start-screen');
    this.graphContainer = document.getElementById('graph-container');
    
    this.nodes = {};
    this.clickHandlersInitialized = false;
    
    this.init();
  }

  // ===========================================
  // STATE MANAGEMENT METHODS
  // ===========================================

  /**
   * Save the application state to localStorage
   */
  saveApplicationState(graphUri, isTemporary = false) {
    const state = {
      graphUri: graphUri,
      timestamp: Date.now(),
      version: '1.0',
      isTemporary: isTemporary // Marca si es una ontología temporal (subida por usuario)
    };
    
    try {
      localStorage.setItem(CONFIG.STATE_KEY, JSON.stringify(state));
      console.log('Application state saved:', state);
    } catch (error) {
      console.error('Error saving application state:', error);
    }
  }

  /**
   * Load the application state from localStorage
   */
  loadApplicationState() {
    try {
      const stateStr = localStorage.getItem(CONFIG.STATE_KEY);
      if (!stateStr) return null;
      
      const state = JSON.parse(stateStr);
      
      // Check if state is recent
      if (Date.now() - state.timestamp > CONFIG.STATE_MAX_AGE) {
        console.log('Application state expired, clearing...');
        this.clearApplicationState();
        return null;
      }
      
      console.log('Application state loaded:', state);
      return state;
    } catch (error) {
      console.error('Error loading application state:', error);
      this.clearApplicationState();
      return null;
    }
  }

  /**
   * Clear the application state from localStorage
   */
  clearApplicationState() {
    try {
      localStorage.removeItem(CONFIG.STATE_KEY);
      console.log('Application state cleared');
    } catch (error) {
      console.error('Error clearing application state:', error);
    }
  }

  // ===========================================
  // BACKEND COMMUNICATION METHODS
  // ===========================================

  /**
   * Check if the backend cache is available for the saved graph
   */
  async checkBackendCacheStatus(graphUri) {
    console.log('Checking backend cache status for graph:', graphUri);
    try {
      const response = await fetch(`${CONFIG.BACKEND_URL}/check-cache-status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph_uri: graphUri })
      });
      
      console.log('Cache check response status:', response.status);
      
      if (!response.ok) {
        console.log('Backend cache check failed:', response.statusText);
        return false;
      }
      
      const result = await response.json();
      console.log('Cache check result:', result);
      return result.cache_available === true;
    } catch (error) {
      console.error('Error checking backend cache status:', error);
      return false;
    }
  }

  /**
   * Verify if a graph is ready and available
   */
  async verifyGraphState(graphUri) {
    console.log('Starting graph state verification for:', graphUri);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      
      console.log('Checking backend cache status...');
      const cacheAvailable = await this.checkBackendCacheStatus(graphUri);
      console.log('Backend cache available:', cacheAvailable);
      
      if (!cacheAvailable) {
        console.log('Backend cache not available for graph:', graphUri);
        clearTimeout(timeoutId);
        return false;
      }
      
      console.log('Checking if backend is generally available...');
      const backendResponse = await fetch(`${CONFIG.BACKEND_URL}/initialize-progress`, {
        method: 'GET',
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      console.log('Backend response status:', backendResponse.status);
      
      if (!backendResponse.ok) {
        console.log('Backend response not ok:', backendResponse.statusText);
        throw new Error('Backend not available');
      }
      
      const progress = await backendResponse.json();
      console.log('Backend progress response:', progress);
      
      const isCompleted = progress.status === 'completed';
      console.log('Graph completion status:', isCompleted);
      return isCompleted;
      
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Backend verification timed out');
      } else {
        console.log('Backend verification failed:', error.message);
      }
      return false;
    }
  }

  /**
   * Fetch available graphs from Virtuoso
   */
  async fetchAvailableGraphs() {
    try {
      const response = await fetch(`${CONFIG.VIRTUOSO_URL}/available-graphs`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      if (data.status === 'error') {
        throw new Error(data.error);
      }
      
      return data.graphs || [];
    } catch (error) {
      console.error('Error fetching graphs from server:', error);
      console.warn('Using fallback mock data');
      return this.getMockGraphData();
    }
  }

  /**
   * Clear backend cache
   */
  async clearBackendCache(graphUri) {
    try {
      await fetch(`${CONFIG.BACKEND_URL}/clear_cache`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ graph_uri: graphUri })
      });
      console.log('Cleared backend cache for graph:', graphUri);
    } catch (error) {
      console.warn('Failed to clear backend cache:', error);
    }
  }

  // ===========================================
  // UTILITY METHODS
  // ===========================================

  /**
   * Check if the given graph URI is the same as the currently saved one
   */
  isSameGraph(graphUri) {
    const savedState = this.loadApplicationState();
    return savedState && savedState.graphUri === graphUri;
  }

  /**
   * Utility function to create delays
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Type text with animation
   */
  async typeText(element, text) {
    if (!element || !text) return;
    
    for (let i = 0; i < text.length; i++) {
      element.textContent = text.substring(0, i + 1);
      await this.delay(50);
    }
  }

  /**
   * Add visual click effect to node
   */
  addNodeClickEffect(nodeId) {
    const node = document.getElementById(nodeId);
    if (node) {
      const originalTransform = node.style.transform;
      node.style.transform = 'scale(0.95)';
      setTimeout(() => {
        node.style.transform = originalTransform || 'scale(1)';
      }, 150);
    }
  }

  /**
   * Create loading message element
   */
  createLoadingMessage(type, primaryText, secondaryText = '', tertiaryText = '') {
    const colors = {
      same: '#22c55e',
      different: '#3b82f6', 
      expired: '#f59e0b',
      error: '#ef4444'
    };

    const icons = {
      same: '✅',
      different: '🔄',
      expired: '⚠️',
      error: '❌'
    };

    return `
      <div style="text-align: center; padding: 20px; color: white;">
        <div class="spinner"></div>
        <p style="color: ${colors[type]}; font-weight: 500; margin-top: 10px;">
          ${icons[type]} ${primaryText}
        </p>
        ${secondaryText ? `<p style="color: #94a3b8; font-size: 0.9em;">${secondaryText}</p>` : ''}
        ${tertiaryText ? `<p style="color: #64748b; font-size: 0.8em; margin-top: 8px;">${tertiaryText}</p>` : ''}
      </div>
    `;
  }

  /**
   * Show temporary loading message
   */
  showTemporaryMessage(content, duration = 2000) {
    const loadingDiv = this.startScreen.querySelector('.loading-graphs') || 
                       document.createElement('div');
    loadingDiv.className = 'loading-graphs';
    loadingDiv.innerHTML = content;
    
    if (!this.startScreen.querySelector('.loading-graphs')) {
      this.startScreen.appendChild(loadingDiv);
    }
    
    setTimeout(() => {
      if (loadingDiv.parentNode) {
        loadingDiv.remove();
      }
    }, duration);
  }

  /**
   * Get mock graph data for fallback
   */
  getMockGraphData() {
    return [
      {
        name: 'Ontología de Pizza',
        description: 'Ejemplo clásico de ontología con pizzas y sus ingredientes',
        uri: 'http://www.co-ode.org/ontologies/pizza/pizza.owl',
        triples: 945
      },
      {
        name: 'Instancias Local', 
        description: 'Instancias definidas localmente en el sistema',
        uri: 'http://example.org/local/instances',
        triples: 156
      },
      {
        name: 'Ontología Final',
        description: 'Ontología principal del proyecto',
        uri: 'http://example.org/ontology/final',
        triples: 2341
      }
    ];
  }

  /**
   * Show the main menu (startup screen)
   */
  async showMainMenu() {
    console.log('InteractiveStartupManager.showMainMenu called');
    
    // Check if we need to cleanup a temporary ontology
    const savedState = this.loadApplicationState();
    if (savedState && savedState.isTemporary) {
      console.log('Cleaning up temporary ontology before returning to menu:', savedState.graphUri);
      await this.cleanupTemporaryOntology(savedState.graphUri);
      this.clearApplicationState();
    }
    
    // Hide main app
    const mainApp = document.getElementById('main-app');
    if (mainApp) {
      console.log('Hiding main app');
      mainApp.classList.add('hidden');
    } else {
      console.error('main-app element not found');
    }
    
    // Hide KG init screen if visible
    this.hideKGInitScreen();
    
    // Show startup screen
    console.log('Showing startup screen');
    this.startScreen.classList.remove('hidden');
    this.startScreen.classList.add('show');
    
    // Hide loading screen
    this.loadingScreen.classList.add('hidden');
    
    // Re-initialize click handlers to ensure event listeners work
    // Force reinitialize in case they were lost, with a small delay to ensure DOM is ready
    setTimeout(() => {
      this.clickHandlersInitialized = false;
      this.initializeClickHandlers();
    }, 100);
    
    console.log('Main menu shown successfully');
  }

  /**
   * Initialize the startup manager
   */
  async init() {
    try {
      // Check if there's a saved application state
      const savedState = this.loadApplicationState();
      
      if (savedState && savedState.graphUri) {
        console.log('Found saved application state, restoring graph:', savedState.graphUri);
        
        // Show loading screen briefly
        this.loadingScreen.classList.remove('hidden');
        
        // Add a message about restoring
        const progressText = this.loadingScreen.querySelector('.loading-progress-text');
        if (progressText) {
          progressText.textContent = 'Verificando estado del grafo...';
        }
        
        // Add an emergency timeout to prevent infinite loading
        const emergencyTimeout = setTimeout(() => {
          console.warn('Emergency timeout triggered - forcing normal startup');
          this.clearApplicationState();
          this.startNormalFlow();
        }, 15000); // 15 seconds emergency timeout
        
        // Wait a moment to show the restoration message
        await this.delay(800);
        
        // Verify if the graph is actually ready before restoring
        try {
          console.log('Verifying graph state for:', savedState.graphUri);
          const isReady = await this.checkBackendCacheStatus(savedState.graphUri);
          console.log('Graph verification result:', isReady);
          
          // Clear emergency timeout since we got a response
          clearTimeout(emergencyTimeout);
          
          if (isReady) {
            console.log('Graph is ready, starting restoration...');
            if (progressText) {
              progressText.textContent = 'Restaurando sesión anterior...';
            }
            await this.delay(500);
            console.log('Calling startMainApplication with:', savedState.graphUri);
            this.startMainApplication(savedState.graphUri);
          } else {
            console.log('Graph not ready, clearing saved state and showing startup');
            this.clearApplicationState();
            if (progressText) {
              progressText.textContent = 'Reiniciando aplicación...';
            }
            await this.delay(500);
            this.startNormalFlow();
          }
        } catch (error) {
          // Clear emergency timeout on error too
          clearTimeout(emergencyTimeout);
          console.error('Error verifying graph state:', error);
          console.log('Error verifying state, clearing and showing startup');
          this.clearApplicationState();
          if (progressText) {
            progressText.textContent = 'Reiniciando aplicación...';
          }
          await this.delay(500);
          this.startNormalFlow();
        }
        
        return;
      }
      
      // Normal startup flow if no saved state
      this.startNormalFlow();
      
    } catch (error) {
      console.error('Error during loading:', error);
      this.showError('Error al cargar la aplicación');
    }
  }

  /**
   * Start the normal startup flow
   */
  async startNormalFlow() {
    // Show loading screen
    this.loadingScreen.classList.remove('hidden');
    
    // Initialize modal handlers
    this.initializeModalHandlers();
    
    // Simulate loading time
    await this.simulateLoading();
    
    // Transition to interactive graph
    this.transitionToGraph();
    
    // Initialize interactive elements
    setTimeout(() => this.initializeGraph(), 1600);
  }

  // ===== INITIALIZATION METHODS =====

  /**
   * Initialize modal event handlers
   */
  initializeModalHandlers() {
    // Modal close handlers
    document.getElementById('modal-close')?.addEventListener('click', () => {
      this.hideUploadModal();
    });

    document.getElementById('graph-modal-close')?.addEventListener('click', () => {
      this.hideGraphListModal();
    });

    // Close modals when clicking outside
    document.getElementById('upload-modal')?.addEventListener('click', (e) => {
      if (e.target.id === 'upload-modal') {
        this.hideUploadModal();
      }
    });

    document.getElementById('graph-list-modal')?.addEventListener('click', (e) => {
      if (e.target.id === 'graph-list-modal') {
        this.hideGraphListModal();
      }
    });

    // Setup upload area
    this.setupUploadArea();
  }

  /**
   * Simulate loading process with realistic messages
   */
  async simulateLoading() {
    const loadingTasks = [
      { name: 'Iniciando aplicación...', duration: 800 },
      { name: 'Cargando componentes...', duration: 700 },
      { name: 'Preparando interfaz...', duration: 600 },
      { name: 'Configurando controles...', duration: 700 },
      { name: 'Iniciando sistema de grafos...', duration: 500 }
    ];

    const progressText = this.loadingScreen.querySelector('.loading-progress-text');

    for (const task of loadingTasks) {
      if (progressText) {
        // Clear text and wait a moment
        progressText.textContent = '';
        await this.delay(100);
        
        // Type the new text
        await this.typeText(progressText, task.name);
        
        // Wait for the task duration
        await this.delay(task.duration);
      }
    }
  }

  /**
   * Type text with animation - fixed version
   */
  async typeText(element, text) {
    if (!element || !text) return;
    
    for (let i = 0; i < text.length; i++) {
      element.textContent = text.substring(0, i + 1);
      await this.delay(50); // Consistent typing speed
    }
  }

  /**
   * Utility function to create delays
   */
  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Transition from loading screen to interactive graph
   */
  transitionToGraph() {
    this.startScreen.classList.remove('hidden');
    
    setTimeout(() => {
      this.startScreen.classList.add('show');
    }, 100);
    
    setTimeout(() => {
      this.loadingScreen.classList.add('hidden');
    }, 1500);
  }

  /**
   * Initialize interactive graph elements
   */
  /**
   * Initialize interactive graph elements
   */
  initializeGraph() {
    // Get all nodes
    const nodeElements = this.graphContainer.querySelectorAll('.graph-node');
    
    // Initialize node data
    nodeElements.forEach(nodeEl => {
      const nodeId = nodeEl.id;
      this.nodes[nodeId] = {
        element: nodeEl,              
        x: 0,
        y: 0,
        type: nodeEl.dataset.node
      };
    });

    this.initializeClickHandlers();
    this.initializeControls();
  }

  /**
   * Center entire graph
   */
  centerGraph() {
    this.resetNodePositions();
  }

  /**
   * Initialize click handlers for nodes
   */
  initializeClickHandlers() {
    // Avoid duplicate event listeners
    if (this.clickHandlersInitialized) {
      console.log('Click handlers already initialized, skipping...');
      return;
    }
    
    const loadGraphsNode = document.getElementById('load-graphs-node');
    const uploadNode = document.getElementById('upload-node');
    
    if (loadGraphsNode) {
      console.log('Adding click handler for load-graphs-node');
      loadGraphsNode.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Load graphs node clicked');
        this.handleLoadGraphsOption();
      });
    } else {
      console.error('load-graphs-node not found');
    }
    
    if (uploadNode) {
      console.log('Adding click handler for upload-node');
      uploadNode.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        console.log('Upload node clicked');
        this.handleUploadOption();
      });
    } else {
      console.error('upload-node not found');
    }
    
    this.clickHandlersInitialized = true;
    console.log('Click handlers initialized successfully');
  }

  /**
   * Initialize control buttons
   */
  initializeControls() {
    const resetButton = document.getElementById('reset-positions');
    const centerButton = document.getElementById('center-graph');
    
    if (resetButton) {
      resetButton.addEventListener('click', () => this.resetNodePositions());
    }
    
    if (centerButton) {
      centerButton.addEventListener('click', () => this.centerGraph());
    }
  }

  /**
   * Handle load graphs option
   */
  async handleLoadGraphsOption() {
    console.log('handleLoadGraphsOption called');
    this.addNodeClickEffect('load-graphs-node');
    await this.delay(200);
    console.log('About to show graph list modal');
    this.showGraphListModal();
  }

  /**
   * Handle upload option
   */
  async handleUploadOption() {
    console.log('handleUploadOption called');
    this.addNodeClickEffect('upload-node');
    await this.delay(200);
    console.log('About to show upload modal');
    this.showUploadModal();
  }

  // ===== MODAL MANAGEMENT METHODS =====

  /**
   * Show graph list modal
   */
  async showGraphListModal() {
    const modal = document.getElementById('graph-list-modal');
    const loadingDiv = modal.querySelector('.loading-graphs');
    const graphList = document.getElementById('graph-list');

    modal.classList.remove('hidden');
    setTimeout(() => modal.classList.add('show'), 10);

    // Show loading state
    loadingDiv.style.display = 'block';
    loadingDiv.innerHTML = `
      <div style="text-align: center; padding: 40px;">
        <div class="spinner"></div>
        <p style="color: #334155; font-weight: 500; margin-bottom: 10px;">Cargando grafos disponibles...</p>
        <p style="color: #64748b; font-size: 0.9rem;">Conectando con Virtuoso</p>
      </div>
    `;
    graphList.style.display = 'none';

    try {
      // Fetch graphs directly without using StartupManager
      const graphs = await this.fetchAvailableGraphs();
      
      // Hide loading and show graphs
      loadingDiv.style.display = 'none';
      this.populateGraphList(graphs);
      graphList.style.display = 'block';
    } catch (error) {
      console.error('Error fetching graphs:', error);
      loadingDiv.innerHTML = `
        <div style="color: #ef4444; text-align: center; padding: 20px;">
          <p style="font-size: 2rem; margin-bottom: 15px;">❌</p>
          <p style="font-size: 1.1rem; margin-bottom: 10px; color: #dc2626; font-weight: 600;">Error al cargar los grafos</p>
          <p style="font-size: 0.9rem; color: #64748b; line-height: 1.4;">
            Verifica la conexión con Virtuoso<br>
            <small style="font-size: 0.8rem; color: #94a3b8;">Endpoint: ${error.message}</small>
          </p>
        </div>
      `;
    }
  }

  /**
   * Populate the graph list with available graphs
   */
  populateGraphList(graphs) {
    const graphList = document.getElementById('graph-list');
    
    if (graphs.length === 0) {
      graphList.innerHTML = `
        <div style="text-align: center; padding: 40px; color: #64748b;">
          <p style="font-size: 2rem; margin-bottom: 15px;">📭</p>
          <p style="font-size: 1.1rem; margin-bottom: 10px; color: #334155;">No hay grafos disponibles</p>
          <p style="font-size: 0.9rem; color: #64748b;">Contacta con el administrador o sube una ontología</p>
        </div>
      `;
      return;
    }

    graphList.innerHTML = graphs.map(graph => {
      // Use complete URI instead of name
      const graphUri = graph.uri || 'URI no disponible';
      
      return `
        <div class="graph-item" data-graph-uri="${graph.uri}">
          <div class="graph-name">${graphUri}</div>
          <div class="graph-stats">
            ${graph.triples || 0} tripletas
          </div>
        </div>
      `;
    }).join('');

    // Add click handlers to graph items
    const self = this; // Preserve context
    graphList.querySelectorAll('.graph-item').forEach((item) => {
      item.addEventListener('click', function() {
        const graphUri = item.dataset.graphUri;
        self.selectGraph(graphUri);
      });
    });
  }

  /**
   * Select a graph and start the main application
   */
  async selectGraph(graphUri, isTemporary = false) {
    console.log('selectGraph called with URI:', graphUri);
    console.log('isTemporary:', isTemporary);
    
    try {
      const savedState = this.loadApplicationState();
      const isSame = savedState && savedState.graphUri === graphUri;
      console.log('Saved state:', savedState);
      console.log('isSameGraph result:', isSame);
      
      // ARREGLO: Siempre verificar si hay una ontología temporal anterior que necesite limpieza
      // cuando se va a cargar un grafo diferente (independientemente de si el nuevo es temporal o no)
      if (savedState && savedState.isTemporary && savedState.graphUri !== graphUri) {
        console.log('Cleaning up previous temporary ontology:', savedState.graphUri);
        await this.cleanupTemporaryOntology(savedState.graphUri);
      } else if (savedState && savedState.isTemporary) {
        console.log('Previous ontology was temporary but same URI, no cleanup needed');
      } else if (savedState) {
        console.log('Previous ontology was not temporary, no cleanup needed');
      } else {
        console.log('No previous state found, no cleanup needed');
      }
      
      if (isSame) {
        await this.handleSameGraphSelection(graphUri);
        return;
      }
      
      if (savedState && savedState.graphUri !== graphUri) {
        await this.handleDifferentGraphSelection(savedState, graphUri);
      }
      
      await this.initializeNewGraph(graphUri, isTemporary);
      
    } catch (error) {
      console.error('Error in selectGraph:', error);
      this.clearApplicationState();
      this.hideKGInitScreen();
      this.showError('Error al cargar el grafo seleccionado: ' + error.message);
    }
  }

  /**
   * Clean up temporary ontology from Virtuoso
   */
  async cleanupTemporaryOntology(graphUri) {
    try {
      console.log('🧹 [CLEANUP] Starting cleanup of temporary ontology:', graphUri);
      
      const response = await fetch(`${CONFIG.BACKEND_URL}/cleanup-ontology`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        credentials: 'omit', // Explicitly don't send credentials
        body: JSON.stringify({ graph_uri: graphUri })
      });

      if (response.ok) {
        const result = await response.json();
        console.log('✅ [CLEANUP] Temporary ontology cleaned up successfully:', result);
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.warn('⚠️  [CLEANUP] Failed to cleanup temporary ontology:', graphUri, 'Error:', errorData.error || response.statusText);
      }
    } catch (error) {
      console.warn('❌ [CLEANUP] Error cleaning up temporary ontology:', error);
    }
  }

  /**
   * Handle selection of the same graph
   */
  async handleSameGraphSelection(graphUri) {
    console.log('Same graph selected, checking if backend cache is still available...');
    
    const cacheAvailable = await this.checkBackendCacheStatus(graphUri);
    console.log('Backend cache available:', cacheAvailable);
    
    this.hideGraphListModal();
    
    if (cacheAvailable) {
      console.log('Backend cache is available, restoring previous state...');
      const content = this.createLoadingMessage(
        'same',
        'Mismo grafo detectado',
        'Restaurando estado anterior sin reentrenar...',
        'Limpiando instancias previas para evitar superposiciones'
      );
      this.showTemporaryMessage(content, 2000);
      
      setTimeout(() => {
        this.startMainApplication(graphUri);
      }, 2000);
    } else {
      console.log('Backend cache not available for same graph, need to retrain...');
      const content = this.createLoadingMessage(
        'expired',
        'Cache expirado',
        'Mismo grafo pero necesita reentrenamiento...'
      );
      this.showTemporaryMessage(content, 2500);
      this.clearApplicationState();
      
      setTimeout(() => {
        this.initializeNewGraph(graphUri);
      }, 2500);
    }
  }

  /**
   * Handle selection of a different graph
   */
  async handleDifferentGraphSelection(savedState, graphUri) {
    console.log('Different graph selected, clearing previous state...');
    console.log('Previous graph:', savedState.graphUri, 'New graph:', graphUri);
    
    // If previous graph was temporary, clean it up
    if (savedState.isTemporary) {
      console.log('Cleaning up previous temporary ontology:', savedState.graphUri);
      await this.cleanupTemporaryOntology(savedState.graphUri);
    }
    
    const content = this.createLoadingMessage(
      'different',
      'Grafo diferente detectado',
      'Limpiando cache anterior y preparando nuevo entrenamiento...'
    );
    this.showTemporaryMessage(content, 2500);
    
    this.clearApplicationState();
    await this.clearBackendCache(savedState.graphUri);
  }

  /**
   * Initialize a new graph
   */
  async initializeNewGraph(graphUri, isTemporary = false) {
    this.hideGraphListModal();
    this.showKGInitScreen(graphUri);
    
    const response = await fetch(`${CONFIG.BACKEND_URL}/select-graph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        graph_uri: graphUri,
        is_temporary: isTemporary 
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to select graph: ${response.statusText}`);
    }
    
    const result = await response.json();
    if (result.status === 'error') {
      throw new Error(result.error);
    }
    
    console.log('Graph selected on backend, initialization started:', result.message);
    this.startProgressPolling(graphUri, isTemporary);
  }

  /**
   * Show upload modal
   */
  showUploadModal() {
    const modal = document.getElementById('upload-modal');
    modal.classList.remove('hidden');
    setTimeout(() => modal.classList.add('show'), 10);
  }

  /**
   * Hide upload modal
   */
  hideUploadModal() {
    const modal = document.getElementById('upload-modal');
    modal.classList.remove('show');
    setTimeout(() => modal.classList.add('hidden'), 300);
  }

  /**
   * Hide graph list modal
   */
  hideGraphListModal() {
    const modal = document.getElementById('graph-list-modal');
    modal.classList.remove('show');
    setTimeout(() => modal.classList.add('hidden'), 300);
  }

  /**
   * Setup upload area functionality
   */
  setupUploadArea() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('ontology-file-input');

    if (!uploadArea || !fileInput) return;

    // Click to select file
    uploadArea.addEventListener('click', () => {
      fileInput.click();
    });

    // File input change handler
    fileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) {
        this.handleFileUpload(file);
      }
    });

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', (e) => {
      e.preventDefault();
      uploadArea.classList.add('drag-over');
    });

    uploadArea.addEventListener('dragleave', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
    });

    uploadArea.addEventListener('drop', (e) => {
      e.preventDefault();
      uploadArea.classList.remove('drag-over');
      
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        this.handleFileUpload(files[0]);
      }
    });
  }

  /**
   * Handle file upload
   */
  async handleFileUpload(file) {
    console.log('Uploading file:', file.name);
    
    // Validate file type
    const validExtensions = ['.owl', '.ttl', '.rdf', '.n3'];
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!validExtensions.includes(fileExtension)) {
      this.showError('Formato de archivo no soportado. Usa: .owl, .ttl, .rdf, .n3');
      return;
    }

    // Show upload progress
    const uploadArea = document.getElementById('upload-area');
    uploadArea.innerHTML = `
      <div class="upload-icon">⏳</div>
      <p>Subiendo ${file.name}...</p>
      <div class="upload-progress">
        <div class="upload-progress-bar" style="width: 0%"></div>
      </div>
      <p style="font-size: 0.9rem; color: #888; margin-top: 10px;">
        Validando formato y parseando contenido...
      </p>
    `;

    // Simulate progress animation
    const progressBar = uploadArea.querySelector('.upload-progress-bar');
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress > 90) progress = 90; // Stop at 90% until real completion
      progressBar.style.width = `${progress}%`;
    }, 200);

    try {
      // Here you would implement the actual file upload
      const result = await this.uploadOntologyFile(file);
      
      // Complete progress
      clearInterval(progressInterval);
      progressBar.style.width = '100%';
      
      // Success
      uploadArea.innerHTML = `
        <div class="upload-icon">✅</div>
        <p>¡Archivo subido exitosamente!</p>
        <p style="color: #4caf50; font-size: 0.9rem;">
          ${result.triples} tripletas procesadas
        </p>
        <p style="color: #4caf50; font-size: 0.9rem;">Iniciando aplicación...</p>
      `;

      await this.delay(1500);
      this.hideUploadModal();
      
      // Use selectGraph with temporary flag for uploaded ontologies
      await this.selectGraph(result.graph_uri, true); // true = temporary ontology
      
    } catch (error) {
      console.error('Upload error:', error);
      
      // Stop progress animation
      if (typeof progressInterval !== 'undefined') {
        clearInterval(progressInterval);
      }
      
      // Reset upload area
      uploadArea.innerHTML = `
        <div class="upload-icon">📁</div>
        <p>Arrastra tu archivo de ontología aquí o haz clic para seleccionar</p>
        <p class="upload-formats">Formatos soportados: .owl, .ttl, .rdf, .n3</p>
        <div style="color: #ff6b6b; margin-top: 10px; padding: 10px; background: rgba(255, 107, 107, 0.1); border-radius: 5px;">
          ❌ Error: ${error.message}
        </div>
      `;
    }
  }

  /**
   * Upload ontology file to server
   */
  async uploadOntologyFile(file) {
    const formData = new FormData();
    formData.append('ontology', file);

    const response = await fetch(`${CONFIG.BACKEND_URL}/upload-ontology`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.error || `Upload failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Start the main application
   */
  startMainApplication(source) {
    console.log('Starting main application with source:', source);
    
    // Hide loading screen immediately
    console.log('Hiding loading screen...');
    this.loadingScreen.classList.add('hidden');
    
    // Fade out start screen
    this.startScreen.classList.remove('show');
    console.log('Removed show class from start screen');
    
    setTimeout(() => {
      console.log('Starting screen fade-out timeout callback');
      this.startScreen.classList.add('hidden');
      const mainApp = document.getElementById('main-app');
      if (mainApp) {
        console.log('Found main-app element, removing hidden class');
        mainApp.classList.remove('hidden');
      } else {
        console.error('main-app element not found!');
      }
      
      // Allow scrolling again
      document.body.style.overflow = '';
      
      // Show main app elements that were initially hidden
      this.showMainAppElements();
      console.log('Called showMainAppElements');
      
      // Initialize the main graph application
      console.log('About to call initializeMainApp');
      this.initializeMainApp(source);
    }, 800);
  }

  /**
   * Initialize the main graph application
   */
  async initializeMainApp(source) {
    try {
      // This should call your existing graph initialization code
      console.log('Main application initialized with source:', source);
      console.log('About to dispatch appReady event');
      
      // You can dispatch a custom event that your main app can listen to
      window.dispatchEvent(new CustomEvent('appReady', { 
        detail: { source } 
      }));
      
      console.log('appReady event dispatched successfully');
      
    } catch (error) {
      console.error('Error initializing main app:', error);
      this.showError('Error al inicializar la aplicación principal');
    }
  }

  /**
   * Show all main application elements
   */
  showMainAppElements() {
    // Remove the loading class from body to show main app elements
    document.body.classList.remove('loading');
  }

  /**
   * Add visual click effect to node
   */
  addNodeClickEffect(nodeId) {
    const node = document.getElementById(nodeId);
    if (node) {
      const originalTransform = node.style.transform;
      node.style.transform = 'scale(0.95)';
      setTimeout(() => {
        node.style.transform = originalTransform || 'scale(1)';
      }, 150);
    }
  }

  /**
   * Transition to main application
   */
  transitionToMainApp() {
    const mainApp = document.getElementById('main-app');
    if (mainApp) {
      this.startScreen.style.transition = 'opacity 0.8s ease-out';
      this.startScreen.style.opacity = '0';
      
      setTimeout(() => {
        this.startScreen.classList.add('hidden');
        mainApp.classList.remove('hidden');
      }, 800);
    }
  }

  /**
   * Show error message
   */
  showError(message) {
    console.error(message);
    
    // Create a visual error notification
    const errorDiv = document.createElement('div');
    errorDiv.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: linear-gradient(135deg, #ff6b6b, #ff5252);
      color: white;
      padding: 15px 20px;
      border-radius: 10px;
      box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
      z-index: 12000;
      font-family: 'Space Grotesk', sans-serif;
      font-weight: 500;
      max-width: 300px;
      animation: slideInRight 0.3s ease-out;
    `;
    errorDiv.textContent = message;

    // Add CSS animations if not already present
    if (!document.querySelector('#error-animations')) {
      const style = document.createElement('style');
      style.id = 'error-animations';
      style.textContent = `
        @keyframes slideInRight {
          from { opacity: 0; transform: translateX(100%); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes slideOutRight {
          from { opacity: 1; transform: translateX(0); }
          to { opacity: 0; transform: translateX(100%); }
        }
      `;
      document.head.appendChild(style);
    }

    document.body.appendChild(errorDiv);

    // Remove after 5 seconds
    setTimeout(() => {
      errorDiv.style.animation = 'slideOutRight 0.3s ease-out';
      setTimeout(() => errorDiv.remove(), 300);
    }, 5000);
  }

  /**
   * Fetch available graphs from Virtuoso
   */
  async fetchAvailableGraphs() {
    try {
      const response = await fetch('http://192.168.216.102:32323/available-graphs');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      if (data.status === 'error') {
        throw new Error(data.error);
      }
      
      return data.graphs || [];
    } catch (error) {
      console.error('Error fetching graphs from server:', error);
      
      // Fallback to mock data if server is unavailable
      console.warn('Using fallback mock data');
      return [
        {
          name: 'Ontología de Pizza',
          description: 'Ejemplo clásico de ontología con pizzas y sus ingredientes',
          uri: 'http://www.co-ode.org/ontologies/pizza/pizza.owl',
          triples: 945
        },
        {
          name: 'Instancias Local', 
          description: 'Instancias definidas localmente en el sistema',
          uri: 'http://example.org/local/instances',
          triples: 156
        },
        {
          name: 'Ontología Final',
          description: 'Ontología principal del proyecto',
          uri: 'http://example.org/ontology/final',
          triples: 2341
        }
      ];
    }
  }

  /**
   * Show full-screen KG initialization
   */
  showKGInitScreen(graphUri) {
    // Hide startup screen
    this.startScreen.classList.remove('show');
    this.startScreen.classList.add('hidden');
    
    // Show KG initialization screen
    const kgInitScreen = document.getElementById('kg-init-screen');
    const graphNameElement = document.getElementById('kg-init-graph-name');
    
    if (kgInitScreen && graphNameElement) {
      // Set graph name
      graphNameElement.textContent = graphUri;
      
      // Initialize progress to 0% immediately
      this.updateKGProgress({
        progress: 0,
        current_step: 'Conectando con el servidor...',
        status: 'idle',
        logs: []
      });
      
      // Add initial connecting message after a short delay
      setTimeout(() => {
        this.updateKGProgress({
          progress: 0,
          current_step: 'Enviando solicitud de inicialización...',
          status: 'idle',
          logs: ['Estableciendo conexión con el backend...']
        });
      }, 500);
      
      // Show with animation
      kgInitScreen.classList.remove('hidden');
      setTimeout(() => {
        kgInitScreen.classList.add('show');
      }, 50);
    }
  }

  /**
   * Hide KG initialization screen
   */
  hideKGInitScreen() {
    const kgInitScreen = document.getElementById('kg-init-screen');
    if (kgInitScreen) {
      kgInitScreen.classList.remove('show');
      setTimeout(() => {
        kgInitScreen.classList.add('hidden');
      }, 600);
    }
  }
  // ===== PROGRESS TRACKING METHODS =====

  /**
   * Start polling for initialization progress
   */
  startProgressPolling(graphUri, isTemporary = false) {
    const pollInterval = 1000; // Poll every 1 second
    let pollCount = 0;
    const maxPolls = 300; // Maximum 5 minutes
    
    const pollProgress = async () => {
      try {
        const response = await fetch('http://192.168.216.102:5000/initialize-progress');
        if (!response.ok) {
          throw new Error(`Progress polling failed: ${response.statusText}`);
        }
        
        const progress = await response.json();
        
        this.updateKGProgress(progress);
        
        // Check if initialization is complete
        if (progress.status === 'completed') {
          console.log('✅ Initialization completed, transitioning to main app');
          
          // Save the application state with temporary flag
          this.saveApplicationState(graphUri, isTemporary);
          
          setTimeout(() => {
            this.hideKGInitScreen();
            setTimeout(() => {
              this.startMainApplication(graphUri);
            }, 600);
          }, 1000);
          return;
        }
        
        // Check for errors
        if (progress.status === 'error') {
          console.log('❌ Initialization error:', progress.error_message);
          this.hideKGInitScreen();
          this.showError('Error durante la inicialización: ' + (progress.error_message || 'Error desconocido'));
          return;
        }
        
        // Continue polling if still running
        if (progress.status === 'running' && pollCount < maxPolls) {
          pollCount++;
          setTimeout(pollProgress, pollInterval);
        } else if (pollCount >= maxPolls) {
          console.log('⏰ Polling timeout reached');
          this.hideKGInitScreen();
          this.showError('Timeout: La inicialización está tomando demasiado tiempo.');
        }
        
      } catch (error) {
        console.error('❌ Error polling progress:', error);
        this.hideKGInitScreen();
        this.showError('Error al obtener el progreso de inicialización: ' + error.message);
      }
    };
    
    // Start polling immediately
    setTimeout(pollProgress, 200);
  }

  /**
   * Update KG initialization progress display
   */
  updateKGProgress(progress) {
    const progressFill = document.getElementById('kg-progress-fill');
    const progressStep = document.getElementById('kg-progress-step');
    const progressPercentage = document.getElementById('kg-progress-percentage');
    const statusValue = document.getElementById('kg-status-value');
    const logsContainer = document.getElementById('kg-logs-container');
    
    // Update progress bar
    if (progressFill) {
      const progressValue = progress.progress || 0;
      progressFill.style.width = `${progressValue}%`;
    }
    
    // Update step description
    if (progressStep) {
      progressStep.textContent = progress.current_step || 'Preparando...';
    }
    
    // Update percentage
    if (progressPercentage) {
      progressPercentage.textContent = `${progress.progress || 0}%`;
    }
    
    // Update status
    if (statusValue) {
      let statusText = 'Conectando...';
      if (progress.status === 'running') {
        statusText = 'En progreso';
      } else if (progress.status === 'completed') {
        statusText = 'Completado';
      } else if (progress.status === 'error') {
        statusText = 'Error';
      } else if (progress.status === 'idle') {
        statusText = 'Preparando...';
      }
      statusValue.textContent = statusText;
    }
    
    // Update logs
    if (logsContainer && progress.logs && progress.logs.length > 0) {
      // Clear existing logs except the startup message if it's the first update with no logs
      if (progress.logs.length === 0 && progress.status === 'idle') {
        // Keep the startup log
        return;
      }
      
      // Clear container and rebuild with new logs
      const startupEntry = logsContainer.querySelector('.kg-log-entry.startup');
      logsContainer.innerHTML = '';
      
      // Add startup entry back if no logs yet
      if (progress.logs.length === 0) {
        if (startupEntry) {
          logsContainer.appendChild(startupEntry);
        }
        return;
      }
      
      // Add all logs from progress
      progress.logs.forEach(log => {
        const logEntry = document.createElement('div');
        logEntry.className = 'kg-log-entry';
        
        // Extract timestamp and message
        const timeMatch = log.match(/^\[([^\]]+)\]/);
        const timestamp = timeMatch ? timeMatch[1] : 'Sistema';
        const message = log.replace(/^\[[^\]]+\]\s*/, '');
        
        // Determine log type for styling
        let logType = '';
        if (log.includes('ERROR')) {
          logType = 'error';
        } else if (log.includes('WARNING')) {
          logType = 'warning';
        } else if (log.includes('✅') || log.includes('SUCCESS')) {
          logType = 'success';
        }
        
        if (logType) {
          logEntry.classList.add(logType);
        }
        
        logEntry.innerHTML = `
          <span class="kg-log-time">[${timestamp}]</span>
          <span class="kg-log-message">${message}</span>
        `;
        
        logsContainer.appendChild(logEntry);
      });
      
      // Auto-scroll to bottom
      logsContainer.scrollTop = logsContainer.scrollHeight;
    }
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  console.log('Creating InteractiveStartupManager instance');
  window.interactiveStartupManager = new InteractiveStartupManager();
  console.log('InteractiveStartupManager instance created and registered globally');
});

// Global function to clear application state and restart
window.clearApplicationStateAndRestart = function() {
  if (window.interactiveStartupManager) {
    window.interactiveStartupManager.clearApplicationState();
  }
  location.reload();
};

// Emergency function to clear all localStorage and restart
window.emergencyClearState = function() {
  try {
    localStorage.removeItem('atmentis_app_state');
    console.log('Emergency: localStorage cleared');
    location.reload();
  } catch (error) {
    console.error('Emergency clear failed:', error);
  }
};

// Debug function to check current state
window.debugAppState = function() {
  try {
    const state = localStorage.getItem('atmentis_app_state');
    console.log('Current localStorage state:', state ? JSON.parse(state) : 'No state saved');
    if (window.interactiveStartupManager) {
      console.log('StartupManager instance available');
      console.log('Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(window.interactiveStartupManager)));
    } else {
      console.log('No StartupManager instance found');
    }
  } catch (error) {
    console.error('Debug failed:', error);
  }
};

// Global function to show main menu
window.showMainMenu = function() {
  console.log('showMainMenu called');
  if (window.interactiveStartupManager) {
    console.log('Found startup manager, calling showMainMenu');
    window.interactiveStartupManager.showMainMenu();
  } else {
    console.error('No startup manager found! Cannot show main menu');
    // Fallback: reload the page to restart the application
    location.reload();
  }
};

// Global function to show confirmation banner for back to menu
window.showBackToMenuConfirmation = function() {
  console.log('showBackToMenuConfirmation called');
  const banner = document.getElementById('back-to-menu-confirmation');
  if (banner) {
    console.log('Found banner, showing confirmation');
    banner.style.display = 'flex';
    setTimeout(() => banner.classList.add('show'), 10);
  } else {
    console.error('Banner element not found!');
  }
};

// Global function to hide confirmation banner
window.hideBackToMenuConfirmation = function() {
  console.log('hideBackToMenuConfirmation called');
  const banner = document.getElementById('back-to-menu-confirmation');
  if (banner) {
    console.log('Found banner, hiding confirmation');
    banner.classList.remove('show');
    setTimeout(() => banner.style.display = 'none', 300);
  } else {
    console.error('Banner element not found!');
  }
};
// Debug function to manually test click handlers
window.testMenuClicks = function() {
  console.log('Testing menu clicks...');
  
  const loadGraphsNode = document.getElementById('load-graphs-node');
  const uploadNode = document.getElementById('upload-node');
  
  if (loadGraphsNode) {
    console.log('Simulating click on load-graphs-node...');
    loadGraphsNode.click();
  } else {
    console.error('load-graphs-node not found!');
  }
  
  setTimeout(() => {
    if (uploadNode) {
      console.log('Simulating click on upload-node...');
      uploadNode.click();
    } else {
      console.error('upload-node not found!');
    }
  }, 1000);
};

// ===========================================================================
// AUTOMATIC CLEANUP FOR TEMPORARY ONTOLOGIES
// ===========================================================================

/**
 * Cleanup temporary ontologies when page is closed/refreshed
 */
async function cleanupOnPageUnload() {
  if (!window.interactiveStartupManager) return;
  
  const savedState = window.interactiveStartupManager.loadApplicationState();
  if (savedState && savedState.isTemporary) {
    console.log('Page unload detected - cleaning up temporary ontology:', savedState.graphUri);
    
    // Use sendBeacon for reliable cleanup during page unload
    const cleanupData = JSON.stringify({ graph_uri: savedState.graphUri });
    
    try {
      if (navigator.sendBeacon) {
        navigator.sendBeacon(
          `${CONFIG.BACKEND_URL}/cleanup-ontology`,
          new Blob([cleanupData], { type: 'application/json' })
        );
      } else {
        // Fallback for older browsers
        await window.interactiveStartupManager.cleanupTemporaryOntology(savedState.graphUri);
      }
      
      // Clear the state to prevent double cleanup
      window.interactiveStartupManager.clearApplicationState();
    } catch (error) {
      console.warn('Error during page unload cleanup:', error);
    }
  }
}

// Add event listeners for page unload
window.addEventListener('beforeunload', cleanupOnPageUnload);
window.addEventListener('pagehide', cleanupOnPageUnload);

// Also cleanup on visibility change (when tab becomes hidden)
document.addEventListener('visibilitychange', () => {
  if (document.visibilityState === 'hidden') {
    cleanupOnPageUnload();
  }
});

console.log('✅ Automatic cleanup for temporary ontologies initialized');
