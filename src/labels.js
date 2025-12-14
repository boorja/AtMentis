import { LabelRenderer } from "@interacta/css-labels";

export class CosmosLabels {
  constructor(div, nodes) {
    // Validate that div element is provided and not null
    if (!div) {
      throw new Error('CosmosLabels: Container div element is required and cannot be null');
    }
    
    console.log('Initializing CosmosLabels with div:', div, 'and nodes:', nodes?.length || 0);
    
    this.labelRenderer = new LabelRenderer(div, { pointerEvents: "none" });
    this.labels = [];
    this.hidden = false; // Track if labels are hidden
    this.visibleNodeIds = null; // Set of node IDs that should remain visible when partially hidden
    // Crear un mapa de nodos para buscar por ID
    this.nodeMap = new Map((nodes || []).map(node => [node.id, node]));
  }

  // Helper function to extract the last part of a URL or path
  extractShortName(text) {
    // If it's a URL or path with slashes
    if (typeof text === 'string' && text.includes('/')) {
      // Split by slash and get the last non-empty part
      const parts = text.split('/').filter(part => part.trim() !== '');
      return parts.length > 0 ? parts[parts.length - 1] : text;
    }
    return text;
  }

  /**
   * Updates the internal map of nodes.
   * @param {Array} nodes - The new array of node objects.
   */
  updateNodeMap(nodes) {
    console.log("Updating node map in CosmosLabels with", nodes.length, "nodes.");
    this.nodeMap = new Map(nodes.map(node => [node.id, node]));
  }

  /**
   * Hide all labels except for specified nodes (typically when context menu is shown)
   * @param {Set} visibleNodeIds - Set of node IDs that should keep their labels visible
   */
  hideExcept(visibleNodeIds = new Set()) {
    this.hidden = true;
    this.visibleNodeIds = visibleNodeIds;
    // Trigger an update to redraw only the visible labels
    if (this.lastGraph) {
      this.update(this.lastGraph);
    }
  }

  /**
   * Show all labels again (typically when context menu is hidden)
   */
  show() {
    this.hidden = false;
    this.visibleNodeIds = null;
    // Trigger an update to redraw all labels
    if (this.lastGraph) {
      this.update(this.lastGraph);
    }
  }

  /**
   * Cleanup all labels and reset internal state
   */
  cleanup() {
    console.log('Cleaning up CosmosLabels instance');
    this.hidden = false;
    this.visibleNodeIds = null;
    this.lastGraph = null;
    this.labels = [];
    this.nodeMap.clear();
    
    // Clear the label renderer
    if (this.labelRenderer) {
      try {
        this.labelRenderer.setLabels([]);
        this.labelRenderer.draw(true);
      } catch (error) {
        console.warn('Error clearing label renderer:', error);
      }
    }
  }

  update(graph) {
    this.lastGraph = graph; // Store graph reference for show() method
    
    const trackedNodesPositions = graph.getTrackedNodePositionsMap();
    const newLabels = [];

    trackedNodesPositions.forEach((positions, nodeId) => {
      // If labels are hidden, only show labels for nodes in visibleNodeIds
      if (this.hidden && (!this.visibleNodeIds || !this.visibleNodeIds.has(nodeId))) {
        return; // Skip this node's label
      }

      if (!positions) {
        console.warn(`No positions found for node: ${nodeId}`);
        return;
      }

      const screenPosition = graph.spaceToScreenPosition([
        positions[0] ?? 0,
        positions[1] ?? 0
      ]);

      const radius = graph.spaceToScreenRadius(graph.getNodeRadiusById(nodeId) ?? 5);
      // Usar el mapa interno en lugar de graph.getNodeById
      const node = this.nodeMap.get(nodeId);
      // Get the raw label text
      let rawLabelText = node?.label || String(nodeId);
      // Extract just the last meaningful part
      const labelText = this.extractShortName(rawLabelText);

      newLabels.push({
        id: nodeId,
        text: labelText,
        x: screenPosition[0],
        y: screenPosition[1] - (radius * 0.2), // Reducido para acercar la etiqueta al nodo
        opacity: 1,
        fontSize: 14,
        color: 'white'
      });
    });

    // console.log('Labels array:', newLabels);
    this.labels = newLabels;
    this.labelRenderer.setLabels(this.labels);
    this.labelRenderer.draw(true);
    // console.log("Node labels:", newLabels.map(l => l.text));
    // console.log("Posiciones:", newLabels.map(l => [l.x, l.y]));
  }
}