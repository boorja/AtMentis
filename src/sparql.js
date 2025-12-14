// sparql.js

// Virtuoso configuration object
const virtuosoConfig = {
  endpoint: "http://192.168.216.102:8890/sparql-auth", // Your Virtuoso SPARQL endpoint
  database: "https://khaos.example.org/v4", // The named graph URI
  username: "dba", // Your Virtuoso username
  password: "password", // Your Virtuoso password
  backendUrl: "http://192.168.216.102:32323/query_virtuoso" // URL of your Python backend endpoint
};

/**
 * Update the graph URI configuration
 * @param {string} graphUri - The new graph URI to use for queries
 */
export const updateGraphUri = (graphUri) => {
  if (graphUri && typeof graphUri === 'string') {
    console.log(`[SPARQL] Updating graph URI from "${virtuosoConfig.database}" to "${graphUri}"`);
    virtuosoConfig.database = graphUri;
  } else {
    console.warn(`[SPARQL] Invalid graph URI provided:`, graphUri);
  }
};

/**
 * Helper function to execute a SPARQL query against the Virtuoso backend.
 * @param {string} sparqlQuery - The SPARQL query string.
 * @param {string} graphUri - The specific graph URI to query.
 * @returns {Promise<object>} - A promise that resolves to the graph data { nodes, edges, literals }.
 * @throws {Error} - Throws an error if the network request or query fails.
 */
const queryVirtuoso = async (sparqlQuery, graphUri = virtuosoConfig.database) => {
  console.log(`[queryVirtuoso] Called with graphUri: ${graphUri}`);
  console.log(`[queryVirtuoso] Original query length: ${sparqlQuery.length} characters`);
  
  // Ensure we have a valid graphUri, fallback to default if null/undefined
  if (!graphUri || graphUri === 'null') {
    graphUri = virtuosoConfig.database;
    console.log(`[queryVirtuoso] Invalid graphUri detected, using default: ${graphUri}`);
  }
  
  let finalQuery = sparqlQuery; // Default to original query if modification fails

  const whereMatch = sparqlQuery.match(/WHERE\s*{/i);

  if (whereMatch && whereMatch.index !== undefined) {
    const whereClauseStartIndex = whereMatch.index; // Index of "WHERE {"
    const whereContentStartIndex = whereClauseStartIndex + whereMatch[0].length; // Index right after "WHERE {"

    let braceCount = 1;
    let whereContentEndIndex = -1; // Will point to the closing '}' of the main WHERE clause

    for (let i = whereContentStartIndex; i < sparqlQuery.length; i++) {
      if (sparqlQuery[i] === '{') {
        braceCount++;
      } else if (sparqlQuery[i] === '}') {
        braceCount--;
      }
      if (braceCount === 0) {
        whereContentEndIndex = i; // Found the matching '}' for the main WHERE clause
        break;
      }
    }

    if (whereContentEndIndex !== -1) {
      const queryPrefix = sparqlQuery.substring(0, whereClauseStartIndex); // Prefixes, SELECT, etc., before "WHERE {"
      const originalWhereKeywordAndOpeningBrace = sparqlQuery.substring(whereClauseStartIndex, whereContentStartIndex); // "WHERE {"
      const originalWhereContent = sparqlQuery.substring(whereContentStartIndex, whereContentEndIndex); // Content inside "WHERE { ... }"
      const originalWhereClosingBraceAndSuffix = sparqlQuery.substring(whereContentEndIndex); // "}" and anything after (LIMIT, ORDER BY)
      
      finalQuery = `${queryPrefix}${originalWhereKeywordAndOpeningBrace} GRAPH <${graphUri}> { ${originalWhereContent} } ${originalWhereClosingBraceAndSuffix}`;
    } else {
      console.warn("Could not properly inject GRAPH clause: Main WHERE clause closing brace not found. Using original query.");
      // finalQuery remains sparqlQuery
    }
  } else {
    console.warn("Could not find WHERE clause to inject GRAPH into. Using original query.");
    // finalQuery remains sparqlQuery
  }

  const requestData = {
    virtuoso_endpoint: virtuosoConfig.endpoint,
    virtuoso_database: graphUri,
    virtuoso_username: virtuosoConfig.username,
    virtuoso_password: virtuosoConfig.password,
    query: finalQuery // Send the correctly constructed query
  };

  console.log("Executing SPARQL Query (with GRAPH clause):\n", finalQuery); // Log the query being sent
  console.log(`[queryVirtuoso] Request data:`, requestData);
  console.log(`[queryVirtuoso] Backend URL: ${virtuosoConfig.backendUrl}`);

  const response = await fetch(virtuosoConfig.backendUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(requestData)
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Virtuoso query failed:", response.status, errorText);
    // Include the query that failed in the error message for easier debugging
    throw new Error(`Network response was not ok: ${response.statusText} - ${errorText}\nQuery:\n${finalQuery}`);
  }

  const data = await response.json();
  console.log("Raw data received from backend:", data); // Log raw response
  return data;
};

/**
 * Fetches an initial overview of the ontology, focusing on the class hierarchy.
 * @param {string} source - Optional source identifier for the graph (graph URI or uploaded file)
 * @returns {Promise<object>} - Graph data { nodes, edges, literals }.
 */
export const fetchInitialGraph = async (source = null) => {
  console.log("Fetching top-level classes for initial view...", source ? `from source: ${source}` : "");
  
  // If source is provided and looks like a file upload, we might handle it differently
  if (source && !source.startsWith('http')) {
    console.log(`Loading from uploaded file: ${source}`);
    // For uploaded files, we might still use the same query but note the source
  }
  
  // Query without GRAPH clause - queryVirtuoso will add it.
  const sparqlQuery = `
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?source ?link ?target ?literal_type
    WHERE {
        # Universal strategy: Find ONLY the highest-level classes (true roots)
        ?source rdf:type owl:Class .
        
        # Universal filters that apply to any ontology
        FILTER (!isBlank(?source))
        FILTER (?source != owl:Thing)
        FILTER (?source != owl:Nothing)
        FILTER (?source != rdfs:Resource)
        FILTER (?source != rdfs:Class)
        
        # Exclude W3C standard vocabularies (universal exclusion)
        FILTER (!CONTAINS(STR(?source), "http://www.w3.org/1999/"))
        FILTER (!CONTAINS(STR(?source), "http://www.w3.org/2000/"))
        FILTER (!CONTAINS(STR(?source), "http://www.w3.org/2001/"))
        FILTER (!CONTAINS(STR(?source), "http://www.w3.org/2002/"))
        FILTER (!CONTAINS(STR(?source), "XMLSchema"))
        FILTER (!CONTAINS(STR(?source), "dublincore"))
        
        # STRICT hierarchy roots - classes that have NO meaningful superclasses
        FILTER NOT EXISTS {
            ?source rdfs:subClassOf ?superClass .
            FILTER (?superClass != owl:Thing)
            FILTER (?superClass != ?source)  # Exclude self-references
            FILTER (!isBlank(?superClass))   # Exclude anonymous classes
        }
        
        # Additional strict filtering: Exclude overly specific or derived classes
        
        # Exclude anonymous/complex class expressions
        FILTER NOT EXISTS {
            ?source owl:equivalentClass ?complex .
            ?complex rdf:type owl:Class .
            FILTER (isBlank(?complex))
        }
        
        # Exclude classes that are clearly restrictions or composed classes
        FILTER NOT EXISTS {
            ?source rdfs:subClassOf ?restriction .
            ?restriction rdf:type owl:Restriction .
        }
        
        # IMPORTANT: Include classes that are meaningful in the ontology
        # This includes classes with subclasses, instances, or that are used in relationships
        {
            # Classes that have direct subclasses
            ?subClass rdfs:subClassOf ?source .
            FILTER (?subClass != ?source)
            FILTER (!isBlank(?subClass))
        } UNION {
            # OR classes that have instances
            ?instance rdf:type ?source .
            FILTER (!isBlank(?instance))
        } UNION {
            # OR classes that are used in important structural relationships
            { ?prop rdfs:domain ?source . } UNION { ?prop rdfs:range ?source . }
        } UNION {
            # OR classes that are direct children of owl:Thing and have documentation
            # This captures important top-level classes that might not have subclasses yet
            ?source rdfs:subClassOf owl:Thing .
            FILTER NOT EXISTS {
                ?source rdfs:subClassOf ?intermediate .
                ?intermediate rdfs:subClassOf owl:Thing .
                FILTER (?intermediate != owl:Thing)
                FILTER (?intermediate != ?source)
                FILTER (!isBlank(?intermediate))
            }
            # Must have some form of documentation to be considered important
            {
                ?source rdfs:label ?label .
            } UNION {
                ?source rdfs:comment ?comment .
            }
        }

        # Get human-readable labels as literal data for these top-level classes
        OPTIONAL { 
            ?source rdfs:label ?label_en . 
            FILTER (LANG(?label_en) = "en" || LANG(?label_en) = "")
        }
        OPTIONAL { 
            ?source rdfs:label ?label_any . 
            FILTER (!BOUND(?label_en))
        }
        OPTIONAL { 
            ?source rdfs:comment ?comment_en . 
            FILTER (LANG(?comment_en) = "en" || LANG(?comment_en) = "")
        }
        
        # Return the class labels as literal properties (marked with literal_type)
        BIND(rdfs:label AS ?link)
        BIND(COALESCE(
            ?label_en,
            ?label_any, 
            STRAFTER(STR(?source), "#"), 
            STRAFTER(STR(?source), "/"),
            STR(?source)
        ) as ?target)
        BIND(xsd:string AS ?literal_type)
    }
    ORDER BY STRLEN(STR(?source)) ?source
    LIMIT 15  # Increased limit for top-level classes
  `;
  try {
    const data = await queryVirtuoso(sparqlQuery, virtuosoConfig.database);
    
    // If we get too many classes, filter to show only the most important ones
    if (data && data.nodes && data.nodes.length > 15) {
      console.log(`Query returned ${data.nodes.length} top-level classes, using fallback for better filtering...`);
      return await fetchInitialGraphFallback();
    }
    
    // If we don't get enough classes with the strict query, try a more permissive approach
    if (!data || !data.nodes || data.nodes.length < 1) {
      console.log("Few top-level classes found, trying fallback query...");
      return await fetchInitialGraphFallback();
    }
    
    return data && data.nodes ? data : { nodes: [], edges: [], literals: {} };
  } catch (error) {
    console.error('Error fetching initial graph:', error);
    console.log("Trying fallback query due to error...");
    return await fetchInitialGraphFallback();
  }
};

/**
 * Fallback strategy for fetching initial graph when the main query returns few results.
 * This function tries different approaches to find important classes in the ontology.
 * @returns {Promise<object>} - Graph data { nodes, edges, literals }.
 */
const fetchInitialGraphFallback = async () => {
  console.log("Using fallback strategy to find important classes...");
  
  const fallbackQuery = `
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

    SELECT DISTINCT ?source ?link ?target ?literal_type
    WHERE {
        # Simplified fallback: Find ONLY the most important top-level classes
        {
            # Strategy 1: Classes that are superclasses of many others (high-level concepts)
            ?source rdf:type owl:Class .
            
            # Basic exclusions
            FILTER (!isBlank(?source))
            FILTER (?source != owl:Thing)
            FILTER (?source != owl:Nothing)
            FILTER (?source != rdfs:Resource)
            FILTER (?source != rdfs:Class)
            FILTER (!CONTAINS(STR(?source), "http://www.w3.org/"))
            
            # Must have NO superclasses (true roots)
            FILTER NOT EXISTS { 
                ?source rdfs:subClassOf ?super . 
                FILTER (?super != owl:Thing)
                FILTER (?super != ?source)
                FILTER (!isBlank(?super))
            }
            
            # Must have subclasses (important hierarchy roots)
            ?subClass rdfs:subClassOf ?source .
            FILTER (?subClass != ?source)
            FILTER (!isBlank(?subClass))
        }
        UNION
        {
            # Strategy 2: Well-documented root concepts (often indicate main domains)
            ?source rdf:type owl:Class .
            
            FILTER (!isBlank(?source))
            FILTER (?source != owl:Thing)
            FILTER (?source != owl:Nothing)
            FILTER (!CONTAINS(STR(?source), "http://www.w3.org/"))
            
            # Must have NO superclasses
            FILTER NOT EXISTS { 
                ?source rdfs:subClassOf ?super . 
                FILTER (?super != owl:Thing)
                FILTER (?super != ?source)
                FILTER (!isBlank(?super))
            }
            
            # Must have documentation (indicates intentional high-level concepts)
            { 
                ?source rdfs:label ?label . 
                FILTER (LANG(?label) = "en" || LANG(?label) = "")
            } UNION { 
                ?source rdfs:comment ?comment . 
                FILTER (LANG(?comment) = "en" || LANG(?comment) = "")
            }
            
            # Ensure it's used in domain/range, has instances, or is a documented direct child of Thing
            {
                ?instance rdf:type ?source .
                FILTER (!isBlank(?instance))
            } UNION {
                { ?prop rdfs:domain ?source . } UNION { ?prop rdfs:range ?source . }
            } UNION {
                # Include direct children of owl:Thing that are documented
                ?source rdfs:subClassOf owl:Thing .
                FILTER NOT EXISTS {
                    ?source rdfs:subClassOf ?intermediate .
                    ?intermediate rdfs:subClassOf owl:Thing .
                    FILTER (?intermediate != owl:Thing)
                    FILTER (?intermediate != ?source)
                    FILTER (!isBlank(?intermediate))
                }
            }
        }
        UNION
        {
            # Strategy 3: Include all direct children of owl:Thing that are meaningful
            # This catches important ontology classes that might not have instances yet
            ?source rdf:type owl:Class .
            ?source rdfs:subClassOf owl:Thing .
            
            FILTER (!isBlank(?source))
            FILTER (?source != owl:Thing)
            FILTER (?source != owl:Nothing)
            FILTER (!CONTAINS(STR(?source), "http://www.w3.org/"))
            
            # Must be a direct child of owl:Thing
            FILTER NOT EXISTS {
                ?source rdfs:subClassOf ?intermediate .
                ?intermediate rdfs:subClassOf owl:Thing .
                FILTER (?intermediate != owl:Thing)
                FILTER (?intermediate != ?source)
                FILTER (!isBlank(?intermediate))
            }
            
            # Must have some documentation or be used somewhere
            {
                { ?source rdfs:label ?label . } UNION { ?source rdfs:comment ?comment . }
            } UNION {
                { ?prop rdfs:domain ?source . } UNION { ?prop rdfs:range ?source . }
            } UNION {
                ?source ?anyProp ?anyValue .
                FILTER (?anyProp != rdf:type)
            }
        }

        # Get comprehensive labels as literal data for these top-level classes
        OPTIONAL { 
            ?source rdfs:label ?label_en . 
            FILTER (LANG(?label_en) = "en" || LANG(?label_en) = "")
        }
        OPTIONAL { 
            ?source rdfs:label ?label_any . 
            FILTER (!BOUND(?label_en))
        }
        OPTIONAL { 
            ?source rdfs:comment ?comment_en . 
            FILTER (LANG(?comment_en) = "en" || LANG(?comment_en) = "")
        }
        
        # Return the class labels as literal properties (marked with literal_type)
        BIND(rdfs:label AS ?link)
        BIND(COALESCE(
            ?label_en,
            ?label_any, 
            STRAFTER(STR(?source), "#"), 
            STRAFTER(STR(?source), "/"),
            STR(?source)
        ) as ?target)
        BIND(xsd:string AS ?literal_type)
    }
    ORDER BY STRLEN(STR(?source)) ?source
    LIMIT 30  # Increased limit to include more important top-level classes
  `;

  try {
    const data = await queryVirtuoso(fallbackQuery, virtuosoConfig.database);
    return data && data.nodes ? data : { nodes: [], edges: [], literals: {} };
  } catch (error) {
    console.error('Error in fallback query:', error);
    return { nodes: [], edges: [], literals: {} };
  }
};

/**
 * Fetches instances of a given class (and its subclasses) and their properties.
 * @param {string} classUri - The URI of the class.
 * @returns {Promise<object>} - Graph data { nodes, edges, literals } including instances and the class node.
 */
export const fetchClassInstances = async (classUri) => {
  console.log(`Fetching instances for class: ${classUri}`);
  if (!classUri || typeof classUri !== 'string' || !classUri.startsWith('http')) {
      console.error("fetchClassInstances called with invalid classUri:", classUri);
      return { nodes: [], edges: [], literals: {} };
  }
   // Query without GRAPH clause - queryVirtuoso will add it.
  const sparqlQuery = `
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?source ?link ?target (DATATYPE(?target) AS ?literal_type)
    WHERE {
        # Find instances of the class or any of its subclasses (transitively)
        ?source rdf:type ?type .
        ?type rdfs:subClassOf* <${classUri}> .
        FILTER (!isBlank(?source)) # Only named individuals
        FILTER (?type != owl:NamedIndividual) # Filter out NamedIndividual types

        {
            # Get only data properties (literals) of the instance, exclude object properties
            ?source ?link ?target .
            FILTER (isLiteral(?target)) # Only include literal values (data properties)
            # Filter out NamedIndividual type relations
            FILTER (!(?link = rdf:type && ?target = owl:NamedIndividual))
        } UNION {
            # Ensure the rdf:type link from instance to its *specific* type is included
            # but exclude NamedIndividual
            BIND(rdf:type AS ?link)
            BIND(?type AS ?target)
            BIND(xsd:anyURI AS ?literal_type)
        } UNION {
            # Add a link back to the queried class for context in visualization
            BIND(<http://example.org/viz#instanceOfQueriedClass> AS ?link) # Custom predicate
            BIND(<${classUri}> AS ?target)
            BIND(xsd:anyURI AS ?literal_type)
        }
    }
    LIMIT 500
  `;

  try {
    const data = await queryVirtuoso(sparqlQuery, virtuosoConfig.database);

    // Ensure the parent class node itself is included in the results
    if (data && data.nodes && !data.nodes.some(node => node.id === classUri)) {
        // Query for the label *without* GRAPH clause here, queryVirtuoso will add it
        const labelQuery = `
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE { 
                <${classUri}> rdfs:label ?label . 
                FILTER (LANG(?label) = "en" || LANG(?label) = "")
            } LIMIT 1`;
        let label = classUri.split(/[#/]/).pop();
        try {
            const labelData = await queryVirtuoso(labelQuery, virtuosoConfig.database);
            if (labelData && labelData.literals && labelData.literals[classUri] && labelData.literals[classUri]['http://www.w3.org/2000/01/rdf-schema#label']) {
                 label = labelData.literals[classUri]['http://www.w3.org/2000/01/rdf-schema#label'][0];
            }
        } catch (labelError) {
            console.warn(`Could not fetch label for class ${classUri}:`, labelError);
        }
        console.log(`Adding missing class node to instance data: ${classUri} with label: ${label}`);
        if (!data.nodes) data.nodes = []; // Ensure nodes array exists
        data.nodes.push({ id: classUri, label: label });
    }
    return data && data.nodes ? data : { nodes: [], edges: [], literals: {} };
  } catch (error) {
    console.error(`Error fetching instances for ${classUri}:`, error);
    return {
        nodes: [{ id: classUri, label: classUri.split(/[#/]/).pop() || classUri }],
        edges: [],
        literals: {}
    };
  }
};

/**
 * Fetches direct subclasses of a given class and their properties (like labels).
 * @param {string} classUri - The URI of the parent class.
 * @returns {Promise<object>} - Graph data { nodes, edges, literals } including subclasses and the parent class node.
 */
export const fetchSubclasses = async (classUri) => {
  console.log(`Fetching subclasses for: ${classUri}`);
   if (!classUri || typeof classUri !== 'string' || !classUri.startsWith('http')) {
      console.error("fetchSubclasses called with invalid classUri:", classUri);
      return { nodes: [], edges: [], literals: {} };
  }
  // Query without GRAPH clause - queryVirtuoso will add it.
  const sparqlQuery = `
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?source ?link ?target (DATATYPE(?target) AS ?literal_type)
    WHERE {
        {
            # Find direct subclasses
            ?source rdfs:subClassOf <${classUri}> .
            FILTER (!isBlank(?source) && ?source != <${classUri}>)
            
            # Filter out NamedIndividual and owl:Thing as they're not meaningful subclasses
            FILTER (?source != owl:NamedIndividual)
            FILTER (?source != owl:Thing)
        } UNION {
            # Find classes that are equivalent to intersections containing the target class
            ?source owl:equivalentClass ?equiv .
            ?equiv owl:intersectionOf ?list .
            ?list rdf:rest*/rdf:first <${classUri}> .
            FILTER (!isBlank(?source) && ?source != <${classUri}>)
            FILTER (?source != owl:NamedIndividual)
            FILTER (?source != owl:Thing)
        }

        {
            # Include the subclass relationship edge back to the parent
            BIND(rdfs:subClassOf AS ?link)
            BIND(<${classUri}> AS ?target)
            BIND(xsd:anyURI AS ?literal_type)
            # Don't include edges to owl:Thing
            FILTER (<${classUri}> != owl:Thing)
        } UNION {
            # Get labels for the subclass - prioritize English
            ?source rdfs:label ?target .
            FILTER (LANG(?target) = "en" || LANG(?target) = "")
            BIND(rdfs:label AS ?link)
            BIND(xsd:string AS ?literal_type)
        }
    }
    LIMIT 500
  `;

  try {
    const data = await queryVirtuoso(sparqlQuery, virtuosoConfig.database);

    // Ensure the parent class node exists in the result
    if (data && data.nodes && !data.nodes.some(node => node.id === classUri)) {
         // Query for the label *without* GRAPH clause here, queryVirtuoso will add it
        const labelQuery = `
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE { 
                <${classUri}> rdfs:label ?label . 
                FILTER (LANG(?label) = "en" || LANG(?label) = "")
            } LIMIT 1`;
        let label = classUri.split(/[#/]/).pop();
        try {
            const labelData = await queryVirtuoso(labelQuery, virtuosoConfig.database);
            if (labelData && labelData.literals && labelData.literals[classUri] && labelData.literals[classUri]['http://www.w3.org/2000/01/rdf-schema#label']) {
                 label = labelData.literals[classUri]['http://www.w3.org/2000/01/rdf-schema#label'][0];
            }
        } catch (labelError) {
            console.warn(`Could not fetch label for parent class ${classUri}:`, labelError);
        }
        console.log(`Adding missing parent class node to subclass data: ${classUri} with label: ${label}`);
         if (!data.nodes) data.nodes = []; // Ensure nodes array exists
        data.nodes.push({ id: classUri, label: label });
    }
    return data && data.nodes ? data : { nodes: [], edges: [], literals: {} };
  } catch (error) {
    console.error(`Error fetching subclasses for ${classUri}:`, error);
    return {
        nodes: [{ id: classUri, label: classUri.split(/[#/]/).pop() || classUri }],
        edges: [],
        literals: {}
    };
  }
};

/**
 * Fetches data properties of a specific instance.
 * @param {string} instanceUri - The URI of the instance.
 * @returns {Promise<object>} - Graph data { nodes, edges, literals } including data properties.
 */
export const fetchInstanceDataProperties = async (instanceUri) => {
  console.log(`Fetching data properties for instance: ${instanceUri}`);
  if (!instanceUri || typeof instanceUri !== 'string' || !instanceUri.startsWith('http')) {
      console.error("fetchInstanceDataProperties called with invalid instanceUri:", instanceUri);
      return { nodes: [], edges: [], literals: {} };
  }

  const sparqlQuery = `
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?source ?link ?target (DATATYPE(?target) AS ?literal_type)
    WHERE {
        # Get all data properties of the instance
        <${instanceUri}> ?link ?target .
        FILTER (isLiteral(?target)) # Only data properties (literal values)
        FILTER (?link != rdf:type) # Exclude type relationships
        
        # Exclude annotation properties
        FILTER (?link != rdfs:label)
        FILTER (?link != rdfs:comment)
        FILTER (?link != owl:versionInfo)
        FILTER (?link != <http://www.w3.org/2004/02/skos/core#prefLabel>)
        FILTER (?link != <http://www.w3.org/2004/02/skos/core#altLabel>)
        FILTER (?link != <http://www.w3.org/2004/02/skos/core#definition>)
        FILTER (?link != <http://purl.org/dc/elements/1.1/title>)
        FILTER (?link != <http://purl.org/dc/elements/1.1/description>)
        FILTER (?link != <http://purl.org/dc/elements/1.1/creator>)
        FILTER (?link != <http://purl.org/dc/elements/1.1/date>)
        FILTER (?link != <http://purl.org/dc/terms/title>)
        FILTER (?link != <http://purl.org/dc/terms/description>)
        FILTER (?link != <http://purl.org/dc/terms/creator>)
        FILTER (?link != <http://purl.org/dc/terms/created>)
        FILTER (?link != <http://purl.org/dc/terms/modified>)
        
        # Also exclude properties that are explicitly declared as annotation properties
        FILTER NOT EXISTS {
            ?link rdf:type owl:AnnotationProperty .
        }
        
        BIND(<${instanceUri}> AS ?source)
    }
    LIMIT 100
  `;

  try {
    const data = await queryVirtuoso(sparqlQuery, virtuosoConfig.database);
    
    // Ensure the instance node itself is included in the results
    if (data && data.nodes && !data.nodes.some(node => node.id === instanceUri)) {
        const labelQuery = `
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE { 
                <${instanceUri}> rdfs:label ?label . 
                FILTER (LANG(?label) = "en" || LANG(?label) = "")
            } LIMIT 1`;
        let label = instanceUri.split(/[#/]/).pop();
        try {
            const labelData = await queryVirtuoso(labelQuery, virtuosoConfig.database);
            if (labelData && labelData.literals && labelData.literals[instanceUri] && labelData.literals[instanceUri]['http://www.w3.org/2000/01/rdf-schema#label']) {
                 label = labelData.literals[instanceUri]['http://www.w3.org/2000/01/rdf-schema#label'][0];
            }
        } catch (labelError) {
            console.warn(`Could not fetch label for instance ${instanceUri}:`, labelError);
        }
        console.log(`Adding missing instance node to data properties result: ${instanceUri} with label: ${label}`);
        if (!data.nodes) data.nodes = [];
        data.nodes.push({ id: instanceUri, label: label });
    }
    return data && data.nodes ? data : { nodes: [], edges: [], literals: {} };
  } catch (error) {
    console.error(`Error fetching data properties for ${instanceUri}:`, error);
    return {
        nodes: [{ id: instanceUri, label: instanceUri.split(/[#/]/).pop() || instanceUri }],
        edges: [],
        literals: {}
    };
  }
};

/**
 * Fetches object assertions (object properties) of a specific instance.
 * @param {string} instanceUri - The URI of the instance.
 * @returns {Promise<object>} - Graph data { nodes, edges, literals } including object properties.
 */
export const fetchInstanceObjectAssertions = async (instanceUri) => {
  console.log(`[SPARQL] fetchInstanceObjectAssertions called with instanceUri: ${instanceUri}`);
  console.log(`[SPARQL] instanceUri type: ${typeof instanceUri}`);
  console.log(`[SPARQL] instanceUri starts with http: ${instanceUri && instanceUri.startsWith('http')}`);
  
  if (!instanceUri || typeof instanceUri !== 'string' || !instanceUri.startsWith('http')) {
      console.error("[SPARQL] fetchInstanceObjectAssertions called with invalid instanceUri:", instanceUri);
      return { nodes: [], edges: [], literals: {} };
  }

  console.log(`[SPARQL] Building SPARQL query for instance: ${instanceUri}`);

  const sparqlQuery = `
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT DISTINCT ?source ?link ?target (DATATYPE(?target) AS ?literal_type) ?targetLabel
    WHERE {
        {
            # Get all object properties of the instance
            <${instanceUri}> ?link ?target .
            FILTER (!isLiteral(?target)) # Only object properties (URI values)
            FILTER (?link != rdf:type) # Exclude type relationships
            FILTER (!isBlank(?target)) # Exclude blank nodes
            
            # Get label for the target object
            OPTIONAL { ?target rdfs:label ?targetLabel . }
            
            BIND(<${instanceUri}> AS ?source)
            BIND(xsd:anyURI AS ?literal_type)
        }
        UNION
        {
            # Include labels for the source node
            <${instanceUri}> rdfs:label ?target .
            FILTER (LANG(?target) = "en" || LANG(?target) = "")
            BIND(<${instanceUri}> AS ?source)
            BIND(rdfs:label AS ?link)
            BIND("http://www.w3.org/2001/XMLSchema#string" AS ?literal_type)
            BIND("" AS ?targetLabel)
        }
        UNION
        {
            # Include labels for target nodes
            <${instanceUri}> ?objectProperty ?target .
            FILTER (!isLiteral(?target) && ?objectProperty != rdf:type && !isBlank(?target))
            
            ?target rdfs:label ?targetLabel .
            FILTER (LANG(?targetLabel) = "en" || LANG(?targetLabel) = "")
            BIND(?target AS ?source)
            BIND(rdfs:label AS ?link)
            BIND(?targetLabel AS ?target)
            BIND("http://www.w3.org/2001/XMLSchema#string" AS ?literal_type)
        }
    }
    LIMIT 100
  `;

  console.log(`[SPARQL] Constructed SPARQL query:\n${sparqlQuery}`);

  try {
    console.log(`[SPARQL] About to call queryVirtuoso with database: ${virtuosoConfig.database}`);
    const data = await queryVirtuoso(sparqlQuery, virtuosoConfig.database);
    console.log(`[SPARQL] queryVirtuoso returned data:`, data);
    
    // Ensure the instance node itself is included in the results
    if (data && data.nodes && !data.nodes.some(node => node.id === instanceUri)) {
        const labelQuery = `
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?label WHERE { 
                <${instanceUri}> rdfs:label ?label . 
                FILTER (LANG(?label) = "en" || LANG(?label) = "")
            } LIMIT 1`;
        let label = instanceUri.split(/[#/]/).pop();
        try {
            const labelData = await queryVirtuoso(labelQuery, virtuosoConfig.database);
            if (labelData && labelData.literals && labelData.literals[instanceUri] && labelData.literals[instanceUri]['http://www.w3.org/2000/01/rdf-schema#label']) {
                 label = labelData.literals[instanceUri]['http://www.w3.org/2000/01/rdf-schema#label'][0];
            }
        } catch (labelError) {
            console.warn(`Could not fetch label for instance ${instanceUri}:`, labelError);
        }
        console.log(`Adding missing instance node to object assertions result: ${instanceUri} with label: ${label}`);
        if (!data.nodes) data.nodes = [];
        data.nodes.push({ id: instanceUri, label: label });
    }
    
    console.log(`[SPARQL] Final data being returned from fetchInstanceObjectAssertions:`, data);
    return data && data.nodes ? data : { nodes: [], edges: [], literals: {} };
  } catch (error) {
    console.error(`[SPARQL] Error fetching object assertions for ${instanceUri}:`, error);
    return {
        nodes: [{ id: instanceUri, label: instanceUri.split(/[#/]/).pop() || instanceUri }],
        edges: [],
        literals: {}
    };
  }
};

/**
 * Checks if a given node URI represents an instance (individual) rather than a class.
 * @param {string} nodeUri - The URI of the node to check.
 * @returns {Promise<boolean>} - True if the node is an instance, false if it's a class.
 */
export const isInstanceNode = async (nodeUri) => {
  console.log(`[isInstanceNode] Checking if node is an instance: ${nodeUri}`);
  if (!nodeUri || typeof nodeUri !== 'string' || !nodeUri.startsWith('http')) {
      console.error("[isInstanceNode] Called with invalid nodeUri:", nodeUri);
      return false;
  }

  try {
    // Comprehensive query to analyze the node's characteristics
    const analysisQuery = `
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
      PREFIX owl: <http://www.w3.org/2002/07/owl#>
      
      SELECT DISTINCT ?source ?link ?target WHERE {
        {
          # Get outgoing relations from this node
          <${nodeUri}> ?link ?target .
          BIND(<${nodeUri}> AS ?source)
        }
        UNION
        {
          # Get incoming rdf:type relations (if this node is used as a class)
          ?source rdf:type <${nodeUri}> .
          BIND(rdf:type AS ?link)
          BIND(<${nodeUri}> AS ?target)
        }
        UNION
        {
          # Get incoming rdfs:subClassOf relations (if this node is used as a superclass)
          ?source rdfs:subClassOf <${nodeUri}> .
          BIND(rdfs:subClassOf AS ?link)
          BIND(<${nodeUri}> AS ?target)
        }
      }
      LIMIT 50
    `;

    console.log(`[isInstanceNode] Running comprehensive analysis query for ${nodeUri}`);
    const data = await queryVirtuoso(analysisQuery, virtuosoConfig.database);
    console.log(`[isInstanceNode] Analysis response for ${nodeUri}:`, {
      edges: data?.edges?.length || 0,
      literals: Object.keys(data?.literals || {}).length
    });
    
    // Analyze the results
    let hasTypeRelations = false;
    let hasSubClassRelations = false;
    let hasDataProperties = false;
    let usedAsClass = false;
    let usedAsSuperClass = false;
    let hasObjectProperties = false;
    
    // Check edges for various indicators
    if (data.edges && data.edges.length > 0) {
      console.log(`[isInstanceNode] Found ${data.edges.length} edges for ${nodeUri}`);
      
      data.edges.forEach(edge => {
        console.log(`[isInstanceNode] Edge: ${edge.source} -> ${edge.link} -> ${edge.target}`);
        
        // Check outgoing relations from this node
        if (edge.source === nodeUri) {
          if (edge.link === 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' || edge.link.includes('#type')) {
            if (edge.target !== 'http://www.w3.org/2002/07/owl#NamedIndividual') {
              hasTypeRelations = true;
              console.log(`[isInstanceNode] Found type relation: ${edge.target}`);
            }
          } else if (edge.link === 'http://www.w3.org/2000/01/rdf-schema#subClassOf' || edge.link.includes('#subClassOf')) {
            hasSubClassRelations = true;
            console.log(`[isInstanceNode] Found subClassOf relation: ${edge.target}`);
          } else {
            // Check if it's a data property (has literal value) or object property
            if (edge.target && typeof edge.target === 'string' && !edge.target.startsWith('http')) {
              hasDataProperties = true;
              console.log(`[isInstanceNode] Found data property: ${edge.link} -> ${edge.target}`);
            } else if (edge.target && edge.target.startsWith('http')) {
              hasObjectProperties = true;
              console.log(`[isInstanceNode] Found object property: ${edge.link} -> ${edge.target}`);
            }
          }
        }
        
        // Check if this node is used as a class (incoming rdf:type)
        if (edge.target === nodeUri && (edge.link === 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type' || edge.link.includes('#type'))) {
          usedAsClass = true;
          console.log(`[isInstanceNode] Node is used as a class by: ${edge.source}`);
        }
        
        // Check if this node is used as a superclass (incoming rdfs:subClassOf)
        if (edge.target === nodeUri && (edge.link === 'http://www.w3.org/2000/01/rdf-schema#subClassOf' || edge.link.includes('#subClassOf'))) {
          usedAsSuperClass = true;
          console.log(`[isInstanceNode] Node is used as a superclass by: ${edge.source}`);
        }
      });
    }
    
    // Check literals for additional data properties
    if (data.literals && Object.keys(data.literals).length > 0) {
      console.log(`[isInstanceNode] Found literals for ${nodeUri}:`, Object.keys(data.literals));
      Object.keys(data.literals).forEach(uri => {
        if (uri === nodeUri) {
          const props = data.literals[uri];
          console.log(`[isInstanceNode] URI ${uri} has literal properties:`, Object.keys(props));
          
          // Check for type and subClassOf in literals
          if (props['http://www.w3.org/1999/02/22-rdf-syntax-ns#type']) {
            hasTypeRelations = true;
            console.log(`[isInstanceNode] Found type in literals:`, props['http://www.w3.org/1999/02/22-rdf-syntax-ns#type']);
          }
          if (props['http://www.w3.org/2000/01/rdf-schema#subClassOf']) {
            hasSubClassRelations = true;
            console.log(`[isInstanceNode] Found subClassOf in literals:`, props['http://www.w3.org/2000/01/rdf-schema#subClassOf']);
          }
          
          // Any other literal properties indicate data properties (common in instances)
          const otherProps = Object.keys(props).filter(prop => 
            !prop.includes('#type') && !prop.includes('#subClassOf') && !prop.includes('#label')
          );
          if (otherProps.length > 0) {
            hasDataProperties = true;
            console.log(`[isInstanceNode] Found data properties:`, otherProps);
          }
        }
      });
    }
    
    console.log(`[isInstanceNode] Final analysis for ${nodeUri}:`, {
      hasTypeRelations,
      hasSubClassRelations,
      hasDataProperties,
      hasObjectProperties,
      usedAsClass,
      usedAsSuperClass
    });
    
    // Enhanced classification logic
    // Strong indicators that it's a CLASS:
    if (hasSubClassRelations || usedAsClass || usedAsSuperClass) {
      console.log(`[isInstanceNode] ${nodeUri} is a CLASS (has class indicators)`);
      return false;
    }
    
    // Strong indicators that it's an INSTANCE:
    if (hasTypeRelations && !hasSubClassRelations) {
      console.log(`[isInstanceNode] ${nodeUri} is an INSTANCE (has rdf:type, no rdfs:subClassOf)`);
      return true;
    }
    
    // Additional indicators for instances:
    if (hasDataProperties && !usedAsClass && !usedAsSuperClass) {
      console.log(`[isInstanceNode] ${nodeUri} is likely an INSTANCE (has data properties, not used as class)`);
      return true;
    }
    
    // Heuristic based on URI patterns (common naming conventions)
    const uriParts = nodeUri.split(/[#/]/);
    const lastPart = uriParts[uriParts.length - 1];
    
    // Instance naming patterns (lowercase, contains numbers, specific suffixes)
    const instancePatterns = [
      /^[a-z]/, // starts with lowercase
      /\d/, // contains numbers
      /_(instance|ind|item|object|\d+)$/i, // ends with instance indicators
      /\d{4}-\d{2}-\d{2}/, // date patterns
      /uuid|guid/i, // unique identifiers
    ];
    
    // Class naming patterns (PascalCase, abstract terms)
    const classPatterns = [
      /^[A-Z][a-z]+[A-Z]/, // PascalCase
      /^[A-Z][a-z]*$/, // Simple capitalized word
      /(Class|Type|Category|Concept)$/i, // explicit class suffixes
    ];
    
    const instanceScore = instancePatterns.filter(pattern => pattern.test(lastPart)).length;
    const classScore = classPatterns.filter(pattern => pattern.test(lastPart)).length;
    
    console.log(`[isInstanceNode] URI pattern analysis for ${lastPart}: instanceScore=${instanceScore}, classScore=${classScore}`);
    
    if (instanceScore > classScore && instanceScore > 0) {
      console.log(`[isInstanceNode] ${nodeUri} is likely an INSTANCE (based on URI patterns)`);
      return true;
    }
    
    if (classScore > instanceScore && classScore > 0) {
      console.log(`[isInstanceNode] ${nodeUri} is likely a CLASS (based on URI patterns)`);
      return false;
    }
    
    // If no strong indicators, default to class (more conservative approach)
    console.log(`[isInstanceNode] ${nodeUri} - no clear classification, defaulting to CLASS`);
    return false;
    
  } catch (error) {
    console.error(`[isInstanceNode] Error checking if ${nodeUri} is an instance:`, error);
    
    // Fallback: use basic URI pattern analysis
    try {
      const uriParts = nodeUri.split(/[#/]/);
      const lastPart = uriParts[uriParts.length - 1];
      
      // Simple heuristic: if it contains numbers or starts with lowercase, likely an instance
      if (/\d/.test(lastPart) || /^[a-z]/.test(lastPart)) {
        console.log(`[isInstanceNode] Fallback: ${nodeUri} treated as INSTANCE (URI pattern)`);
        return true;
      }
    } catch (fallbackError) {
      console.error(`[isInstanceNode] Fallback analysis failed:`, fallbackError);
    }
    
    return false;
  }
};

/**
 * Fetches annotation properties for a given node URI.
 * This includes labels, comments, descriptions, and other common annotation properties.
 * @param {string} nodeUri - The URI of the node to fetch annotations for.
 * @returns {Promise<object>} - A promise that resolves to the annotation data.
 */
export const fetchNodeAnnotations = async (nodeUri) => {
  console.log(`[fetchNodeAnnotations] Fetching annotations for: ${nodeUri}`);
  
  if (!nodeUri || typeof nodeUri !== 'string' || !nodeUri.startsWith('http')) {
    console.error("[fetchNodeAnnotations] Called with invalid nodeUri:", nodeUri);
    return { annotations: [] };
  }

  try {
    // Consulta SIMPLE y PRECISA: Solo propiedades declaradas como owl:AnnotationProperty
    const annotationQuery = `
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX owl: <http://www.w3.org/2002/07/owl#>
      
      SELECT DISTINCT ?source ?link ?target (DATATYPE(?target) AS ?literal_type) (LANG(?target) AS ?language)
      WHERE {
        # El individuo de interés tiene una propiedad y un valor
        <${nodeUri}> ?link ?target .
        
        # Subconsulta: solo propiedades que son AnnotationProperty
        {
          SELECT ?link
          WHERE {
            ?link rdf:type owl:AnnotationProperty .
          }
        }
        
        BIND(<${nodeUri}> AS ?source)
      }
      ORDER BY ?link ?target
      LIMIT 100
    `;

    console.log(`[fetchNodeAnnotations] Running annotation query for ${nodeUri}`);
    
    // Use queryVirtuoso like other working functions
    const data = await queryVirtuoso(annotationQuery, virtuosoConfig.database);
    console.log(`[fetchNodeAnnotations] queryVirtuoso returned data for ${nodeUri}:`, data);
    
    // DEBUG: Log detailed structure of returned data
    console.log(`[fetchNodeAnnotations] Data structure analysis:`);
    console.log(`  - nodes length: ${data.nodes ? data.nodes.length : 'undefined'}`);
    console.log(`  - edges length: ${data.edges ? data.edges.length : 'undefined'}`);
    console.log(`  - literals keys: ${data.literals ? Object.keys(data.literals) : 'undefined'}`);
    if (data.literals && data.literals[nodeUri]) {
      console.log(`  - literals for nodeUri: ${Object.keys(data.literals[nodeUri])}`);
    }
    if (data.nodes && data.nodes.length > 0) {
      console.log(`  - sample node:`, data.nodes[0]);
    }
    if (data.edges && data.edges.length > 0) {
      console.log(`  - sample edge:`, data.edges[0]);
    }
    
    // Process the data returned by queryVirtuoso
    const annotations = [];
    
    // Procesar tanto literals como edges (URIs)
    if (data) {
      // Procesar literales (valores textuales)
      if (data.literals && data.literals[nodeUri]) {
        const nodeLiterals = data.literals[nodeUri];
        Object.keys(nodeLiterals).forEach(property => {
          const propertyName = getPropertyDisplayName(property);
          const values = nodeLiterals[property];
          
          values.forEach(value => {
            annotations.push({
              property: property,
              propertyName: propertyName,
              value: value,
              valueType: 'literal',
              language: null // queryVirtuoso doesn't preserve language info in this format
            });
          });
        });
      }
      
      // Procesar edges (anotaciones que apuntan a URIs)
      if (data.edges && data.edges.length > 0) {
        data.edges.forEach(edge => {
          // Solo procesar edges que tienen como source nuestro nodeUri
          if (edge.source === nodeUri) {
            const propertyName = getPropertyDisplayName(edge.link);
            
            // Para URIs, mostrar solo la parte final como valor legible
            const displayValue = edge.target.split(/[#/]/).pop() || edge.target;
            
            annotations.push({
              property: edge.link,
              propertyName: propertyName,
              value: displayValue,
              fullValue: edge.target, // Mantener URI completa para referencia
              valueType: 'uri',
              language: null
            });
          }
        });
      }
      
      // Procesar links (si existen en formato diferente)
      if (data.links && data.links[nodeUri]) {
        const nodeLinks = data.links[nodeUri];
        Object.keys(nodeLinks).forEach(property => {
          const propertyName = getPropertyDisplayName(property);
          const targets = nodeLinks[property];
          
          targets.forEach(target => {
            // Para URIs, mostrar solo la parte final como valor legible
            const displayValue = target.split(/[#/]/).pop() || target;
            annotations.push({
              property: property,
              propertyName: propertyName,
              value: displayValue,
              fullValue: target, // Mantener URI completa para referencia
              valueType: 'uri',
              language: null
            });
          });
        });
      }
      
      // Si no hay datos estructurados, intentar procesar directamente desde los nodos
      if (annotations.length === 0 && data.nodes) {
        data.nodes.forEach(node => {
          if (node.id === nodeUri && node.properties) {
            Object.keys(node.properties).forEach(property => {
              const propertyName = getPropertyDisplayName(property);
              const value = node.properties[property];
              
              annotations.push({
                property: property,
                propertyName: propertyName,
                value: value,
                valueType: typeof value === 'string' && value.startsWith('http') ? 'uri' : 'literal',
                language: null
              });
            });
          }
        });
      }
    }
    
    console.log(`[fetchNodeAnnotations] Processed ${annotations.length} annotations for ${nodeUri}`);
    
    return { annotations };
    
  } catch (error) {
    console.error(`[fetchNodeAnnotations] Error fetching annotations for ${nodeUri}:`, error);
    return { annotations: [] };
  }
};

/**
 * Helper function to get a human-readable display name for annotation properties
 * @param {string} propertyUri - The full URI of the property
 * @returns {string} - A human-readable name for the property
 */
function getPropertyDisplayName(propertyUri) {
  const propertyMap = {
    // RDF/RDFS/OWL
    'http://www.w3.org/2000/01/rdf-schema#label': 'Label',
    'http://www.w3.org/2000/01/rdf-schema#comment': 'Comment',
    'http://www.w3.org/2000/01/rdf-schema#seeAlso': 'See Also',
    'http://www.w3.org/2000/01/rdf-schema#isDefinedBy': 'Defined By',
    'http://www.w3.org/2002/07/owl#versionInfo': 'Version Info',
    'http://www.w3.org/2002/07/owl#priorVersion': 'Prior Version',
    'http://www.w3.org/2002/07/owl#backwardCompatibleWith': 'Backward Compatible With',
    'http://www.w3.org/2002/07/owl#incompatibleWith': 'Incompatible With',
    
    // Dublin Core Elements
    'http://purl.org/dc/elements/1.1/title': 'Title',
    'http://purl.org/dc/elements/1.1/description': 'Description',
    'http://purl.org/dc/elements/1.1/creator': 'Creator',
    'http://purl.org/dc/elements/1.1/contributor': 'Contributor',
    'http://purl.org/dc/elements/1.1/date': 'Date',
    'http://purl.org/dc/elements/1.1/subject': 'Subject',
    
    // Dublin Core Terms
    'http://purl.org/dc/terms/description': 'Description',
    'http://purl.org/dc/terms/title': 'Title',
    'http://purl.org/dc/terms/creator': 'Creator',
    'http://purl.org/dc/terms/created': 'Created',
    'http://purl.org/dc/terms/modified': 'Modified',
    'http://purl.org/dc/terms/format': 'Format',
    
    // SKOS
    'http://www.w3.org/2004/02/skos/core#prefLabel': 'Preferred Label',
    'http://www.w3.org/2004/02/skos/core#altLabel': 'Alternative Label',
    'http://www.w3.org/2004/02/skos/core#definition': 'Definition',
    'http://www.w3.org/2004/02/skos/core#note': 'Note',
    
    // DCAT (Data Catalog Vocabulary)
    'http://www.w3.org/ns/dcat#accessService': 'Access Service',
    'http://www.w3.org/ns/dcat#endpointDescription': 'Endpoint Description',
    'http://www.w3.org/ns/dcat#endpointURL': 'Endpoint URL',
    'http://www.w3.org/ns/dcat#endpointUrl': 'Endpoint URL',
    'http://www.w3.org/ns/dcat#mediaType': 'Media Type',
    'http://www.w3.org/ns/dcat#service': 'Service',
    'http://www.w3.org/ns/dcat#version': 'Version',
    
    // IDS Core
    'https://w3id.org/idsa/core/description': 'Description',
    'https://w3id.org/idsa/core/title': 'Title',
    'https://w3id.org/idsa/core/keyword': 'Keyword',
    'https://w3id.org/idsa/core/language': 'Language',
    'https://w3id.org/idsa/core/license': 'License',
    'https://w3id.org/idsa/core/version': 'Version',
    
    // EDC
    'https://w3id.org/edc/v0.0.1/ns/id': 'ID',
    'https://w3id.org/edc/v0.0.1/ns/originator': 'Originator',
    
    // FOAF
    'http://xmlns.com/foaf/0.1/name': 'Name',
    
    // DSPACE
    'https://w3id.org/dspace/v0.8/participantId': 'Participant ID',
    
    // ODRL
    'http://www.w3.org/ns/odrl/2/action': 'Action',
    'http://www.w3.org/ns/odrl/2/permission': 'Permission',
    'http://www.w3.org/ns/odrl/2/constraint': 'Constraint',
    'http://www.w3.org/ns/odrl/2/leftOperand': 'Left Operand',
    'http://www.w3.org/ns/odrl/2/operator': 'Operator',
    'http://www.w3.org/ns/odrl/2/rightOperand': 'Right Operand',
    'http://www.w3.org/ns/odrl/2/and': 'And',
    'http://www.w3.org/ns/odrl/2/or': 'Or'
  };
  
  // Si la propiedad está en el mapa, usar el nombre legible
  if (propertyMap[propertyUri]) {
    return propertyMap[propertyUri];
  }
  
  // Para propiedades no conocidas, generar un nombre legible automáticamente
  // Extraer la parte final de la URI
  let localName = propertyUri.split(/[#/]/).pop() || propertyUri;
  
  // Convertir camelCase a título con espacios
  localName = localName.replace(/([A-Z])/g, ' $1') // Añadir espacio antes de mayúsculas
                       .replace(/^./, str => str.toUpperCase()) // Primera letra mayúscula
                       .trim(); // Limpiar espacios
  
  // Si el resultado está vacío, usar la URI completa
  return localName || propertyUri;
}

// Export the config in case it's needed elsewhere
export { virtuosoConfig };