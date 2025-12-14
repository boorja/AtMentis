import io
import pandas as pd

from collections import defaultdict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rdflib import Graph
from SPARQLWrapper import DIGEST, CSV, SPARQLWrapper, JSON

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SPARQLFileQueryRequest(BaseModel):
    file_path: str
    sparql_query: str


class VirtuosoRequest(BaseModel):
    virtuoso_endpoint: str
    virtuoso_database: str
    virtuoso_username: str
    virtuoso_password: str
    query: str


@app.post("/query_rdf")
def query_rdf(request: SPARQLFileQueryRequest):
    g = Graph()
    try:
        g.parse(request.file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse RDF file: {e}")

    try:
        result = g.query(request.sparql_query)
        result = result.serialize(format="csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute query: {e}")

    out_dict = _process_csv_bytes_to_cosmos_dict(result)

    return JSONResponse(content=out_dict)


@app.post("/query_virtuoso")
def query_virtuoso(request: VirtuosoRequest):
    try:
        sparql = SPARQLWrapper(request.virtuoso_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(request.virtuoso_username, request.virtuoso_password)
        sparql.addDefaultGraph(request.virtuoso_database)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to setup Virtuoso: {e}")

    try:
        sparql.setQuery(query=request.query)
        sparql.setReturnFormat(CSV)
        result = sparql.queryAndConvert()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute query: {e}")

    out_dict = _process_csv_bytes_to_cosmos_dict(result)
    return JSONResponse(content=out_dict)


@app.get("/available-graphs")
def get_available_graphs():
    """Get list of available graphs from Virtuoso"""
    try:
        # Virtuoso configuration - you may want to make this configurable
        virtuoso_endpoint = "http://192.168.216.102:8890/sparql"
        virtuoso_username = "dba"
        virtuoso_password = "password"
        
        sparql = SPARQLWrapper(virtuoso_endpoint)
        sparql.setHTTPAuth(DIGEST)
        sparql.setCredentials(virtuoso_username, virtuoso_password)
        
        # Query to get all named graphs
        query = """
        SELECT DISTINCT ?g
        WHERE {
            GRAPH ?g {
                ?s ?p ?o .
            }
        }
        ORDER BY ?g
        """
        
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        results = sparql.queryAndConvert()
        
        graphs = []
        
        for result in results.get("results", {}).get("bindings", []):
            graph_uri = result["g"]["value"]
            
            # Skip system/internal graphs
            if any(skip in graph_uri.lower() for skip in ['system', 'virtrdf', 'http://www.openlinksw.com']):
                continue
                
            # Try to get some basic info about the graph
            info_query = f"""
            SELECT (COUNT(*) as ?triples)
            WHERE {{
                GRAPH <{graph_uri}> {{
                    ?s ?p ?o .
                }}
            }}
            """
            
            try:
                sparql.setQuery(info_query)
                sparql.setReturnFormat(JSON)
                info_results = sparql.queryAndConvert()
                triple_count = info_results.get("results", {}).get("bindings", [{}])[0].get("triples", {}).get("value", "0")
                
                # Extract a readable name from the URI
                name = graph_uri.split('/')[-1] or graph_uri.split('/')[-2] or graph_uri
                if '#' in name:
                    name = name.split('#')[-1]
                
                # Handle empty or whitespace-only names
                if not name or not name.strip():
                    name = None
                else:
                    name = name.replace('_', ' ').replace('-', ' ').title()
                
                graphs.append({
                    "name": name,
                    "uri": graph_uri,
                    "description": f"Grafo con {triple_count} tripletas",
                    "triples": int(triple_count) if triple_count.isdigit() else 0
                })
            except Exception as info_error:
                print(f"Error getting info for graph {graph_uri}: {info_error}")
                
                # Extract name with null handling
                name = graph_uri.split('/')[-1] or graph_uri.split('/')[-2]
                if name and '#' in name:
                    name = name.split('#')[-1]
                if not name or not name.strip():
                    name = None
                
                graphs.append({
                    "name": name,
                    "uri": graph_uri,
                    "description": "Información no disponible",
                    "triples": 0
                })
        
        return JSONResponse(content={
            "graphs": graphs,
            "status": "success"
        })
        
    except Exception as e:
        print(f"Error fetching available graphs: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error fetching available graphs: {str(e)}", "status": "error"}
        )


RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
DCT_TITLE = "http://purl.org/dc/terms/title"


def _process_csv_bytes_to_cosmos_dict(csv_bytes: bytes):
    try:
        query_result_df = pd.read_csv(io.StringIO(csv_bytes.decode("utf-8")))

        # Handle cases where query returns fewer columns
        if len(query_result_df) == 0:
            # Empty result set
            print("Query returned empty result set")
            return {"nodes": [], "edges": [], "literals": {}}

        # Check number of columns and handle accordingly
        num_columns = len(query_result_df.columns)
        print(f"Query returned {num_columns} columns: {query_result_df.columns.tolist()}")
        
        if num_columns == 1:
            # Single column result (like label queries) - return empty graph data
            print("Single column result detected - returning empty graph data")
            return {"nodes": [], "edges": [], "literals": {}}
        elif num_columns == 2:
            # Two column result - also not sufficient for graph structure
            print("Two column result detected - returning empty graph data")
            return {"nodes": [], "edges": [], "literals": {}}
        elif num_columns < 3:
            # Any other insufficient column count
            print(f"Insufficient columns ({num_columns}) for graph structure - returning empty data")
            return {"nodes": [], "edges": [], "literals": {}}

        # Only proceed if we have 3 or more columns for proper graph structure
        # Ensure consistent column naming even if query returns more
        if num_columns >= 4:
            query_result_df = query_result_df.iloc[:, :4]
            query_result_df.columns = ["source", "link", "target", "literal_type"]
        else:  # exactly 3 columns
            query_result_df.columns = ["source", "link", "target"]
            # Add empty literal_type column
            query_result_df["literal_type"] = None

        print(f"Processing {len(query_result_df)} rows with columns: {query_result_df.columns.tolist()}")

        query_result_df = query_result_df.replace({float("nan"): None})

        nodes_data = defaultdict(lambda: {"id": None})
        edges_data = []
        literals_data = defaultdict(lambda: defaultdict(list))
        processed_nodes = set()

        for _, row in query_result_df.iterrows():
            source_uri = row["source"]
            link_uri = row["link"]
            target_uri = row["target"]
            literal_type = row.get(
                "literal_type"
            )  # Use .get() in case column doesn't exist

            # --- Validate URIs ---
            if source_uri is None or str(source_uri).lower() == "null":
                continue

            if link_uri is None or str(link_uri).lower() == "null":
                if not literal_type:
                    continue

            if not literal_type and (
                target_uri is None or str(target_uri).lower() == "null"
            ):
                continue

            # --- Ensure source node exists ---
            if source_uri not in processed_nodes:
                nodes_data[source_uri]["id"] = source_uri
                processed_nodes.add(source_uri)

            # Determine if this should be treated as a literal
            is_literal = (
                (
                    literal_type
                    and literal_type.strip()
                    and literal_type.lower() != "null"
                )
                or (link_uri == RDFS_LABEL)
                or (
                    # If target doesn't start with http:// it's likely a literal
                    target_uri
                    and not str(target_uri).startswith("http")
                )
            )

            if is_literal:
                # --- Store literal ---
                if target_uri is None:
                    continue
                literals_data[source_uri][link_uri].append(target_uri)
            else:
                # --- Process edge and ensure target node exists ---
                if target_uri not in processed_nodes:
                    nodes_data[target_uri]["id"] = target_uri
                    processed_nodes.add(target_uri)
                edges_data.append(
                    {"source": source_uri, "target": target_uri, "link": link_uri}
                )

        # Convert defaultdict nodes_data to a list of node objects
        final_nodes = []
        for node_id, node_data in nodes_data.items():
            # Start with the URI as label
            label = (
                node_id.split("/")[-1].split("#")[-1]
                if "/" in node_id or "#" in node_id
                else node_id
            )

            # Check if there's an English rdfs:label in literals for this node
            if node_id in literals_data and RDFS_LABEL in literals_data[node_id]:
                labels = literals_data[node_id][RDFS_LABEL]
                if labels:
                    # Use the first available label (already processed to prioritize English)
                    label = labels[0]

            final_nodes.append({"id": node_id, "label": label})

        out_dict = {
            "nodes": final_nodes,
            "edges": edges_data,
            "literals": dict(literals_data),
        }
        
        print(f"Successfully processed: {len(final_nodes)} nodes, {len(edges_data)} edges")

    except Exception as e:
        print(f"Error processing CSV: {e}")
        print(f"CSV Data causing error:\n{csv_bytes.decode('utf-8', errors='ignore')[:500]}...")
        print("Returning empty data due to processing error")
        return {"nodes": [], "edges": [], "literals": {}}

    return out_dict


if __name__ == "__main__":
    import uvicorn

    print("Starting Uvicorn server...")  # Add a print statement for confirmation
    uvicorn.run(app, host="0.0.0.0", port=32323)
