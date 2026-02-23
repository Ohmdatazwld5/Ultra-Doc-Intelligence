"""
GraphRAG Engine for Ultra Doc-Intelligence.
Builds a knowledge graph from documents for relationship-based queries.
Uses NetworkX for in-memory graph storage (production would use Neo4j).
"""

import json
import logging
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

import networkx as nx
from openai import OpenAI

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities in logistics documents."""
    # Transactional entities
    SHIPMENT = "shipment"
    SHIPPER = "shipper"
    CONSIGNEE = "consignee"
    CARRIER = "carrier"
    LOCATION = "location"
    EQUIPMENT = "equipment"
    COMMODITY = "commodity"
    RATE = "rate"
    DATE = "date"
    DOCUMENT = "document"
    # Product/Informational entities
    COMPANY = "company"
    PRODUCT = "product"
    FEATURE = "feature"
    PROCESS = "process"
    SERVICE = "service"
    TECHNOLOGY = "technology"
    BENEFIT = "benefit"
    TERM = "term"


class RelationType(Enum):
    """Types of relationships between entities."""
    # Transactional relationships
    SHIPS_FROM = "ships_from"
    SHIPS_TO = "ships_to"
    CARRIED_BY = "carried_by"
    CONTAINS = "contains"
    USES_EQUIPMENT = "uses_equipment"
    HAS_RATE = "has_rate"
    PICKUP_DATE = "pickup_date"
    DELIVERY_DATE = "delivery_date"
    MENTIONED_IN = "mentioned_in"
    LOCATED_AT = "located_at"
    # Product/Informational relationships
    OFFERS = "offers"
    HAS_FEATURE = "has_feature"
    PROVIDES = "provides"
    ENABLES = "enables"
    INTEGRATES_WITH = "integrates_with"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    SUPPORTS = "supports"


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    source_doc_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type.value,
            "properties": self.properties,
            "source_doc_id": self.source_doc_id
        }


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.relation_type.value,
            "properties": self.properties
        }


@dataclass
class GraphQueryResult:
    """Result from a graph query."""
    answer: str
    entities: List[Entity]
    relationships: List[Relationship]
    subgraph: Dict[str, Any]
    confidence: float
    reasoning_path: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "subgraph": self.subgraph,
            "confidence": self.confidence,
            "reasoning_path": self.reasoning_path
        }


class GraphRAGEngine:
    """
    Knowledge Graph-based RAG engine.
    Extracts entities and relationships from documents,
    then answers queries using graph traversal + LLM reasoning.
    """
    
    def __init__(
        self,
        api_key: str,
        llm_model: str = "grok-4-1-fast",
        base_url: str = "https://api.x.ai/v1"
    ):
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_model = llm_model
        
        # In-memory knowledge graph using NetworkX
        self.graph = nx.MultiDiGraph()
        
        # Entity registry for deduplication
        self.entity_registry: Dict[str, Entity] = {}
        
        # Document tracking
        self.indexed_documents: Set[str] = set()
    
    def _generate_entity_id(self, entity_type: EntityType, name: str) -> str:
        """Generate unique entity ID based on type and normalized name."""
        normalized = name.lower().strip()
        hash_input = f"{entity_type.value}:{normalized}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def extract_entities_and_relationships(
        self,
        document_text: str,
        document_id: str
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from document using LLM.
        """
        system_prompt = """You are a logistics document analyzer that extracts entities and relationships.

## Entity Types (for transactional documents):
- SHIPMENT: Load/shipment identifiers (PRO, BOL, Reference numbers)
- SHIPPER: Company shipping the goods
- CONSIGNEE: Company receiving the goods
- CARRIER: Trucking/transport company
- LOCATION: Addresses, cities, facilities
- EQUIPMENT: Trailer types (Dry Van, Flatbed, Reefer)
- COMMODITY: Goods being shipped
- RATE: Pricing information
- DATE: Dates and times

## Entity Types (for product/informational documents):
- COMPANY: Companies, organizations, vendors mentioned
- PRODUCT: Software products, platforms, solutions
- FEATURE: Specific features or capabilities
- PROCESS: Business processes (billing, claims, payment)
- SERVICE: Services offered
- TECHNOLOGY: Technologies, integrations, systems
- BENEFIT: Benefits, advantages, value propositions
- TERM: Industry terms, concepts

## Relationship Types (transactional):
- SHIPS_FROM: Shipment → Shipper location
- SHIPS_TO: Shipment → Consignee location
- CARRIED_BY: Shipment → Carrier
- CONTAINS: Shipment → Commodity
- USES_EQUIPMENT: Shipment → Equipment
- HAS_RATE: Shipment → Rate
- PICKUP_DATE: Shipment → Date
- DELIVERY_DATE: Shipment → Date
- LOCATED_AT: Shipper/Consignee → Location

## Relationship Types (informational):
- OFFERS: Company → Product/Service
- HAS_FEATURE: Product → Feature
- PROVIDES: Product → Benefit
- ENABLES: Feature → Process
- INTEGRATES_WITH: Product → Technology
- PART_OF: Feature → Product
- RELATED_TO: Entity → Entity
- SUPPORTS: Product → Process

## Output Format:
Return JSON with:
{
  "entities": [
    {"name": "...", "type": "SHIPMENT|COMPANY|PRODUCT|FEATURE|...", "properties": {...}},
    ...
  ],
  "relationships": [
    {"source": "entity_name", "target": "entity_name", "type": "OFFERS|HAS_FEATURE|...", "properties": {...}},
    ...
  ]
}

Extract ALL entities and relationships found in the document. If it's a product datasheet, focus on companies, products, features, and processes."""

        user_prompt = f"""Extract entities and relationships from this logistics document:

---
{document_text[:8000]}
---

Return ONLY valid JSON."""

        try:
            logger.info(f"Extracting entities from document {document_id}, text length: {len(document_text)}")
            
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            raw_response = response.choices[0].message.content
            logger.info(f"LLM raw response (first 500 chars): {raw_response[:500]}")
            
            result_text = self._clean_json_response(raw_response)
            logger.info(f"Cleaned JSON (first 500 chars): {result_text[:500]}")
            
            data = json.loads(result_text)
            logger.info(f"Parsed entities count: {len(data.get('entities', []))}, relationships: {len(data.get('relationships', []))}")
            
            entities = []
            relationships = []
            
            # Process entities
            for e in data.get("entities", []):
                try:
                    entity_type = EntityType[e["type"].upper()]
                    entity_id = self._generate_entity_id(entity_type, e["name"])
                    
                    entity = Entity(
                        id=entity_id,
                        name=e["name"],
                        entity_type=entity_type,
                        properties=e.get("properties", {}),
                        source_doc_id=document_id
                    )
                    entities.append(entity)
                except (KeyError, ValueError) as ex:
                    logger.warning(f"Skipping invalid entity: {e}, error: {ex}")
            
            # Process relationships
            for r in data.get("relationships", []):
                try:
                    rel_type = RelationType[r["type"].upper()]
                    
                    # Find source and target entity IDs
                    source_name = r["source"]
                    target_name = r["target"]
                    
                    # Look up entity IDs
                    source_entity = next((e for e in entities if e.name == source_name), None)
                    target_entity = next((e for e in entities if e.name == target_name), None)
                    
                    if source_entity and target_entity:
                        relationship = Relationship(
                            source_id=source_entity.id,
                            target_id=target_entity.id,
                            relation_type=rel_type,
                            properties=r.get("properties", {})
                        )
                        relationships.append(relationship)
                except (KeyError, ValueError) as ex:
                    logger.warning(f"Skipping invalid relationship: {r}, error: {ex}")
            
            return entities, relationships
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Raw text: {result_text[:500] if 'result_text' in locals() else 'N/A'}")
            return [], []
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}", exc_info=True)
            return [], []
    
    def _clean_json_response(self, text: str) -> str:
        """Clean markdown from JSON response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        return text
    
    def index_document(
        self,
        document_text: str,
        document_id: str,
        document_name: str = ""
    ) -> Dict[str, Any]:
        """
        Index a document into the knowledge graph.
        Extracts entities and relationships, adds to graph.
        """
        if document_id in self.indexed_documents:
            return {
                "status": "already_indexed",
                "document_id": document_id,
                "entities_count": 0,
                "relationships_count": 0
            }
        
        # Add document node
        doc_entity = Entity(
            id=f"doc_{document_id[:8]}",
            name=document_name or document_id,
            entity_type=EntityType.DOCUMENT,
            properties={"indexed_at": datetime.now().isoformat()},
            source_doc_id=document_id
        )
        self._add_entity(doc_entity)
        
        # Extract entities and relationships
        entities, relationships = self.extract_entities_and_relationships(
            document_text, document_id
        )
        
        # Add to graph
        for entity in entities:
            self._add_entity(entity)
            # Link entity to document
            self.graph.add_edge(
                entity.id,
                doc_entity.id,
                relation=RelationType.MENTIONED_IN.value
            )
        
        for rel in relationships:
            self._add_relationship(rel)
        
        self.indexed_documents.add(document_id)
        
        return {
            "status": "indexed",
            "document_id": document_id,
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "entities": [e.to_dict() for e in entities],
            "relationships": [r.to_dict() for r in relationships]
        }
    
    def _add_entity(self, entity: Entity) -> None:
        """Add or merge entity into graph."""
        if entity.id in self.entity_registry:
            # Merge properties
            existing = self.entity_registry[entity.id]
            existing.properties.update(entity.properties)
        else:
            self.entity_registry[entity.id] = entity
            self.graph.add_node(
                entity.id,
                name=entity.name,
                type=entity.entity_type.value,
                properties=entity.properties
            )
    
    def _add_relationship(self, rel: Relationship) -> None:
        """Add relationship to graph."""
        self.graph.add_edge(
            rel.source_id,
            rel.target_id,
            relation=rel.relation_type.value,
            properties=rel.properties
        )
    
    def query(self, question: str) -> GraphQueryResult:
        """
        Answer a question using the knowledge graph.
        Uses LLM to understand intent, then graph traversal, then LLM reasoning.
        """
        logger.info(f"GraphRAG query: {question}")
        logger.info(f"Graph has {self.graph.number_of_nodes()} nodes, {len(self.entity_registry)} entities")
        
        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty - no documents indexed")
            return GraphQueryResult(
                answer="No documents have been indexed into the knowledge graph yet.",
                entities=[],
                relationships=[],
                subgraph={},
                confidence=0.0,
                reasoning_path=["No graph data available"]
            )
        
        # Step 1: Identify relevant entities from question
        relevant_entities = self._find_relevant_entities(question)
        logger.info(f"Found {len(relevant_entities)} relevant entities")
        
        # Step 2: Extract subgraph around relevant entities
        subgraph_data = self._extract_subgraph(relevant_entities)
        logger.info(f"Extracted subgraph: {subgraph_data['node_count']} nodes, {subgraph_data['edge_count']} edges")
        
        # Step 3: Use LLM to reason over subgraph and answer
        answer, confidence, reasoning = self._reason_over_graph(
            question, subgraph_data, relevant_entities
        )
        
        # Build relationships list
        relationships = []
        for src, tgt, data in self.graph.edges(data=True):
            if src in [e.id for e in relevant_entities] or tgt in [e.id for e in relevant_entities]:
                try:
                    relationships.append(Relationship(
                        source_id=src,
                        target_id=tgt,
                        relation_type=RelationType(data.get("relation", "mentioned_in")),
                        properties=data.get("properties", {})
                    ))
                except ValueError:
                    pass
        
        return GraphQueryResult(
            answer=answer,
            entities=relevant_entities,
            relationships=relationships,
            subgraph=subgraph_data,
            confidence=confidence,
            reasoning_path=reasoning
        )
    
    def _find_relevant_entities(self, question: str) -> List[Entity]:
        """Find entities relevant to the question."""
        question_lower = question.lower()
        relevant = []
        
        for entity_id, entity in self.entity_registry.items():
            # Simple keyword matching (production would use embeddings)
            if entity.name.lower() in question_lower:
                relevant.append(entity)
            elif any(keyword in question_lower for keyword in [
                entity.entity_type.value,
                entity.name.lower().split()[0] if entity.name else ""
            ]):
                relevant.append(entity)
        
        # If no direct match, get all non-document entities
        if not relevant:
            relevant = [
                e for e in self.entity_registry.values()
                if e.entity_type != EntityType.DOCUMENT
            ][:10]  # Limit to 10
        
        return relevant
    
    def _extract_subgraph(
        self,
        entities: List[Entity],
        max_hops: int = 2
    ) -> Dict[str, Any]:
        """Extract subgraph centered on given entities."""
        entity_ids = {e.id for e in entities}
        nodes = set()
        edges = []
        
        # BFS to find connected nodes within max_hops
        for entity_id in entity_ids:
            if entity_id in self.graph:
                # Get neighbors within max_hops
                for hop in range(1, max_hops + 1):
                    try:
                        neighbors = nx.single_source_shortest_path_length(
                            self.graph, entity_id, cutoff=hop
                        )
                        nodes.update(neighbors.keys())
                    except nx.NetworkXError:
                        pass
        
        # Build subgraph data
        subgraph_nodes = []
        for node_id in nodes:
            if node_id in self.entity_registry:
                entity = self.entity_registry[node_id]
                subgraph_nodes.append({
                    "id": node_id,
                    "name": entity.name,
                    "type": entity.entity_type.value
                })
        
        for src, tgt, data in self.graph.edges(data=True):
            if src in nodes and tgt in nodes:
                edges.append({
                    "source": src,
                    "target": tgt,
                    "relation": data.get("relation", "related_to")
                })
        
        return {
            "nodes": subgraph_nodes,
            "edges": edges,
            "node_count": len(subgraph_nodes),
            "edge_count": len(edges)
        }
    
    def _reason_over_graph(
        self,
        question: str,
        subgraph: Dict[str, Any],
        entities: List[Entity]
    ) -> Tuple[str, float, List[str]]:
        """Use LLM to reason over graph structure and answer question."""
        
        # Format graph context
        nodes_desc = "\n".join([
            f"- {n['name']} ({n['type']})"
            for n in subgraph.get("nodes", [])
        ])
        
        edges_desc = "\n".join([
            f"- {self._get_entity_name(e['source'])} --[{e['relation']}]--> {self._get_entity_name(e['target'])}"
            for e in subgraph.get("edges", [])
        ])
        
        system_prompt = """You are a knowledge graph reasoning assistant. 
Answer questions based on the entities and relationships provided.

Provide:
1. A clear answer based on the graph data
2. Your reasoning path (how you traversed the graph)
3. Confidence level (high/medium/low)

If the information is not in the graph, say so."""

        user_prompt = f"""Question: {question}

## Knowledge Graph Context:

### Entities:
{nodes_desc if nodes_desc else "No entities found"}

### Relationships:
{edges_desc if edges_desc else "No relationships found"}

Answer the question based on this graph. Explain your reasoning path."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            answer = response.choices[0].message.content
            
            # Extract confidence from answer
            confidence = 0.7  # Default medium
            if "high confidence" in answer.lower():
                confidence = 0.9
            elif "low confidence" in answer.lower():
                confidence = 0.4
            elif "not found" in answer.lower() or "no information" in answer.lower():
                confidence = 0.2
            
            # Build reasoning path
            reasoning = [
                f"Found {len(entities)} relevant entities",
                f"Explored subgraph with {subgraph['node_count']} nodes and {subgraph['edge_count']} edges",
                "Applied LLM reasoning over graph structure"
            ]
            
            return answer, confidence, reasoning
            
        except Exception as e:
            logger.error(f"Graph reasoning failed: {e}")
            return f"Error reasoning over graph: {e}", 0.0, ["Error occurred"]
    
    def _get_entity_name(self, entity_id: str) -> str:
        """Get entity name from ID."""
        if entity_id in self.entity_registry:
            return self.entity_registry[entity_id].name
        return entity_id
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        entity_type_counts = {}
        for entity in self.entity_registry.values():
            t = entity.entity_type.value
            entity_type_counts[t] = entity_type_counts.get(t, 0) + 1
        
        relationship_type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "unknown")
            relationship_type_counts[rel] = relationship_type_counts.get(rel, 0) + 1
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "indexed_documents": len(self.indexed_documents),
            "entity_types": entity_type_counts,
            "relationship_types": relationship_type_counts
        }
    
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Get all entities in the graph."""
        return [e.to_dict() for e in self.entity_registry.values()]
    
    def get_entity_relationships(self, entity_id: str) -> Dict[str, Any]:
        """Get all relationships for a specific entity."""
        if entity_id not in self.graph:
            return {"error": "Entity not found"}
        
        outgoing = []
        incoming = []
        
        for _, tgt, data in self.graph.out_edges(entity_id, data=True):
            outgoing.append({
                "target": self._get_entity_name(tgt),
                "target_id": tgt,
                "relation": data.get("relation", "related_to")
            })
        
        for src, _, data in self.graph.in_edges(entity_id, data=True):
            incoming.append({
                "source": self._get_entity_name(src),
                "source_id": src,
                "relation": data.get("relation", "related_to")
            })
        
        return {
            "entity": self.entity_registry.get(entity_id, {}).to_dict() if entity_id in self.entity_registry else {},
            "outgoing_relationships": outgoing,
            "incoming_relationships": incoming
        }
    
    def visualize_graph(self) -> Dict[str, Any]:
        """Return graph data in a format suitable for visualization."""
        nodes = []
        for entity in self.entity_registry.values():
            nodes.append({
                "id": entity.id,
                "label": entity.name,
                "type": entity.entity_type.value,
                "color": self._get_type_color(entity.entity_type)
            })
        
        edges = []
        for src, tgt, data in self.graph.edges(data=True):
            edges.append({
                "from": src,
                "to": tgt,
                "label": data.get("relation", ""),
                "arrows": "to"
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def _get_type_color(self, entity_type: EntityType) -> str:
        """Get color for entity type visualization."""
        colors = {
            EntityType.SHIPMENT: "#FF6B6B",
            EntityType.SHIPPER: "#4ECDC4",
            EntityType.CONSIGNEE: "#45B7D1",
            EntityType.CARRIER: "#96CEB4",
            EntityType.LOCATION: "#FFEAA7",
            EntityType.EQUIPMENT: "#DDA0DD",
            EntityType.COMMODITY: "#98D8C8",
            EntityType.RATE: "#F7DC6F",
            EntityType.DATE: "#BB8FCE",
            EntityType.DOCUMENT: "#85C1E9"
        }
        return colors.get(entity_type, "#CCCCCC")
    
    def clear_graph(self) -> None:
        """Clear all data from the graph."""
        self.graph.clear()
        self.entity_registry.clear()
        self.indexed_documents.clear()
