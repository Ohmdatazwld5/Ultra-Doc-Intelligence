"""
Structured Data Extractor v2 for Ultra Doc-Intelligence.
Dynamic LLM-first extraction with validation - no brittle regex patterns.
Production-ready approach using semantic understanding.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from openai import OpenAI

logger = logging.getLogger(__name__)


class FieldConfidence(Enum):
    """Confidence levels for extracted fields."""
    HIGH = "high"       # Explicitly stated, unambiguous
    MEDIUM = "medium"   # Inferred from context
    LOW = "low"         # Uncertain, may need verification
    NOT_FOUND = "not_found"


@dataclass
class ExtractedField:
    """A single extracted field with metadata."""
    value: Optional[str] = None
    confidence: FieldConfidence = FieldConfidence.NOT_FOUND
    source_text: Optional[str] = None  # The text snippet it was extracted from
    validation_passed: bool = True
    notes: List[str] = field(default_factory=list)


@dataclass
class ShipmentData:
    """Structured shipment data extracted from documents."""
    shipment_id: Optional[str] = None
    shipper: Optional[str] = None
    consignee: Optional[str] = None
    pickup_datetime: Optional[str] = None
    delivery_datetime: Optional[str] = None
    equipment_type: Optional[str] = None
    mode: Optional[str] = None
    rate: Optional[str] = None
    currency: Optional[str] = None
    weight: Optional[str] = None
    carrier_name: Optional[str] = None
    
    # Metadata
    extraction_confidence: float = 0.0
    fields_extracted: int = 0
    fields_total: int = 11
    extraction_notes: List[str] = field(default_factory=list)
    field_details: Dict[str, ExtractedField] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with clean output."""
        return {
            "shipment_id": self.shipment_id,
            "shipper": self.shipper,
            "consignee": self.consignee,
            "pickup_datetime": self.pickup_datetime,
            "delivery_datetime": self.delivery_datetime,
            "equipment_type": self.equipment_type,
            "mode": self.mode,
            "rate": self.rate,
            "currency": self.currency,
            "weight": self.weight,
            "carrier_name": self.carrier_name,
            "_metadata": {
                "extraction_confidence": round(self.extraction_confidence, 4),
                "fields_extracted": self.fields_extracted,
                "fields_total": self.fields_total,
                "extraction_notes": self.extraction_notes
            }
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class DynamicExtractor:
    """
    Production-ready structured data extractor using dynamic LLM extraction.
    No hardcoded regex - uses semantic understanding + validation.
    """
    
    # Field definitions with semantic descriptions and validation rules
    FIELD_SCHEMA = {
        "shipment_id": {
            "description": "Unique identifier for this shipment/load",
            "semantic_hints": [
                "confirmation number", "reference number", "PRO number", 
                "BOL number", "load number", "booking number", "order number",
                "shipment ID", "tracking number", "waybill number"
            ],
            "validation": {
                "min_length": 3,
                "max_length": 50,
                "must_contain_alphanum": True,
                "reject_patterns": ["date", "time", "phone", "fax"]
            }
        },
        "shipper": {
            "description": "Company/party shipping the goods (origin)",
            "semantic_hints": [
                "shipper", "ship from", "origin", "consignor", 
                "pickup location", "sender", "supplier"
            ],
            "validation": {
                "min_length": 2,
                "max_length": 200,
                "should_be_company_name": True
            }
        },
        "consignee": {
            "description": "Company/party receiving the goods (destination)",
            "semantic_hints": [
                "consignee", "ship to", "destination", "receiver",
                "delivery location", "deliver to", "customer"
            ],
            "validation": {
                "min_length": 2,
                "max_length": 200,
                "should_be_company_name": True
            }
        },
        "pickup_datetime": {
            "description": "Scheduled pickup date and time",
            "semantic_hints": [
                "pickup date", "pickup time", "ship date", 
                "ready date", "origin date", "collection date"
            ],
            "validation": {
                "should_be_datetime": True
            }
        },
        "delivery_datetime": {
            "description": "Scheduled delivery date and time",
            "semantic_hints": [
                "delivery date", "delivery time", "arrival date",
                "due date", "expected delivery", "ETA"
            ],
            "validation": {
                "should_be_datetime": True
            }
        },
        "equipment_type": {
            "description": "Type of trailer/container",
            "semantic_hints": [
                "equipment type", "trailer type", "container type",
                "vehicle type", "equipment"
            ],
            "validation": {
                "valid_values": [
                    "dry van", "reefer", "flatbed", "step deck", 
                    "container", "tanker", "van", "refrigerated",
                    "53'", "48'", "40'", "20'", "intermodal"
                ]
            }
        },
        "mode": {
            "description": "Transportation mode",
            "semantic_hints": [
                "mode", "service type", "transport mode", "shipping method"
            ],
            "validation": {
                "canonical_values": {
                    "TL": ["tl", "truckload", "truck load", "full truckload"],
                    "FTL": ["ftl", "full truckload", "full truck load"],
                    "LTL": ["ltl", "less than truckload", "less-than-truckload"],
                    "INTERMODAL": ["intermodal", "rail", "im"],
                    "AIR": ["air", "air freight"],
                    "OCEAN": ["ocean", "sea", "fcl", "lcl"]
                }
            }
        },
        "rate": {
            "description": "Total rate/cost for the shipment",
            "semantic_hints": [
                "total rate", "total amount", "total cost", "rate",
                "line haul", "freight charges", "amount due"
            ],
            "validation": {
                "should_be_numeric": True,
                "min_value": 0,
                "max_value": 1000000
            }
        },
        "currency": {
            "description": "Currency of the rate",
            "semantic_hints": ["currency", "cur"],
            "validation": {
                "valid_values": ["USD", "CAD", "EUR", "GBP", "MXN", "AUD"]
            }
        },
        "weight": {
            "description": "Total shipment weight with unit",
            "semantic_hints": [
                "weight", "gross weight", "total weight", "lbs", "kg"
            ],
            "validation": {
                "should_contain_number": True,
                "should_contain_unit": True
            }
        },
        "carrier_name": {
            "description": "Trucking/transport company",
            "semantic_hints": [
                "carrier", "carrier name", "trucking company",
                "transport company", "hauler", "motor carrier"
            ],
            "validation": {
                "min_length": 2,
                "max_length": 200,
                "should_be_company_name": True
            }
        }
    }
    
    def __init__(
        self, 
        api_key: str, 
        llm_model: str = "grok-4-1-fast-reasoning",
        base_url: str = "https://api.x.ai/v1",
        max_retries: int = 2
    ):
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_model = llm_model
        self.max_retries = max_retries
    
    def extract(self, document_text: str) -> ShipmentData:
        """
        Extract structured shipment data using dynamic LLM extraction.
        
        Args:
            document_text: Full text of the document
            
        Returns:
            ShipmentData with extracted fields and confidence scores
        """
        # Step 1: LLM extraction with field-level confidence
        raw_results, field_confidences = self._llm_extract_with_confidence(document_text)
        
        # Step 2: Validate and clean extracted values
        validated_results = self._validate_and_clean(raw_results)
        
        # Step 3: If critical fields missing, retry with focused extraction
        if self._should_retry(validated_results):
            retry_results, retry_confidences = self._focused_extraction(
                document_text, 
                validated_results
            )
            validated_results = self._merge_results(validated_results, retry_results)
            field_confidences.update(retry_confidences)
        
        # Step 4: Build ShipmentData object
        shipment = self._build_shipment_data(validated_results, field_confidences)
        
        return shipment
    
    def _llm_extract_with_confidence(
        self, 
        text: str
    ) -> Tuple[Dict[str, Any], Dict[str, FieldConfidence]]:
        """Extract all fields with confidence ratings."""
        
        # Build dynamic field descriptions
        field_descriptions = "\n".join([
            f"- **{field}**: {schema['description']}. Look for: {', '.join(schema['semantic_hints'][:3])}"
            for field, schema in self.FIELD_SCHEMA.items()
        ])
        
        system_prompt = f"""You are an expert logistics document analyzer. Extract structured data from shipping documents.

## Fields to Extract:
{field_descriptions}

## Output Format:
Return a JSON object with this EXACT structure:
{{
  "fields": {{
    "shipment_id": {{"value": "...", "confidence": "high|medium|low|not_found", "source": "brief quote from doc"}},
    "shipper": {{"value": "...", "confidence": "high|medium|low|not_found", "source": "..."}},
    "consignee": {{"value": "...", "confidence": "high|medium|low|not_found", "source": "..."}},
    "pickup_datetime": {{"value": "YYYY-MM-DD HH:MM or null", "confidence": "...", "source": "..."}},
    "delivery_datetime": {{"value": "YYYY-MM-DD HH:MM or null", "confidence": "...", "source": "..."}},
    "equipment_type": {{"value": "...", "confidence": "...", "source": "..."}},
    "mode": {{"value": "TL|LTL|FTL|INTERMODAL|AIR|OCEAN or null", "confidence": "...", "source": "..."}},
    "rate": {{"value": "numeric only, no $ or currency symbols", "confidence": "...", "source": "..."}},
    "currency": {{"value": "USD|CAD|EUR|etc or null", "confidence": "...", "source": "..."}},
    "weight": {{"value": "number with unit like 42500 lbs", "confidence": "...", "source": "..."}},
    "carrier_name": {{"value": "...", "confidence": "...", "source": "..."}}
  }}
}}

## Rules:
1. Set confidence="high" if explicitly stated
2. Set confidence="medium" if inferred from context  
3. Set confidence="low" if uncertain
4. Set confidence="not_found" and value=null if not in document
5. For rate: use the TOTAL amount including surcharges
6. For dates: convert to YYYY-MM-DD format when possible
7. Return ONLY valid JSON, no markdown"""

        user_prompt = f"""Analyze this document and extract all shipment data:

---
{text[:10000]}
---

Return the JSON with all fields extracted."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            result_text = self._clean_json_response(response.choices[0].message.content)
            data = json.loads(result_text)
            
            # Parse results
            results = {}
            confidences = {}
            
            fields_data = data.get("fields", data)  # Handle both formats
            for field_name in self.FIELD_SCHEMA.keys():
                field_data = fields_data.get(field_name, {})
                if isinstance(field_data, dict):
                    value = field_data.get("value")
                    conf_str = field_data.get("confidence", "not_found")
                else:
                    value = field_data
                    conf_str = "medium" if value else "not_found"
                
                # Clean null-like values
                if value in [None, "null", "NULL", "None", "", "N/A", "n/a", "not found", "Not found"]:
                    value = None
                    conf_str = "not_found"
                
                results[field_name] = value
                confidences[field_name] = FieldConfidence(conf_str) if conf_str in ["high", "medium", "low", "not_found"] else FieldConfidence.MEDIUM
            
            return results, confidences
            
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return {}, {f: FieldConfidence.NOT_FOUND for f in self.FIELD_SCHEMA.keys()}
    
    def _clean_json_response(self, text: str) -> str:
        """Clean markdown and other formatting from JSON response."""
        text = text.strip()
        
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]  # Remove ```json
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            text = text[start:end]
        
        return text
    
    def _validate_and_clean(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted values."""
        validated = {}
        
        for field_name, value in results.items():
            if value is None:
                validated[field_name] = None
                continue
            
            schema = self.FIELD_SCHEMA.get(field_name, {})
            validation = schema.get("validation", {})
            
            clean_value = str(value).strip()
            is_valid = True
            
            # Length validation
            min_len = validation.get("min_length", 0)
            max_len = validation.get("max_length", 10000)
            if len(clean_value) < min_len or len(clean_value) > max_len:
                is_valid = False
            
            # Reject patterns (for shipment_id)
            reject_patterns = validation.get("reject_patterns", [])
            for pattern in reject_patterns:
                if pattern.lower() in clean_value.lower():
                    is_valid = False
                    break
            
            # Numeric validation (for rate)
            if validation.get("should_be_numeric"):
                # Extract numeric value
                numeric_match = re.search(r'[\d,]+\.?\d*', clean_value)
                if numeric_match:
                    clean_value = numeric_match.group().replace(",", "")
                else:
                    is_valid = False
            
            # Mode canonicalization
            if field_name == "mode" and validation.get("canonical_values"):
                clean_value = self._canonicalize_mode(clean_value, validation["canonical_values"])
            
            # Currency defaults
            if field_name == "currency" and not clean_value:
                # Will be set based on rate context later
                clean_value = None
            
            validated[field_name] = clean_value if is_valid else None
        
        return validated
    
    def _canonicalize_mode(self, value: str, canonical_map: Dict[str, List[str]]) -> str:
        """Convert mode to canonical form."""
        value_lower = value.lower().strip()
        for canonical, variants in canonical_map.items():
            if value_lower in variants or value_lower == canonical.lower():
                return canonical
        return value.upper() if len(value) <= 3 else value
    
    def _should_retry(self, results: Dict[str, Any]) -> bool:
        """Check if we should retry extraction for missing critical fields."""
        critical_fields = ["shipment_id", "shipper", "consignee", "carrier_name"]
        found = sum(1 for f in critical_fields if results.get(f))
        return found < 2  # Retry if less than half of critical fields found
    
    def _focused_extraction(
        self, 
        text: str, 
        existing_results: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, FieldConfidence]]:
        """Retry extraction with focused prompts for missing fields."""
        missing_fields = [f for f, v in existing_results.items() if v is None]
        
        if not missing_fields:
            return {}, {}
        
        # Build focused prompt for missing fields only
        field_list = ", ".join(missing_fields)
        
        prompt = f"""I need to find these SPECIFIC fields that were missed in first extraction: {field_list}

Look VERY carefully in the document for:
- Shipper/Ship From section - extract the COMPANY NAME
- Consignee/Ship To section - extract the COMPANY NAME  
- Carrier section - extract the CARRIER COMPANY NAME
- Any reference/confirmation/PRO/BOL numbers for shipment_id
- Dates/times for pickup and delivery

Document:
---
{text[:8000]}
---

Return JSON with ONLY the missing fields you can find:
{{"field_name": "value", ...}}"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = self._clean_json_response(response.choices[0].message.content)
            data = json.loads(result_text)
            
            results = {}
            confidences = {}
            for field in missing_fields:
                value = data.get(field)
                if value and value not in ["null", "None", "", "N/A", "not found"]:
                    results[field] = value
                    confidences[field] = FieldConfidence.MEDIUM
            
            return results, confidences
            
        except Exception as e:
            logger.error(f"Focused extraction failed: {e}")
            return {}, {}
    
    def _merge_results(
        self, 
        existing: Dict[str, Any], 
        new: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge results, preferring new values for None fields."""
        merged = existing.copy()
        for field, value in new.items():
            if merged.get(field) is None and value is not None:
                merged[field] = value
        return merged
    
    def _build_shipment_data(
        self, 
        results: Dict[str, Any],
        confidences: Dict[str, FieldConfidence]
    ) -> ShipmentData:
        """Build final ShipmentData object with statistics."""
        
        shipment = ShipmentData(
            shipment_id=results.get("shipment_id"),
            shipper=results.get("shipper"),
            consignee=results.get("consignee"),
            pickup_datetime=results.get("pickup_datetime"),
            delivery_datetime=results.get("delivery_datetime"),
            equipment_type=results.get("equipment_type"),
            mode=results.get("mode"),
            rate=results.get("rate"),
            currency=results.get("currency") or ("USD" if results.get("rate") else None),
            weight=results.get("weight"),
            carrier_name=results.get("carrier_name"),
        )
        
        # Calculate fields extracted
        field_values = [
            shipment.shipment_id, shipment.shipper, shipment.consignee,
            shipment.pickup_datetime, shipment.delivery_datetime,
            shipment.equipment_type, shipment.mode, shipment.rate,
            shipment.currency, shipment.weight, shipment.carrier_name
        ]
        shipment.fields_extracted = sum(1 for v in field_values if v is not None)
        
        # Calculate confidence score
        shipment.extraction_confidence = self._calculate_confidence(
            shipment.fields_extracted,
            shipment.fields_total,
            confidences
        )
        
        # Build field details
        for field_name in self.FIELD_SCHEMA.keys():
            shipment.field_details[field_name] = ExtractedField(
                value=results.get(field_name),
                confidence=confidences.get(field_name, FieldConfidence.NOT_FOUND)
            )
        
        return shipment
    
    def _calculate_confidence(
        self,
        fields_extracted: int,
        fields_total: int,
        confidences: Dict[str, FieldConfidence]
    ) -> float:
        """Calculate overall extraction confidence."""
        
        # Base: field completion rate
        completion_rate = fields_extracted / fields_total
        
        # Confidence weights
        conf_weights = {
            FieldConfidence.HIGH: 1.0,
            FieldConfidence.MEDIUM: 0.7,
            FieldConfidence.LOW: 0.4,
            FieldConfidence.NOT_FOUND: 0.0
        }
        
        # Average confidence of extracted fields
        extracted_confidences = [
            conf_weights[c] for f, c in confidences.items()
            if c != FieldConfidence.NOT_FOUND
        ]
        avg_confidence = sum(extracted_confidences) / len(extracted_confidences) if extracted_confidences else 0
        
        # Critical field bonus
        critical_fields = ["shipment_id", "shipper", "consignee", "rate"]
        critical_bonus = sum(
            0.05 for f in critical_fields 
            if confidences.get(f) in [FieldConfidence.HIGH, FieldConfidence.MEDIUM]
        )
        
        # Final score
        score = (completion_rate * 0.5) + (avg_confidence * 0.3) + critical_bonus
        return min(1.0, max(0.0, score))


# Alias for backward compatibility
StructuredExtractor = DynamicExtractor
