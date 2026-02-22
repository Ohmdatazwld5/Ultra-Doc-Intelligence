"""
Structured Data Extractor for Ultra Doc-Intelligence.
Extracts standardized shipment data from logistics documents.
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from openai import OpenAI

from app.config import SHIPMENT_FIELDS

logger = logging.getLogger(__name__)


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with clean output."""
        result = {
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
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class StructuredExtractor:
    """
    Extracts structured shipment data from logistics documents.
    Uses a combination of pattern matching and LLM extraction (xAI Grok).
    """
    
    def __init__(self, api_key: str, llm_model: str = "grok-4-1-fast-reasoning", base_url: str = "https://api.x.ai/v1"):
        self.llm_client = OpenAI(api_key=api_key, base_url=base_url)
        self.llm_model = llm_model
        
        # Regex patterns for common logistics data
        self.patterns = {
            "shipment_id": [
                r'(?:PRO|BOL|Order|Load|Shipment|Reference|Booking)[\s#:]*([A-Z0-9\-]{4,20})',
                r'(?:PRO\s*(?:Number|#|No\.?)?)\s*:?\s*([A-Z0-9\-]+)',
                r'(?:Load\s*(?:Number|#|No\.?)?)\s*:?\s*([A-Z0-9\-]+)',
            ],
            "rate": [
                r'\$\s*([\d,]+\.?\d*)',
                r'(?:Total|Rate|Amount|Cost)[\s:]*\$?\s*([\d,]+\.?\d*)',
                r'(?:Line\s*Haul|Freight)[\s:]*\$?\s*([\d,]+\.?\d*)',
            ],
            "weight": [
                r'([\d,]+\.?\d*)\s*(?:lbs?|pounds?|LBS)',
                r'([\d,]+\.?\d*)\s*(?:kgs?|kilograms?|KGS)',
                r'(?:Weight|Gross\s*Weight)[\s:]*(\d[\d,\.]*)\s*(?:lbs?|kgs?)?',
            ],
            "currency": [
                r'(?:Currency|CUR)[\s:]*([A-Z]{3})',
                r'\b(USD|CAD|EUR|GBP|MXN)\b',
            ],
            "equipment_type": [
                r'(?:Equipment|Trailer|Container)[\s:]*([A-Za-z\s]+(?:Van|Flatbed|Reefer|Container|Tanker))',
                r'\b(Dry\s*Van|Reefer|Flatbed|Step\s*Deck|Tanker|Intermodal|Container)\b',
                r'(?:53|48|40|20)[\'"]?\s*(Dry\s*Van|Reefer|Flatbed|Container)',
            ],
            "mode": [
                r'(?:Mode|Service)[\s:]*([A-Z]{2,3})',
                r'\b(TL|LTL|FTL|Truckload|Less[\s\-]?Than[\s\-]?Truckload|Intermodal|Air|Ocean|Rail)\b',
            ],
            "datetime": [
                r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s*(?:@|at)?\s*(\d{1,2}:\d{2}\s*(?:AM|PM)?)?',
                r'(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\s*(?:@|at)?\s*(\d{1,2}:\d{2})?',
                r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
            ]
        }
    
    def extract(self, document_text: str, use_llm: bool = True) -> ShipmentData:
        """
        Extract structured shipment data from document text.
        
        Args:
            document_text: Full text of the document
            use_llm: Whether to use LLM for enhanced extraction
            
        Returns:
            ShipmentData with extracted fields
        """
        # First, try pattern-based extraction
        pattern_results = self._pattern_extraction(document_text)
        
        # Then, use LLM for comprehensive extraction
        if use_llm:
            llm_results = self._llm_extraction(document_text)
            
            # Merge results, preferring LLM but validating with patterns
            final_results = self._merge_results(pattern_results, llm_results)
        else:
            final_results = pattern_results
        
        # Create ShipmentData object
        shipment = ShipmentData(
            shipment_id=final_results.get("shipment_id"),
            shipper=final_results.get("shipper"),
            consignee=final_results.get("consignee"),
            pickup_datetime=final_results.get("pickup_datetime"),
            delivery_datetime=final_results.get("delivery_datetime"),
            equipment_type=final_results.get("equipment_type"),
            mode=final_results.get("mode"),
            rate=final_results.get("rate"),
            currency=final_results.get("currency"),
            weight=final_results.get("weight"),
            carrier_name=final_results.get("carrier_name"),
            extraction_notes=final_results.get("_notes", [])
        )
        
        # Calculate extraction statistics
        shipment.fields_extracted = sum(
            1 for field in [
                shipment.shipment_id, shipment.shipper, shipment.consignee,
                shipment.pickup_datetime, shipment.delivery_datetime,
                shipment.equipment_type, shipment.mode, shipment.rate,
                shipment.currency, shipment.weight, shipment.carrier_name
            ]
            if field is not None
        )
        
        # Calculate confidence based on extraction completeness and validation
        shipment.extraction_confidence = self._calculate_extraction_confidence(
            shipment, pattern_results, final_results
        )
        
        return shipment
    
    def _pattern_extraction(self, text: str) -> Dict[str, Any]:
        """Extract data using regex patterns."""
        results = {"_notes": []}
        
        # Shipment ID
        for pattern in self.patterns["shipment_id"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results["shipment_id"] = match.group(1).strip()
                break
        
        # Rate
        for pattern in self.patterns["rate"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rate_value = match.group(1).replace(",", "")
                results["rate"] = rate_value
                break
        
        # Weight
        for pattern in self.patterns["weight"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                weight = match.group(1) if match.lastindex >= 1 else match.group(0)
                # Try to include unit
                full_match = match.group(0)
                results["weight"] = full_match.strip()
                break
        
        # Currency
        for pattern in self.patterns["currency"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results["currency"] = match.group(1).upper()
                break
        # Default to USD if rate found but no currency
        if "rate" in results and "currency" not in results:
            if "$" in text:
                results["currency"] = "USD"
        
        # Equipment type
        for pattern in self.patterns["equipment_type"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results["equipment_type"] = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
                break
        
        # Mode
        for pattern in self.patterns["mode"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                mode = match.group(1).upper()
                # Normalize common modes
                mode_map = {
                    "TRUCKLOAD": "TL",
                    "FULL TRUCKLOAD": "FTL",
                    "LESS THAN TRUCKLOAD": "LTL",
                    "LESSTHANTRUCKLOAD": "LTL"
                }
                results["mode"] = mode_map.get(mode.replace("-", " ").replace("  ", " "), mode)
                break
        
        return results
    
    def _llm_extraction(self, text: str) -> Dict[str, Any]:
        """Use LLM for comprehensive extraction."""
        
        # Build field descriptions for the prompt
        field_descriptions = "\n".join([
            f"- {field}: {info['description']}"
            for field, info in SHIPMENT_FIELDS.items()
        ])
        
        system_prompt = """You are a logistics document data extractor. Extract structured shipment information from documents.

RULES:
1. Only extract information explicitly stated in the document
2. Return null for fields not found
3. For dates/times, use ISO format (YYYY-MM-DD HH:MM) when possible
4. For rates, include just the numeric value (no currency symbol)
5. For weight, include the unit (e.g., "45000 lbs")
6. Be precise - do not guess or infer missing data

Return a JSON object with these fields:
""" + field_descriptions

        user_prompt = f"""Extract shipment data from this logistics document:

---
{text[:8000]}  
---

Return ONLY a valid JSON object with the extracted fields. Use null for missing fields."""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                temperature=0.0,  # Deterministic for extraction
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            result_text = response.choices[0].message.content
            results = json.loads(result_text)
            
            # Clean up null strings
            for key, value in results.items():
                if value in ["null", "NULL", "None", "", "N/A", "n/a"]:
                    results[key] = None
            
            results["_notes"] = ["LLM extraction completed"]
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {"_notes": [f"LLM extraction failed: {e}"]}
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return {"_notes": [f"LLM extraction error: {e}"]}
    
    def _merge_results(
        self,
        pattern_results: Dict[str, Any],
        llm_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge pattern and LLM extraction results.
        LLM results are preferred but validated against patterns where possible.
        """
        merged = {"_notes": []}
        
        # Collect notes
        merged["_notes"].extend(pattern_results.get("_notes", []))
        merged["_notes"].extend(llm_results.get("_notes", []))
        
        # Fields to merge
        fields = [
            "shipment_id", "shipper", "consignee", "pickup_datetime",
            "delivery_datetime", "equipment_type", "mode", "rate",
            "currency", "weight", "carrier_name"
        ]
        
        for field in fields:
            pattern_val = pattern_results.get(field)
            llm_val = llm_results.get(field)
            
            if llm_val and pattern_val:
                # Both have values - prefer LLM but note if they differ significantly
                merged[field] = llm_val
                if str(pattern_val).lower() not in str(llm_val).lower():
                    merged["_notes"].append(f"{field}: LLM='{llm_val}', Pattern='{pattern_val}'")
            elif llm_val:
                merged[field] = llm_val
            elif pattern_val:
                merged[field] = pattern_val
                merged["_notes"].append(f"{field}: extracted via pattern only")
            else:
                merged[field] = None
        
        return merged
    
    def _calculate_extraction_confidence(
        self,
        shipment: ShipmentData,
        pattern_results: Dict[str, Any],
        final_results: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the extraction."""
        
        # Base confidence on field extraction rate
        base_confidence = shipment.fields_extracted / shipment.fields_total
        
        # Boost confidence for critical fields
        critical_fields = ["shipment_id", "shipper", "consignee", "rate"]
        critical_found = sum(
            1 for f in critical_fields
            if getattr(shipment, f) is not None
        )
        critical_bonus = (critical_found / len(critical_fields)) * 0.2
        
        # Boost confidence for pattern validation
        pattern_validated = sum(
            1 for field in ["shipment_id", "rate", "weight", "currency", "mode"]
            if pattern_results.get(field) and final_results.get(field)
        )
        validation_bonus = (pattern_validated / 5) * 0.1
        
        confidence = base_confidence * 0.7 + critical_bonus + validation_bonus
        
        return min(1.0, max(0.0, confidence))
    
    def extract_with_context(
        self,
        document_text: str,
        retrieved_chunks: List[Any] = None
    ) -> ShipmentData:
        """
        Extract structured data with optional RAG context.
        Useful when specific chunks have been retrieved for extraction.
        """
        if retrieved_chunks:
            # Combine chunk content for focused extraction
            chunk_text = "\n\n".join([
                rc.chunk.content if hasattr(rc, 'chunk') else str(rc)
                for rc in retrieved_chunks
            ])
            # Use chunks if they're more focused, otherwise use full document
            if len(chunk_text) > 200:
                return self.extract(chunk_text, use_llm=True)
        
        return self.extract(document_text, use_llm=True)
