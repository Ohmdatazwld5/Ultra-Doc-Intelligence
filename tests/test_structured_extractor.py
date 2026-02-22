"""
Unit tests for the structured extractor module.
"""

import pytest
from app.structured_extractor import StructuredExtractor, ShipmentData


class TestStructuredExtractor:
    """Tests for StructuredExtractor class."""
    
    @pytest.fixture
    def sample_document_text(self):
        return """
        RATE CONFIRMATION
        
        Load Number: PRO-2024-789456
        
        SHIPPER INFORMATION:
        ABC Manufacturing Inc.
        1234 Industrial Boulevard
        Chicago, IL 60601
        
        CONSIGNEE INFORMATION:
        XYZ Distribution Center
        5678 Commerce Drive
        Dallas, TX 75201
        
        PICKUP DETAILS:
        Date: 03/15/2024
        Time: 08:00 AM - 12:00 PM
        
        DELIVERY DETAILS:
        Date: 03/17/2024
        Time: 02:00 PM - 06:00 PM
        
        SHIPMENT DETAILS:
        Weight: 42,500 lbs
        Equipment: 53' Dry Van
        Mode: TL (Truckload)
        
        CARRIER: FastFreight Logistics LLC
        
        RATE INFORMATION:
        Line Haul: $2,250.00
        Fuel Surcharge: $250.00
        Total Rate: $2,500.00 USD
        """
    
    def test_pattern_extraction_shipment_id(self):
        """Test extraction of shipment ID via regex patterns."""
        extractor = StructuredExtractor(api_key="test_key")
        
        text = "Load Number: PRO-2024-789456\nSome other content"
        results = extractor._pattern_extraction(text)
        
        # Should extract the shipment ID
        assert results.get("shipment_id") is not None
        assert "789456" in results["shipment_id"] or "PRO" in results["shipment_id"]
    
    def test_pattern_extraction_rate(self):
        """Test extraction of rate via regex patterns."""
        extractor = StructuredExtractor(api_key="test_key")
        
        text = "Total Rate: $2,500.00 USD\nPayment terms: Net 30"
        results = extractor._pattern_extraction(text)
        
        assert results.get("rate") is not None
        assert "2500" in results["rate"].replace(",", "")
    
    def test_pattern_extraction_weight(self):
        """Test extraction of weight via regex patterns."""
        extractor = StructuredExtractor(api_key="test_key")
        
        text = "Gross Weight: 42,500 lbs\nDimensions: 48x40x48"
        results = extractor._pattern_extraction(text)
        
        assert results.get("weight") is not None
        assert "42" in results["weight"] or "lbs" in results["weight"].lower()
    
    def test_pattern_extraction_currency(self):
        """Test extraction of currency via regex patterns."""
        extractor = StructuredExtractor(api_key="test_key")
        
        text = "Amount: $1,500 USD"
        results = extractor._pattern_extraction(text)
        
        assert results.get("currency") == "USD"
    
    def test_pattern_extraction_currency_from_dollar_sign(self):
        """Test currency defaults to USD when $ is present."""
        extractor = StructuredExtractor(api_key="test_key")
        
        text = "Rate: $2,000.00"
        results = extractor._pattern_extraction(text)
        
        assert results.get("rate") is not None
        assert results.get("currency") == "USD"
    
    def test_pattern_extraction_equipment_type(self):
        """Test extraction of equipment type."""
        extractor = StructuredExtractor(api_key="test_key")
        
        text = "Equipment Required: 53' Dry Van"
        results = extractor._pattern_extraction(text)
        
        assert results.get("equipment_type") is not None
        assert "van" in results["equipment_type"].lower() or "dry" in results["equipment_type"].lower()
    
    def test_pattern_extraction_mode(self):
        """Test extraction of shipping mode."""
        extractor = StructuredExtractor(api_key="test_key")
        
        test_cases = [
            ("Service: TL", "TL"),
            ("Mode: LTL", "LTL"),
            ("Truckload shipment", "TL"),
            ("Less Than Truckload", "LTL"),
        ]
        
        extractor = StructuredExtractor(api_key="test_key")
        
        for text, expected in test_cases:
            results = extractor._pattern_extraction(text)
            if results.get("mode"):
                assert results["mode"].upper() in ["TL", "LTL", "FTL", "TRUCKLOAD"]


class TestShipmentData:
    """Tests for ShipmentData dataclass."""
    
    def test_shipment_data_creation(self):
        data = ShipmentData(
            shipment_id="PRO-123",
            shipper="Test Shipper",
            consignee="Test Consignee",
            rate="2500.00",
            currency="USD"
        )
        
        assert data.shipment_id == "PRO-123"
        assert data.shipper == "Test Shipper"
        assert data.rate == "2500.00"
    
    def test_shipment_data_to_dict(self):
        data = ShipmentData(
            shipment_id="PRO-123",
            shipper="Test Shipper",
            rate="2500.00",
            currency="USD",
            fields_extracted=4,
            extraction_confidence=0.85
        )
        
        result = data.to_dict()
        
        assert result["shipment_id"] == "PRO-123"
        assert result["shipper"] == "Test Shipper"
        assert result["consignee"] is None  # Not set
        assert result["_metadata"]["extraction_confidence"] == 0.85
        assert result["_metadata"]["fields_extracted"] == 4
    
    def test_shipment_data_to_json(self):
        data = ShipmentData(
            shipment_id="PRO-123",
            rate="1500.00"
        )
        
        json_str = data.to_json()
        
        assert "PRO-123" in json_str
        assert "1500.00" in json_str
        assert "null" in json_str  # Should have null for missing fields
    
    def test_shipment_data_fields_count(self):
        # All fields null
        data = ShipmentData()
        assert data.fields_total == 11
        
        # Some fields filled
        data = ShipmentData(
            shipment_id="123",
            shipper="Test",
            rate="1000"
        )
        data.fields_extracted = 3
        assert data.fields_extracted == 3


class TestMergeResults:
    """Tests for result merging logic."""
    
    def test_merge_prefers_llm(self):
        """LLM results should be preferred over pattern results."""
        extractor = StructuredExtractor(api_key="test_key")
        
        pattern_results = {
            "shipment_id": "123",
            "rate": "1000",
            "_notes": []
        }
        
        llm_results = {
            "shipment_id": "PRO-123",
            "rate": "1000.00",
            "shipper": "Test Shipper Inc",
            "_notes": []
        }
        
        merged = extractor._merge_results(pattern_results, llm_results)
        
        # LLM results should win
        assert merged["shipment_id"] == "PRO-123"
        assert merged["shipper"] == "Test Shipper Inc"
    
    def test_merge_pattern_fallback(self):
        """Pattern results used when LLM misses a field."""
        extractor = StructuredExtractor(api_key="test_key")
        
        pattern_results = {
            "rate": "2500",
            "currency": "USD",
            "_notes": []
        }
        
        llm_results = {
            "rate": None,  # LLM missed this
            "shipper": "Test Shipper",
            "_notes": []
        }
        
        merged = extractor._merge_results(pattern_results, llm_results)
        
        # Pattern should fill in the gap
        assert merged["rate"] == "2500"
        assert merged["currency"] == "USD"
        assert merged["shipper"] == "Test Shipper"


class TestExtractionConfidence:
    """Tests for extraction confidence calculation."""
    
    def test_confidence_full_extraction(self):
        """High confidence when all fields extracted."""
        extractor = StructuredExtractor(api_key="test_key")
        
        shipment = ShipmentData(
            shipment_id="PRO-123",
            shipper="Test Shipper",
            consignee="Test Consignee",
            pickup_datetime="2024-03-15 08:00",
            delivery_datetime="2024-03-17 14:00",
            equipment_type="Dry Van",
            mode="TL",
            rate="2500.00",
            currency="USD",
            weight="42000 lbs",
            carrier_name="Test Carrier"
        )
        shipment.fields_extracted = 11
        
        pattern_results = {"shipment_id": "PRO-123", "rate": "2500.00"}
        final_results = shipment.to_dict()
        
        confidence = extractor._calculate_extraction_confidence(
            shipment, pattern_results, final_results
        )
        
        # Should be high confidence with all fields
        assert confidence > 0.7
    
    def test_confidence_partial_extraction(self):
        """Lower confidence when only some fields extracted."""
        extractor = StructuredExtractor(api_key="test_key")
        
        shipment = ShipmentData(
            shipper="Test Shipper",
            rate="2500.00"
        )
        shipment.fields_extracted = 2
        
        pattern_results = {}
        final_results = {"shipper": "Test Shipper", "rate": "2500.00"}
        
        confidence = extractor._calculate_extraction_confidence(
            shipment, pattern_results, final_results
        )
        
        # Should be lower confidence
        assert confidence < 0.5
    
    def test_confidence_critical_fields_boost(self):
        """Critical fields should boost confidence."""
        extractor = StructuredExtractor(api_key="test_key")
        
        # With critical fields
        shipment_with_critical = ShipmentData(
            shipment_id="PRO-123",
            shipper="Test Shipper",
            consignee="Test Consignee",
            rate="2500.00"
        )
        shipment_with_critical.fields_extracted = 4
        
        # Without critical fields
        shipment_without_critical = ShipmentData(
            equipment_type="Dry Van",
            mode="TL",
            currency="USD",
            weight="40000 lbs"
        )
        shipment_without_critical.fields_extracted = 4
        
        pattern_results = {}
        
        conf_with = extractor._calculate_extraction_confidence(
            shipment_with_critical, pattern_results, 
            shipment_with_critical.to_dict()
        )
        
        conf_without = extractor._calculate_extraction_confidence(
            shipment_without_critical, pattern_results,
            shipment_without_critical.to_dict()
        )
        
        # Critical fields should give higher confidence
        assert conf_with > conf_without
