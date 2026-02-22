"""
Unit tests for the guardrails module.
"""

import pytest
from app.guardrails import (
    Guardrails,
    GuardrailType,
    GuardrailResult,
    GuardrailsReport,
    AnswerValidator
)
from dataclasses import dataclass


@dataclass
class MockRetrievedChunk:
    """Mock chunk for testing."""
    similarity_score: float
    content: str = "Test content"


class TestGuardrails:
    """Tests for Guardrails class."""
    
    @pytest.fixture
    def guardrails(self):
        return Guardrails(
            retrieval_threshold=0.3,
            confidence_threshold=0.4,
            high_confidence_threshold=0.75,
            min_context_coverage=0.3
        )
    
    def test_retrieval_threshold_pass(self, guardrails):
        chunks = [
            MockRetrievedChunk(similarity_score=0.85),
            MockRetrievedChunk(similarity_score=0.72),
            MockRetrievedChunk(similarity_score=0.45),
        ]
        
        result = guardrails._check_retrieval_threshold(chunks)
        
        assert result.passed is True
        assert result.guardrail_type == GuardrailType.RETRIEVAL_THRESHOLD
        assert result.score >= 0.3
    
    def test_retrieval_threshold_fail(self, guardrails):
        chunks = [
            MockRetrievedChunk(similarity_score=0.15),
            MockRetrievedChunk(similarity_score=0.20),
        ]
        
        result = guardrails._check_retrieval_threshold(chunks)
        
        assert result.passed is False
        assert "below threshold" in result.message.lower() or "best" in result.message.lower()
    
    def test_retrieval_threshold_empty_chunks(self, guardrails):
        result = guardrails._check_retrieval_threshold([])
        
        assert result.passed is False
        assert "no chunks" in result.message.lower()
    
    def test_confidence_threshold_high(self, guardrails):
        result = guardrails._check_confidence_threshold(0.85)
        
        assert result.passed is True
        assert "high" in result.message.lower()
    
    def test_confidence_threshold_medium(self, guardrails):
        result = guardrails._check_confidence_threshold(0.55)
        
        assert result.passed is True
        assert "moderate" in result.message.lower()
    
    def test_confidence_threshold_low(self, guardrails):
        result = guardrails._check_confidence_threshold(0.25)
        
        assert result.passed is False
        assert "low" in result.message.lower()
    
    def test_refusal_detection_positive(self, guardrails):
        answer = "I could not find information about the delivery date in the document."
        
        result = guardrails._detect_appropriate_refusal(answer)
        
        assert result.passed is True  # Refusal is good behavior
        assert result.details.get("refusal_detected") is True
    
    def test_refusal_detection_negative(self, guardrails):
        answer = "The carrier rate is $2,500.00 USD."
        
        result = guardrails._detect_appropriate_refusal(answer)
        
        assert result.passed is False  # Not a refusal, needs validation
        assert result.details.get("refusal_detected") is False
    
    def test_refusal_patterns(self, guardrails):
        refusal_answers = [
            "Not found in document",
            "The document does not contain this information",
            "I cannot find any information about the weight",
            "This information is not available in the document",
            "Unable to determine the pickup date from the provided context"
        ]
        
        for answer in refusal_answers:
            result = guardrails._detect_appropriate_refusal(answer)
            assert result.passed is True, f"Failed to detect refusal: {answer}"
    
    def test_context_coverage_high(self, guardrails):
        answer = "The carrier rate is two thousand five hundred dollars."
        context = "Rate: $2,500.00 USD. Total carrier rate is two thousand five hundred dollars for the shipment."
        
        result = guardrails._check_context_coverage(answer, context)
        
        assert result.passed is True
        assert result.score > 0.3
    
    def test_context_coverage_low(self, guardrails):
        answer = "The delivery will be made by helicopter to the rooftop."
        context = "Shipment via dry van truck to warehouse loading dock."
        
        result = guardrails._check_context_coverage(answer, context)
        
        # Score should be lower due to mismatched content
        assert result.score < 0.5
    
    def test_numeric_validation_pass(self, guardrails):
        answer = "The rate is $2,500.00 and the weight is 42,000 lbs."
        context = "Total Rate: $2,500.00 USD. Gross Weight: 42,000 lbs. Equipment: 53' Van"
        
        result = guardrails._validate_numerics(answer, context)
        
        assert result.passed is True
        assert len(result.details.get("validated_numbers", [])) >= 2
    
    def test_numeric_validation_fail(self, guardrails):
        answer = "The rate is $5,000.00 for this shipment."
        context = "Total Rate: $2,500.00 USD. Line haul: $2,500.00"
        
        result = guardrails._validate_numerics(answer, context)
        
        assert result.passed is False
        assert "5,000" in result.details.get("unvalidated_numbers", []) or "5000" in str(result.details)
    
    def test_numeric_validation_no_numbers(self, guardrails):
        answer = "The shipment is scheduled for next Monday."
        context = "Pickup scheduled for Monday morning."
        
        result = guardrails._validate_numerics(answer, context)
        
        assert result.passed is True  # No numbers to validate
    
    def test_run_all_checks_high_confidence(self, guardrails):
        chunks = [
            MockRetrievedChunk(similarity_score=0.90, content="Rate: $2,500.00"),
            MockRetrievedChunk(similarity_score=0.85, content="Total: $2,500.00"),
        ]
        
        report = guardrails.run_all_checks(
            question="What is the rate?",
            answer="The rate is $2,500.00",
            confidence_score=0.85,
            retrieved_chunks=chunks,
            context="Rate: $2,500.00. Total: $2,500.00 USD."
        )
        
        assert report.overall_passed is True
        assert len(report.triggered_guardrails) == 0
        assert "high confidence" in report.recommendation.lower() or "strong" in report.recommendation.lower()
    
    def test_run_all_checks_low_confidence(self, guardrails):
        chunks = [
            MockRetrievedChunk(similarity_score=0.25, content="Some unrelated text"),
        ]
        
        report = guardrails.run_all_checks(
            question="What is the insurance coverage?",
            answer="The insurance coverage is $100,000",
            confidence_score=0.25,
            retrieved_chunks=chunks,
            context="Some unrelated shipping details"
        )
        
        assert report.overall_passed is False
        assert len(report.triggered_guardrails) > 0
    
    def test_run_all_checks_appropriate_refusal(self, guardrails):
        chunks = [
            MockRetrievedChunk(similarity_score=0.35, content="Rate: $2,500"),
        ]
        
        report = guardrails.run_all_checks(
            question="What is the insurance amount?",
            answer="I could not find information about insurance in the document.",
            confidence_score=0.35,
            retrieved_chunks=chunks,
            context="Rate: $2,500. Weight: 40,000 lbs."
        )
        
        assert report.overall_passed is True  # Appropriate refusal passes
        assert "correctly identified" in report.recommendation.lower()


class TestGuardrailResult:
    """Tests for GuardrailResult dataclass."""
    
    def test_guardrail_result_creation(self):
        result = GuardrailResult(
            guardrail_type=GuardrailType.CONFIDENCE_THRESHOLD,
            passed=True,
            score=0.85,
            message="High confidence",
            details={"level": "high"}
        )
        
        assert result.guardrail_type == GuardrailType.CONFIDENCE_THRESHOLD
        assert result.passed is True
        assert result.score == 0.85


class TestGuardrailsReport:
    """Tests for GuardrailsReport dataclass."""
    
    def test_report_to_dict(self):
        checks = [
            GuardrailResult(
                guardrail_type=GuardrailType.CONFIDENCE_THRESHOLD,
                passed=True,
                score=0.85,
                message="High confidence"
            )
        ]
        
        report = GuardrailsReport(
            overall_passed=True,
            checks=checks,
            triggered_guardrails=[],
            recommendation="Answer is reliable"
        )
        
        result = report.to_dict()
        
        assert result["overall_passed"] is True
        assert len(result["checks"]) == 1
        assert result["checks"][0]["type"] == "confidence_threshold"


class TestAnswerValidator:
    """Tests for AnswerValidator class."""
    
    def test_validate_and_process_valid_answer(self):
        validator = AnswerValidator()
        chunks = [
            MockRetrievedChunk(similarity_score=0.85, content="Rate: $2,500"),
        ]
        
        answer, confidence, is_valid, report = validator.validate_and_process(
            question="What is the rate?",
            answer="The rate is $2,500",
            confidence=0.80,
            retrieved_chunks=chunks,
            context="Rate: $2,500 USD",
            min_confidence=0.4
        )
        
        assert is_valid is True
        assert "rate" in answer.lower()
        assert confidence == 0.80
    
    def test_validate_and_process_low_confidence(self):
        validator = AnswerValidator()
        chunks = [
            MockRetrievedChunk(similarity_score=0.25, content="Unrelated text"),
        ]
        
        answer, confidence, is_valid, report = validator.validate_and_process(
            question="What is the insurance?",
            answer="The insurance is $50,000",
            confidence=0.25,
            retrieved_chunks=chunks,
            context="Unrelated shipping text",
            min_confidence=0.4
        )
        
        assert is_valid is False
        # Answer should be modified to include warning
        assert "confidence" in answer.lower() or "low" in answer.lower()
