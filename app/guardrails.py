"""
Guardrails Module for Ultra Doc-Intelligence.
Implements multiple layers of protection against hallucination and low-confidence answers.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class GuardrailType(Enum):
    """Types of guardrails in the system."""
    RETRIEVAL_THRESHOLD = "retrieval_threshold"
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    CONTEXT_COVERAGE = "context_coverage"
    ANSWER_GROUNDING = "answer_grounding"
    REFUSAL_DETECTION = "refusal_detection"
    NUMERIC_VALIDATION = "numeric_validation"


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""
    guardrail_type: GuardrailType
    passed: bool
    score: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailsReport:
    """Comprehensive report of all guardrail checks."""
    overall_passed: bool
    checks: List[GuardrailResult]
    triggered_guardrails: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_passed": self.overall_passed,
            "checks": [
                {
                    "type": check.guardrail_type.value,
                    "passed": check.passed,
                    "score": round(check.score, 4),
                    "message": check.message
                }
                for check in self.checks
            ],
            "triggered_guardrails": self.triggered_guardrails,
            "recommendation": self.recommendation
        }


class Guardrails:
    """
    Multi-layered guardrails system to ensure answer quality and prevent hallucination.
    
    Implements the following guardrails:
    1. Retrieval Threshold: Ensures retrieved chunks meet minimum similarity
    2. Confidence Threshold: Requires minimum confidence score to answer
    3. Context Coverage: Validates answer is grounded in retrieved context
    4. Answer Grounding: Checks if answer contains information from sources
    5. Refusal Detection: Detects when model appropriately refuses to answer
    6. Numeric Validation: Validates numeric values appear in source
    """
    
    def __init__(
        self,
        retrieval_threshold: float = 0.3,
        confidence_threshold: float = 0.4,
        high_confidence_threshold: float = 0.75,
        min_context_coverage: float = 0.3
    ):
        self.retrieval_threshold = retrieval_threshold
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.min_context_coverage = min_context_coverage
        
        # Patterns indicating the model refused to answer (good behavior)
        self.refusal_patterns = [
            r"not found in (?:the )?document",
            r"document does not (?:contain|mention|include)",
            r"(?:no|cannot find) (?:information|data|details)",
            r"unable to (?:find|locate|determine)",
            r"not (?:mentioned|specified|stated|provided)",
            r"information (?:is )?(?:not available|missing|unclear)",
            r"i (?:could not|couldn't|cannot|can't) find"
        ]
    
    def run_all_checks(
        self,
        question: str,
        answer: str,
        confidence_score: float,
        retrieved_chunks: List[Any],
        context: str
    ) -> GuardrailsReport:
        """
        Run all guardrail checks on a question-answer pair.
        
        Args:
            question: User's question
            answer: Generated answer
            confidence_score: Overall confidence score
            retrieved_chunks: List of retrieved chunks with similarity scores
            context: Combined context string
            
        Returns:
            GuardrailsReport with all check results
        """
        checks = []
        triggered = []
        
        # 1. Retrieval Threshold Check
        retrieval_result = self._check_retrieval_threshold(retrieved_chunks)
        checks.append(retrieval_result)
        if not retrieval_result.passed:
            triggered.append(retrieval_result.guardrail_type.value)
        
        # 2. Confidence Threshold Check
        confidence_result = self._check_confidence_threshold(confidence_score)
        checks.append(confidence_result)
        if not confidence_result.passed:
            triggered.append(confidence_result.guardrail_type.value)
        
        # 3. Refusal Detection (this is actually good behavior)
        refusal_result = self._detect_appropriate_refusal(answer)
        checks.append(refusal_result)
        # Refusal doesn't trigger guardrail - it's the guardrail working correctly
        
        # 4. Context Coverage Check
        coverage_result = self._check_context_coverage(answer, context)
        checks.append(coverage_result)
        if not coverage_result.passed and not refusal_result.passed:
            triggered.append(coverage_result.guardrail_type.value)
        
        # 5. Answer Grounding Check
        grounding_result = self._check_answer_grounding(answer, context, retrieved_chunks)
        checks.append(grounding_result)
        if not grounding_result.passed and not refusal_result.passed:
            triggered.append(grounding_result.guardrail_type.value)
        
        # 6. Numeric Validation
        numeric_result = self._validate_numerics(answer, context)
        checks.append(numeric_result)
        if not numeric_result.passed:
            triggered.append(numeric_result.guardrail_type.value)
        
        # Determine overall pass/fail
        # Pass if: high confidence OR (moderate confidence AND good grounding)
        # OR appropriate refusal detected
        overall_passed = (
            refusal_result.passed or  # Appropriate refusal is passing
            (confidence_score >= self.high_confidence_threshold) or
            (confidence_score >= self.confidence_threshold and 
             grounding_result.passed and 
             coverage_result.passed)
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            overall_passed, triggered, confidence_score, refusal_result.passed
        )
        
        return GuardrailsReport(
            overall_passed=overall_passed,
            checks=checks,
            triggered_guardrails=triggered,
            recommendation=recommendation
        )
    
    def _check_retrieval_threshold(self, retrieved_chunks: List[Any]) -> GuardrailResult:
        """Check if retrieved chunks meet minimum similarity threshold."""
        if not retrieved_chunks:
            return GuardrailResult(
                guardrail_type=GuardrailType.RETRIEVAL_THRESHOLD,
                passed=False,
                score=0.0,
                message="No chunks were retrieved for this query",
                details={"num_chunks": 0}
            )
        
        # Get similarity scores
        scores = []
        for chunk in retrieved_chunks:
            if hasattr(chunk, 'similarity_score'):
                scores.append(chunk.similarity_score)
        
        if not scores:
            return GuardrailResult(
                guardrail_type=GuardrailType.RETRIEVAL_THRESHOLD,
                passed=True,
                score=0.5,
                message="Could not determine similarity scores",
                details={"num_chunks": len(retrieved_chunks)}
            )
        
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        above_threshold = sum(1 for s in scores if s >= self.retrieval_threshold)
        
        passed = max_score >= self.retrieval_threshold
        
        return GuardrailResult(
            guardrail_type=GuardrailType.RETRIEVAL_THRESHOLD,
            passed=passed,
            score=max_score,
            message=f"Best retrieval similarity: {max_score:.2%}" if passed else 
                    f"No chunks above threshold ({self.retrieval_threshold:.0%}). Best: {max_score:.2%}",
            details={
                "max_similarity": max_score,
                "avg_similarity": avg_score,
                "chunks_above_threshold": above_threshold,
                "total_chunks": len(scores)
            }
        )
    
    def _check_confidence_threshold(self, confidence_score: float) -> GuardrailResult:
        """Check if confidence score meets minimum threshold."""
        passed = confidence_score >= self.confidence_threshold
        
        if confidence_score >= self.high_confidence_threshold:
            message = f"High confidence: {confidence_score:.1%}"
            level = "high"
        elif passed:
            message = f"Moderate confidence: {confidence_score:.1%}"
            level = "moderate"
        else:
            message = f"Low confidence: {confidence_score:.1%} (threshold: {self.confidence_threshold:.0%})"
            level = "low"
        
        return GuardrailResult(
            guardrail_type=GuardrailType.CONFIDENCE_THRESHOLD,
            passed=passed,
            score=confidence_score,
            message=message,
            details={"confidence_level": level, "threshold": self.confidence_threshold}
        )
    
    def _detect_appropriate_refusal(self, answer: str) -> GuardrailResult:
        """Detect if the model appropriately refused to answer."""
        answer_lower = answer.lower()
        
        for pattern in self.refusal_patterns:
            if re.search(pattern, answer_lower):
                return GuardrailResult(
                    guardrail_type=GuardrailType.REFUSAL_DETECTION,
                    passed=True,  # This is good behavior!
                    score=1.0,
                    message="Model appropriately indicated information not found",
                    details={"refusal_detected": True, "pattern_matched": pattern}
                )
        
        return GuardrailResult(
            guardrail_type=GuardrailType.REFUSAL_DETECTION,
            passed=False,  # Not a refusal - needs other validation
            score=0.0,
            message="Model provided a direct answer (requires validation)",
            details={"refusal_detected": False}
        )
    
    def _check_context_coverage(self, answer: str, context: str) -> GuardrailResult:
        """Check if answer content appears to be covered by context."""
        if not answer or not context:
            return GuardrailResult(
                guardrail_type=GuardrailType.CONTEXT_COVERAGE,
                passed=False,
                score=0.0,
                message="Missing answer or context",
                details={}
            )
        
        # Extract significant terms from answer
        answer_terms = self._extract_significant_terms(answer)
        
        if not answer_terms:
            return GuardrailResult(
                guardrail_type=GuardrailType.CONTEXT_COVERAGE,
                passed=True,
                score=0.7,
                message="Answer has no significant terms to validate",
                details={"answer_terms": 0}
            )
        
        context_lower = context.lower()
        
        # Check how many answer terms appear in context
        covered_terms = [term for term in answer_terms if term in context_lower]
        coverage = len(covered_terms) / len(answer_terms)
        
        passed = coverage >= self.min_context_coverage
        
        return GuardrailResult(
            guardrail_type=GuardrailType.CONTEXT_COVERAGE,
            passed=passed,
            score=coverage,
            message=f"Context coverage: {coverage:.1%} of answer terms found in context",
            details={
                "total_terms": len(answer_terms),
                "covered_terms": len(covered_terms),
                "sample_covered": covered_terms[:5],
                "sample_missing": [t for t in answer_terms if t not in context_lower][:5]
            }
        )
    
    def _check_answer_grounding(
        self,
        answer: str,
        context: str,
        retrieved_chunks: List[Any]
    ) -> GuardrailResult:
        """
        Check if answer is well-grounded in the retrieved chunks.
        Uses n-gram overlap to detect if answer phrases appear in sources.
        """
        if not answer or not context:
            return GuardrailResult(
                guardrail_type=GuardrailType.ANSWER_GROUNDING,
                passed=False,
                score=0.0,
                message="Missing answer or context",
                details={}
            )
        
        # Generate n-grams from answer (2-grams and 3-grams)
        answer_ngrams = self._extract_ngrams(answer, [2, 3])
        
        if not answer_ngrams:
            return GuardrailResult(
                guardrail_type=GuardrailType.ANSWER_GROUNDING,
                passed=True,
                score=0.8,
                message="Answer too short for n-gram analysis",
                details={}
            )
        
        context_lower = context.lower()
        
        # Check n-gram overlap
        grounded_ngrams = [ng for ng in answer_ngrams if ng in context_lower]
        grounding_score = len(grounded_ngrams) / len(answer_ngrams)
        
        passed = grounding_score >= 0.2  # At least 20% of phrases should be grounded
        
        return GuardrailResult(
            guardrail_type=GuardrailType.ANSWER_GROUNDING,
            passed=passed,
            score=grounding_score,
            message=f"Answer grounding: {grounding_score:.1%} of phrases found in context",
            details={
                "total_ngrams": len(answer_ngrams),
                "grounded_ngrams": len(grounded_ngrams),
                "sample_grounded": grounded_ngrams[:5]
            }
        )
    
    def _validate_numerics(self, answer: str, context: str) -> GuardrailResult:
        """
        Validate that numeric values in the answer appear in the context.
        Critical for logistics where numbers (rates, weights, dates) must be accurate.
        """
        # Extract numbers from answer
        answer_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', answer)
        
        if not answer_numbers:
            return GuardrailResult(
                guardrail_type=GuardrailType.NUMERIC_VALIDATION,
                passed=True,
                score=1.0,
                message="No numeric values in answer to validate",
                details={"numbers_found": 0}
            )
        
        # Check if each number appears in context
        context_numbers = set(re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', context))
        
        validated = []
        unvalidated = []
        
        for num in answer_numbers:
            # Also check without commas
            num_normalized = num.replace(",", "")
            if num in context_numbers or num_normalized in context_numbers:
                validated.append(num)
            else:
                # Check if it's a close match (could be formatting difference)
                try:
                    num_float = float(num_normalized)
                    found_close = False
                    for ctx_num in context_numbers:
                        try:
                            ctx_float = float(ctx_num.replace(",", ""))
                            if abs(num_float - ctx_float) < 0.01 * max(num_float, ctx_float):
                                validated.append(num)
                                found_close = True
                                break
                        except ValueError:
                            continue
                    if not found_close:
                        unvalidated.append(num)
                except ValueError:
                    unvalidated.append(num)
        
        validation_rate = len(validated) / len(answer_numbers) if answer_numbers else 1.0
        passed = validation_rate >= 0.8  # 80% of numbers should be validated
        
        return GuardrailResult(
            guardrail_type=GuardrailType.NUMERIC_VALIDATION,
            passed=passed,
            score=validation_rate,
            message=f"Numeric validation: {len(validated)}/{len(answer_numbers)} numbers verified in context",
            details={
                "validated_numbers": validated,
                "unvalidated_numbers": unvalidated,
                "validation_rate": validation_rate
            }
        )
    
    def _extract_significant_terms(self, text: str) -> List[str]:
        """Extract significant terms from text, excluding common words."""
        # Common words to exclude
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'this', 'that', 'these', 'those',
            'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your',
            'i', 'me', 'my', 'he', 'she', 'him', 'her', 'his', 'not', 'no', 'yes',
            'if', 'then', 'else', 'when', 'where', 'what', 'which', 'who', 'how',
            'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
            'such', 'any', 'only', 'same', 'so', 'than', 'too', 'very', 'just',
            'can', 'as', 'also', 'into', 'about'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter and return significant terms
        return [w for w in words if w not in stopwords]
    
    def _extract_ngrams(self, text: str, ns: List[int]) -> List[str]:
        """Extract n-grams from text."""
        words = text.lower().split()
        ngrams = []
        
        for n in ns:
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i+n])
                # Skip if mostly stopwords
                if len([w for w in words[i:i+n] if len(w) > 3]) >= n // 2:
                    ngrams.append(ngram)
        
        return ngrams
    
    def _generate_recommendation(
        self,
        overall_passed: bool,
        triggered: List[str],
        confidence: float,
        is_refusal: bool
    ) -> str:
        """Generate a user-friendly recommendation based on check results."""
        
        if is_refusal:
            return "The system correctly identified that this information is not available in the document."
        
        if overall_passed and confidence >= self.high_confidence_threshold:
            return "High confidence answer based on strong document evidence."
        
        if overall_passed:
            return "Answer provided with moderate confidence. Verify critical details if needed."
        
        if GuardrailType.RETRIEVAL_THRESHOLD.value in triggered:
            return "The question may not be well-covered by this document. Consider rephrasing or checking if the document contains this information."
        
        if GuardrailType.CONFIDENCE_THRESHOLD.value in triggered:
            return "Low confidence in this answer. The document may not clearly address this question."
        
        if GuardrailType.NUMERIC_VALIDATION.value in triggered:
            return "Some numeric values could not be verified against the document. Please verify these figures."
        
        return "Answer quality checks did not pass. Please verify this information manually."


class AnswerValidator:
    """
    High-level validator that combines guardrails with answer processing.
    Provides a simple interface for validating answers.
    """
    
    def __init__(self, guardrails: Optional[Guardrails] = None):
        self.guardrails = guardrails or Guardrails()
    
    def validate_and_process(
        self,
        question: str,
        answer: str,
        confidence: float,
        retrieved_chunks: List[Any],
        context: str,
        min_confidence: float = 0.4
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """
        Validate an answer and potentially modify it based on guardrails.
        
        Returns:
            Tuple of (final_answer, final_confidence, is_valid, guardrail_report)
        """
        # Run guardrail checks
        report = self.guardrails.run_all_checks(
            question, answer, confidence, retrieved_chunks, context
        )
        
        # If guardrails triggered, modify the answer
        if not report.overall_passed:
            if confidence < min_confidence:
                final_answer = (
                    f"I found some potentially relevant information, but my confidence "
                    f"is too low ({confidence:.1%}) to provide a reliable answer. "
                    f"{report.recommendation}"
                )
            else:
                final_answer = f"{answer}\n\n⚠️ Note: {report.recommendation}"
        else:
            final_answer = answer
        
        return (
            final_answer,
            confidence,
            report.overall_passed,
            report.to_dict()
        )
