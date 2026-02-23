"""
Streamlit UI for Ultra Doc-Intelligence.
Provides a lightweight interface for document upload, Q&A, and structured extraction.
"""

import streamlit as st
import requests
import json
import os
from typing import Optional, Dict, Any
import time

# ============== Configuration ==============

# API endpoint - configurable via secrets, environment, or default
def get_api_base_url():
    """Get API base URL from secrets, environment, or default."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'API_BASE_URL' in st.secrets:
            return st.secrets['API_BASE_URL']
    except Exception:
        pass
    
    # Try environment variable
    env_url = os.environ.get('API_BASE_URL')
    if env_url:
        return env_url
    
    # Default for local development
    return "http://localhost:8000"

API_BASE_URL = get_api_base_url()

# ============== Demo Mode Sample Data ==============

DEMO_EXTRACTION_DATA = {
    "shipment_id": "LD53657",
    "shipper": "ABC Manufacturing Co.",
    "consignee": "XYZ Distribution Center",
    "pickup_datetime": "2026-02-24T08:00:00",
    "delivery_datetime": "2026-02-26T14:00:00",
    "equipment_type": "53' Dry Van",
    "mode": "FTL (Full Truckload)",
    "rate": "$2,450.00",
    "currency": "USD",
    "weight": "42,000 lbs",
    "carrier_name": "FastFreight Logistics LLC"
}

DEMO_QA_RESPONSES = {
    "default": {
        "answer": "Based on the document analysis, the shipment LD53657 is a Full Truckload (FTL) movement from ABC Manufacturing Co. to XYZ Distribution Center. The carrier rate is $2,450.00 USD for a 53' Dry Van equipment type. Pickup is scheduled for February 24, 2026 at 08:00 AM with delivery expected by February 26, 2026 at 2:00 PM.",
        "confidence_score": 0.92,
        "guardrail_triggered": False,
        "sources": [
            {"rank": 1, "similarity_score": 0.95, "page_number": 1, "content": "RATE CONFIRMATION\nLoad #: LD53657\nCarrier: FastFreight Logistics LLC\nRate: $2,450.00 USD"},
            {"rank": 2, "similarity_score": 0.88, "page_number": 1, "content": "Shipper: ABC Manufacturing Co.\nConsignee: XYZ Distribution Center\nEquipment: 53' Dry Van"}
        ]
    },
    "rate": {
        "answer": "The carrier rate for this shipment is $2,450.00 USD. This is a flat rate for Full Truckload (FTL) service using a 53' Dry Van.",
        "confidence_score": 0.96,
        "guardrail_triggered": False,
        "sources": [
            {"rank": 1, "similarity_score": 0.98, "page_number": 1, "content": "RATE: $2,450.00 USD\nPayment Terms: Net 30 Days"}
        ]
    },
    "pickup": {
        "answer": "Pickup is scheduled for February 24, 2026 at 08:00 AM at ABC Manufacturing Co. facility.",
        "confidence_score": 0.94,
        "guardrail_triggered": False,
        "sources": [
            {"rank": 1, "similarity_score": 0.96, "page_number": 1, "content": "PICKUP: 02/24/2026 @ 08:00 AM\nLocation: ABC Manufacturing Co."}
        ]
    },
    "delivery": {
        "answer": "Delivery is expected on February 26, 2026 at 2:00 PM at XYZ Distribution Center.",
        "confidence_score": 0.93,
        "guardrail_triggered": False,
        "sources": [
            {"rank": 1, "similarity_score": 0.95, "page_number": 1, "content": "DELIVERY: 02/26/2026 @ 2:00 PM\nLocation: XYZ Distribution Center"}
        ]
    }
}

DEMO_GRAPH_RESPONSE = {
    "answer": "The knowledge graph shows that FastFreight Logistics LLC (Carrier) is transporting shipment LD53657 from ABC Manufacturing Co. (Shipper) to XYZ Distribution Center (Consignee). The shipment uses a 53' Dry Van and follows an FTL (Full Truckload) mode. Key relationships: Carrier TRANSPORTS Shipment, Shipper SHIPS_TO Consignee, Shipment USES Equipment Type.",
    "confidence": 0.89,
    "entities_found": 6,
    "relationships_found": 5,
    "reasoning": "1. Identified carrier entity: FastFreight Logistics LLC\n2. Found shipper-consignee relationship: ABC Manufacturing -> XYZ Distribution\n3. Mapped equipment and mode attributes\n4. Connected all entities through shipment ID LD53657"
}

def get_demo_qa_response(question: str) -> dict:
    """Get appropriate demo response based on question keywords."""
    question_lower = question.lower()
    if any(word in question_lower for word in ["rate", "cost", "price", "charge"]):
        return DEMO_QA_RESPONSES["rate"]
    elif any(word in question_lower for word in ["pickup", "pick up", "origin"]):
        return DEMO_QA_RESPONSES["pickup"]
    elif any(word in question_lower for word in ["delivery", "deliver", "destination"]):
        return DEMO_QA_RESPONSES["delivery"]
    else:
        return DEMO_QA_RESPONSES["default"]

# Page configuration
st.set_page_config(
    page_title="Ultra Doc-Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== Custom CSS ==============

st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    
    /* Confidence score styling */
    .confidence-high {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    .confidence-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    .confidence-low {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: 600;
    }
    
    /* Source box styling */
    .source-box {
        background-color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
        color: #1F2937 !important;
    }
    
    .source-box strong {
        color: #1E40AF !important;
    }
    
    /* Answer box styling */
    .answer-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        color: #1F2937 !important;
    }
    
    .answer-box strong {
        color: #1E40AF !important;
    }
    
    /* Extraction result styling */
    .extraction-field {
        padding: 0.5rem;
        border-bottom: 1px solid #E5E7EB;
        background-color: #FFFFFF;
    }
    
    .field-name {
        font-weight: 600;
        color: #374151 !important;
    }
    
    .field-value {
        color: #1F2937 !important;
    }
    
    .field-null {
        color: #9CA3AF !important;
        font-style: italic;
    }
    
    /* Guardrail warning */
    .guardrail-warning {
        background-color: #FEF3C7;
        border: 1px solid #F59E0B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #92400E !important;
    }
    
    .guardrail-warning strong {
        color: #B45309 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============== Helper Functions ==============

def get_confidence_class(score: float) -> str:
    """Get CSS class based on confidence score."""
    if score >= 0.75:
        return "confidence-high"
    elif score >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"


def get_confidence_label(score: float) -> str:
    """Get label based on confidence score."""
    if score >= 0.75:
        return "High Confidence"
    elif score >= 0.4:
        return "Medium Confidence"
    else:
        return "Low Confidence"


def check_api_health() -> bool:
    """Check if the API is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def poll_upload_status(task_id: str, progress_bar, status_text) -> Optional[Dict[str, Any]]:
    """
    Poll upload task status until completion.
    Updates progress bar and status text in real-time.
    """
    import time
    
    max_attempts = 600  # 10 minutes max at 1 second intervals
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE_URL}/upload/status/{task_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                progress = data.get("progress", 0)
                message = data.get("message", "Processing...")
                
                # Update progress bar
                progress_bar.progress(progress / 100, text=message)
                status_text.text(f"Status: {status.title()} | {message}")
                
                if status == "completed":
                    return data
                elif status == "failed":
                    st.error(f"Processing failed: {data.get('error', 'Unknown error')}")
                    return None
                
                # Still processing - wait and poll again
                time.sleep(1)
            else:
                st.error(f"Status check failed: {response.status_code}")
                return None
                
        except requests.exceptions.RequestException as e:
            # Connection error - wait and retry
            time.sleep(2)
            continue
    
    st.error("Processing timed out after 10 minutes")
    return None


def upload_document(file) -> Optional[Dict[str, Any]]:
    """Upload a document to the API with progress tracking."""
    try:
        # Start upload
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            task_id = data.get("task_id")
            
            if task_id:
                # Show progress UI
                st.info(f"📄 Processing **{file.name}**...")
                progress_bar = st.progress(0, text="Starting processing...")
                status_text = st.empty()
                
                # Poll for completion
                result = poll_upload_status(task_id, progress_bar, status_text)
                
                if result and result.get("status") == "completed":
                    # Clear progress UI and return result
                    progress_bar.empty()
                    status_text.empty()
                    return result.get("result")
                return None
            else:
                # Synchronous response (legacy/fallback)
                return data
        else:
            st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server. Please ensure the backend is running.")
        return None
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return None


def ask_question(document_id: str, question: str, min_confidence: float = 0.4, use_reasoning: bool = True) -> Optional[Dict[str, Any]]:
    """Ask a question about a document."""
    try:
        payload = {
            "document_id": document_id,
            "question": question,
            "min_confidence": min_confidence,
            "use_reasoning": use_reasoning
        }
        response = requests.post(f"{API_BASE_URL}/ask", json=payload, timeout=120)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Question failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server.")
        return None
    except Exception as e:
        st.error(f"Question error: {str(e)}")
        return None


def extract_data(document_id: str) -> Optional[Dict[str, Any]]:
    """Extract structured data from a document."""
    try:
        payload = {"document_id": document_id, "use_llm": True}
        response = requests.post(f"{API_BASE_URL}/extract", json=payload, timeout=60)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Extraction failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API server.")
        return None
    except Exception as e:
        st.error(f"Extraction error: {str(e)}")
        return None


def get_documents() -> list:
    """Get list of uploaded documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except:
        return []


# ============== Session State Initialization ==============

if "document_id" not in st.session_state:
    st.session_state.document_id = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "last_processed_file" not in st.session_state:
    st.session_state.last_processed_file = None
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False


# ============== Sidebar ==============

with st.sidebar:
    st.markdown("### 📄 Ultra Doc-Intelligence")
    st.markdown("AI-powered logistics document assistant")
    
    st.divider()
    
    # API Status & Demo Mode
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
        st.session_state.demo_mode = False
    else:
        st.warning("🎭 Demo Mode Active")
        st.session_state.demo_mode = True
        st.caption("API not connected. Showing sample responses.")
    
    st.divider()
    
    # Settings - moved up for visibility
    st.markdown("### ⚙️ Settings")
    
    # Model Selection
    use_reasoning = st.toggle(
        "🧠 Reasoning Mode",
        value=True,
        help="ON: Uses grok-4-1-fast-reasoning for deeper analysis (slower)\nOFF: Uses grok-4-1-fast for quick responses"
    )
    
    if use_reasoning:
        st.caption("🔍 **Reasoning**: Deeper analysis, step-by-step thinking")
    else:
        st.caption("⚡ **Fast**: Quick responses, less detailed")
    
    min_confidence = st.slider(
        "Min Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Answers below this confidence will trigger guardrails"
    )
    
    st.divider()
    
    # Document Upload Section
    st.markdown("### 📤 Upload Document")
    
    uploaded_file = st.file_uploader(
        "Choose a logistics document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
        key="file_uploader"
    )
    
    # Auto-process on upload (no button needed)
    if uploaded_file is not None:
        # Check if this is a new file (not already processed)
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_processed_file") != file_key:
            if st.session_state.demo_mode:
                # Demo mode: simulate successful upload
                import hashlib
                demo_doc_id = hashlib.md5(uploaded_file.name.encode()).hexdigest()
                st.session_state.document_id = demo_doc_id
                st.session_state.document_name = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.extraction_result = None
                st.session_state.last_processed_file = file_key
                st.success(f"✅ Demo: Loaded {uploaded_file.name}")
                st.info("📊 Demo: 8 chunks, 2 pages (simulated)")
                st.rerun()
            else:
                with st.spinner("🔄 Auto-indexing document..."):
                    result = upload_document(uploaded_file)
                    if result and result.get("success"):
                        st.session_state.document_id = result["document_id"]
                        st.session_state.document_name = result["filename"]
                        st.session_state.chat_history = []
                        st.session_state.extraction_result = None
                        st.session_state.last_processed_file = file_key
                        st.success(f"✅ Auto-indexed: {result['filename']}")
                        st.info(f"📊 {result['stats']['chunk_count']} chunks, {result['stats']['page_count']} pages")
                        st.rerun()
        else:
            st.success(f"✅ Document ready: {uploaded_file.name}")
    
    st.divider()
    
    # Active Document Info
    if st.session_state.document_id:
        st.markdown("### 📑 Active Document")
        st.info(f"**{st.session_state.document_name}**\n\nID: `{st.session_state.document_id[:12]}...`")
        
        if st.button("🗑️ Clear Document", use_container_width=True):
            st.session_state.document_id = None
            st.session_state.document_name = None
            st.session_state.chat_history = []
            st.session_state.extraction_result = None
            st.session_state.last_processed_file = None
            st.rerun()


# ============== Main Content ==============

# Header
st.markdown('<p class="main-header">🔍 Ultra Doc-Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered logistics document analysis with RAG and guardrails</p>', unsafe_allow_html=True)

# Demo Mode Banner
if st.session_state.demo_mode:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 1rem; 
                border-radius: 0.5rem; 
                margin-bottom: 1rem;
                text-align: center;">
        <strong>🎭 DEMO MODE</strong> - Backend API not connected. Showing sample responses to demonstrate functionality.
        <br><small>Deploy the FastAPI backend to enable live document processing.</small>
    </div>
    """, unsafe_allow_html=True)

# Check for active document
if not st.session_state.document_id:
    st.info("👈 Please upload a logistics document using the sidebar to get started.")
    
    # Show example questions
    st.markdown("### 💡 Example Questions You Can Ask")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - What is the carrier rate?
        - When is the pickup scheduled?
        - What is the shipment ID?
        - What equipment type is required?
        """)
    
    with col2:
        st.markdown("""
        - Who is the consignee?
        - What is the delivery address?
        - What is the total weight?
        - What is the shipping mode?
        """)
    
    st.stop()

# Create tabs for different functionalities
tab_qa, tab_extract, tab_graph, tab_history = st.tabs(["💬 Ask Questions", "📋 Extract Data", "🕸️ GraphRAG", "📜 History"])


# ============== Q&A Tab ==============

with tab_qa:
    st.markdown("### Ask Questions About Your Document")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the carrier rate?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button and question:
        if st.session_state.demo_mode:
            # Demo mode: use sample responses
            result = get_demo_qa_response(question)
            st.session_state.chat_history.append({
                "question": question,
                "result": result
            })
        else:
            with st.spinner("Analyzing document..." + (" (reasoning mode)" if use_reasoning else "")):
                result = ask_question(st.session_state.document_id, question, min_confidence, use_reasoning)
                
                if result:
                    # Add to history
                    st.session_state.chat_history.append({
                        "question": question,
                        "result": result
                    })
    
    # Display latest result
    if st.session_state.chat_history:
        latest = st.session_state.chat_history[-1]
        result = latest["result"]
        
        st.markdown("---")
        
        # Question
        st.markdown(f"**Question:** {latest['question']}")
        
        # Confidence Score
        confidence = result.get("confidence_score", 0)
        confidence_class = get_confidence_class(confidence)
        confidence_label = get_confidence_label(confidence)
        
        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown(
                f'<div class="{confidence_class}">{confidence_label}: {confidence:.1%}</div>',
                unsafe_allow_html=True
            )
        
        # Guardrail Warning
        if result.get("guardrail_triggered"):
            st.markdown(
                f'<div class="guardrail-warning">⚠️ <strong>Guardrail Triggered:</strong> {result.get("guardrail_reason", "Low confidence")}</div>',
                unsafe_allow_html=True
            )
        
        # Answer
        st.markdown(
            f'<div class="answer-box"><strong>Answer:</strong><br>{result.get("answer", "No answer generated")}</div>',
            unsafe_allow_html=True
        )
        
        # Sources
        sources = result.get("sources", [])
        if sources:
            with st.expander(f"📚 View Sources ({len(sources)} chunks)", expanded=False):
                for source in sources:
                    page_info = f" | Page {source['page_number']}" if source.get('page_number') else ""
                    st.markdown(
                        f"""<div class="source-box">
                        <strong>Rank {source['rank']}</strong> | Similarity: {source['similarity_score']:.1%}{page_info}
                        <br><br>{source['content']}
                        </div>""",
                        unsafe_allow_html=True
                    )


# ============== Extraction Tab ==============

with tab_extract:
    st.markdown("### Extract Structured Shipment Data")
    st.markdown("Extract key fields from your logistics document into a structured format.")
    
    if st.button("📋 Run Extraction", type="primary"):
        if st.session_state.demo_mode:
            # Demo mode: use sample extraction data
            st.session_state.extraction_result = {
                "data": DEMO_EXTRACTION_DATA,
                "metadata": {
                    "extraction_confidence": 0.94,
                    "fields_extracted": 11,
                    "fields_total": 11
                }
            }
        else:
            with st.spinner("Extracting structured data..."):
                result = extract_data(st.session_state.document_id)
                if result:
                    st.session_state.extraction_result = result
    
    if st.session_state.extraction_result:
        result = st.session_state.extraction_result
        data = result.get("data", {})
        metadata = result.get("metadata", {})
        
        # Confidence
        extraction_confidence = metadata.get("extraction_confidence", 0)
        fields_extracted = metadata.get("fields_extracted", 0)
        fields_total = metadata.get("fields_total", 11)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Extraction Confidence", f"{extraction_confidence:.1%}")
        with col2:
            st.metric("Fields Extracted", f"{fields_extracted}/{fields_total}")
        with col3:
            st.metric("Completion Rate", f"{fields_extracted/fields_total:.1%}")
        
        st.divider()
        
        # Display extracted data
        col1, col2 = st.columns(2)
        
        field_mapping = {
            "shipment_id": "Shipment ID",
            "shipper": "Shipper",
            "consignee": "Consignee",
            "pickup_datetime": "Pickup Date/Time",
            "delivery_datetime": "Delivery Date/Time",
            "equipment_type": "Equipment Type",
            "mode": "Mode",
            "rate": "Rate",
            "currency": "Currency",
            "weight": "Weight",
            "carrier_name": "Carrier Name"
        }
        
        fields = list(field_mapping.items())
        mid = len(fields) // 2 + 1
        
        with col1:
            for key, label in fields[:mid]:
                value = data.get(key)
                if value:
                    st.markdown(f"**{label}:**")
                    st.info(value)
                else:
                    st.markdown(f"**{label}:**")
                    st.warning("Not found")
        
        with col2:
            for key, label in fields[mid:]:
                value = data.get(key)
                if value:
                    st.markdown(f"**{label}:**")
                    st.info(value)
                else:
                    st.markdown(f"**{label}:**")
                    st.warning("Not found")
        
        st.divider()
        
        # JSON output
        with st.expander("📄 View Raw JSON", expanded=False):
            # Clean JSON without metadata for display
            clean_data = {k: v for k, v in data.items() if not k.startswith("_")}
            st.json(clean_data)
        
        # Download button
        json_str = json.dumps(data, indent=2)
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"extraction_{st.session_state.document_id}.json",
            mime="application/json"
        )


# ============== History Tab ==============

with tab_history:
    st.markdown("### Question History")
    
    if not st.session_state.chat_history:
        st.info("No questions asked yet. Go to the 'Ask Questions' tab to get started.")
    else:
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            result = item["result"]
            confidence = result.get("confidence_score", 0)
            
            with st.expander(
                f"Q: {item['question'][:50]}... | Confidence: {confidence:.1%}",
                expanded=(i == 0)
            ):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(f"**Answer:** {result.get('answer', 'N/A')}")
                st.markdown(f"**Confidence:** {confidence:.1%}")
                
                if result.get("guardrail_triggered"):
                    st.warning(f"Guardrail: {result.get('guardrail_reason')}")


# ============== GraphRAG Tab ==============

with tab_graph:
    st.markdown("### Knowledge Graph Q&A")
    st.markdown("Query relationships between entities across your documents using GraphRAG.")
    
    # Graph Stats
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Index Document")
        if st.button("🕸️ Build Knowledge Graph", type="secondary", use_container_width=True):
            if st.session_state.demo_mode:
                st.success("✅ Demo: Knowledge graph built with 6 entities and 5 relationships")
                st.session_state.graph_stats = {
                    "total_entities": 6,
                    "total_relationships": 5,
                    "documents_indexed": 1
                }
            else:
                with st.spinner("Extracting entities and relationships..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/graph/index",
                            json={"document_id": st.session_state.document_id}
                        )
                        if response.ok:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                        else:
                            st.error(f"Failed: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    with col2:
        st.markdown("#### Graph Statistics")
        if st.button("📊 Refresh Stats", use_container_width=True):
            if st.session_state.demo_mode:
                st.session_state.graph_stats = {
                    "total_entities": 6,
                    "total_relationships": 5,
                    "documents_indexed": 1
                }
            else:
                try:
                    response = requests.get(f"{API_BASE_URL}/graph/stats")
                    if response.ok:
                        stats = response.json()
                        st.session_state.graph_stats = stats
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if "graph_stats" in st.session_state:
            stats = st.session_state.graph_stats
            st.metric("Entities", stats.get("total_entities", 0))
            st.metric("Relationships", stats.get("total_relationships", 0))
            st.metric("Documents", stats.get("documents_indexed", 0))
    
    st.markdown("---")
    st.markdown("#### Ask Relationship Questions")
    
    graph_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., Which carriers have delivered shipments for this shipper?",
        key="graph_question_input"
    )
    
    if st.button("🔍 Query Graph", type="primary"):
        if graph_question:
            if st.session_state.demo_mode:
                # Demo mode: use sample GraphRAG response
                result = DEMO_GRAPH_RESPONSE
                
                # Confidence
                confidence = result.get("confidence", 0)
                confidence_class = get_confidence_class(confidence)
                
                col1, col2 = st.columns([3, 1])
                with col2:
                    st.markdown(
                        f'<div class="{confidence_class}">Confidence: {confidence:.1%}</div>',
                        unsafe_allow_html=True
                    )
                
                # Answer
                st.markdown(
                    f'<div class="answer-box"><strong>Answer:</strong><br>{result.get("answer", "No answer found")}</div>',
                    unsafe_allow_html=True
                )
                
                # Stats
                st.markdown(f"*Found {result.get('entities_found', 0)} entities and {result.get('relationships_found', 0)} relationships*")
                
                # Reasoning
                if result.get("reasoning"):
                    with st.expander("🧠 View Reasoning", expanded=False):
                        st.markdown(result["reasoning"])
            else:
                with st.spinner("Reasoning over knowledge graph..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/graph/query",
                            json={"query": graph_question, "max_entities": 20}
                        )
                        if response.ok:
                            result = response.json()
                            
                            # Confidence
                            confidence = result.get("confidence", 0)
                            confidence_class = get_confidence_class(confidence)
                            
                            col1, col2 = st.columns([3, 1])
                            with col2:
                                st.markdown(
                                    f'<div class="{confidence_class}">Confidence: {confidence:.1%}</div>',
                                    unsafe_allow_html=True
                                )
                            
                            # Answer
                            st.markdown(
                                f'<div class="answer-box"><strong>Answer:</strong><br>{result.get("answer", "No answer found")}</div>',
                                unsafe_allow_html=True
                            )
                            
                            # Stats
                            st.markdown(f"*Found {result.get('entities_found', 0)} entities and {result.get('relationships_found', 0)} relationships*")
                            
                            # Reasoning
                            if result.get("reasoning"):
                                with st.expander("🧠 View Reasoning", expanded=False):
                                    st.markdown(result["reasoning"])
                        else:
                            st.error(f"Query failed: {response.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question.")
    
    st.markdown("---")
    st.markdown("##### Example Questions")
    st.markdown("""
    - What is the relationship between the shipper and carrier?
    - Which locations are connected in this shipment?
    - What equipment type is being used?
    - Show all entities related to this shipment
    """)


# ============== Footer ==============

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        Ultra Doc-Intelligence v1.0 | Powered by xAI Grok & LangChain
    </div>
    """,
    unsafe_allow_html=True
)
