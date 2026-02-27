/**
 * Ultra Doc-Intelligence - Node.js API Gateway
 * 
 * Acts as an API gateway between React frontend and Python ML services.
 * Handles file uploads, proxies requests to FastAPI, and manages sessions.
 */

import express from 'express';
import cors from 'cors';
import multer from 'multer';
import axios from 'axios';
import FormData from 'form-data';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3001;
const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

// Middleware
app.use(cors());
app.use(express.json());

// Configure multer for file uploads
const storage = multer.memoryStorage();
const upload = multer({ 
  storage,
  limits: { fileSize: 200 * 1024 * 1024 }, // 200MB limit
  fileFilter: (req, file, cb) => {
    const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
    if (allowedTypes.includes(file.mimetype) || file.originalname.match(/\.(pdf|docx|txt)$/i)) {
      cb(null, true);
    } else {
      cb(new Error('Only PDF, DOCX, and TXT files are allowed'));
    }
  }
});

// Health check endpoint
app.get('/api/health', async (req, res) => {
  try {
    const response = await axios.get(`${FASTAPI_URL}/health`, { timeout: 5000 });
    res.json({ 
      status: 'healthy', 
      gateway: 'node.js',
      mlService: response.data 
    });
  } catch (error) {
    res.json({ 
      status: 'degraded', 
      gateway: 'node.js',
      mlService: 'unavailable',
      demo_mode: true
    });
  }
});

// Document upload endpoint
app.post('/api/upload', upload.single('file'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    // Forward to FastAPI
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype
    });

    const response = await axios.post(`${FASTAPI_URL}/upload`, formData, {
      headers: formData.getHeaders(),
      timeout: 60000
    });

    res.json(response.data);
  } catch (error) {
    console.error('Upload error:', error.message);
    
    // Demo mode fallback
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      const crypto = await import('crypto');
      const demoDocId = crypto.createHash('md5').update(req.file.originalname).digest('hex');
      return res.json({
        success: true,
        document_id: demoDocId,
        filename: req.file.originalname,
        demo_mode: true,
        stats: { chunk_count: 8, page_count: 2 }
      });
    }
    
    res.status(500).json({ error: error.message });
  }
});

// Upload status polling endpoint
app.get('/api/upload/status/:taskId', async (req, res) => {
  try {
    const response = await axios.get(`${FASTAPI_URL}/upload/status/${req.params.taskId}`, { timeout: 10000 });
    res.json(response.data);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Q&A endpoint
app.post('/api/ask', async (req, res) => {
  try {
    const { document_id, question, min_confidence = 0.4, use_reasoning = true } = req.body;

    const response = await axios.post(`${FASTAPI_URL}/ask`, {
      document_id,
      question,
      min_confidence,
      use_reasoning
    }, { timeout: 120000 });

    res.json(response.data);
  } catch (error) {
    console.error('Ask error:', error.message);
    
    // Demo mode fallback
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return res.json(getDemoQAResponse(req.body.question));
    }
    
    res.status(500).json({ error: error.message });
  }
});

// Extraction endpoint
app.post('/api/extract', async (req, res) => {
  try {
    const response = await axios.post(`${FASTAPI_URL}/extract`, req.body, { timeout: 60000 });
    res.json(response.data);
  } catch (error) {
    console.error('Extract error:', error.message);
    
    // Demo mode fallback
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return res.json(getDemoExtractionResponse());
    }
    
    res.status(500).json({ error: error.message });
  }
});

// GraphRAG index endpoint
app.post('/api/graph/index', async (req, res) => {
  try {
    const response = await axios.post(`${FASTAPI_URL}/graph/index`, req.body, { timeout: 120000 });
    res.json(response.data);
  } catch (error) {
    console.error('Graph index error:', error.message);
    
    // Demo mode fallback
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return res.json({ 
        message: 'Knowledge graph built with 6 entities and 5 relationships',
        demo_mode: true
      });
    }
    
    res.status(500).json({ error: error.message });
  }
});

// GraphRAG query endpoint
app.post('/api/graph/query', async (req, res) => {
  try {
    const response = await axios.post(`${FASTAPI_URL}/graph/query`, req.body, { timeout: 60000 });
    res.json(response.data);
  } catch (error) {
    console.error('Graph query error:', error.message);
    
    // Demo mode fallback
    if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
      return res.json(getDemoGraphResponse());
    }
    
    res.status(500).json({ error: error.message });
  }
});

// GraphRAG stats endpoint
app.get('/api/graph/stats', async (req, res) => {
  try {
    const response = await axios.get(`${FASTAPI_URL}/graph/stats`, { timeout: 10000 });
    res.json(response.data);
  } catch (error) {
    // Demo mode fallback
    res.json({ 
      total_entities: 6, 
      total_relationships: 5, 
      documents_indexed: 1,
      demo_mode: true
    });
  }
});

// Demo response helpers
function getDemoQAResponse(question) {
  const q = question?.toLowerCase() || '';
  
  if (q.includes('rate') || q.includes('cost') || q.includes('price')) {
    return {
      answer: 'The carrier rate for this shipment is $2,450.00 USD. This is a flat rate for Full Truckload (FTL) service using a 53\' Dry Van.',
      confidence_score: 0.96,
      guardrail_triggered: false,
      demo_mode: true,
      sources: [{ rank: 1, similarity_score: 0.98, page_number: 1, content: 'RATE: $2,450.00 USD\nPayment Terms: Net 30 Days' }]
    };
  }
  
  if (q.includes('pickup') || q.includes('origin')) {
    return {
      answer: 'Pickup is scheduled for February 24, 2026 at 08:00 AM at ABC Manufacturing Co. facility.',
      confidence_score: 0.94,
      guardrail_triggered: false,
      demo_mode: true,
      sources: [{ rank: 1, similarity_score: 0.96, page_number: 1, content: 'PICKUP: 02/24/2026 @ 08:00 AM\nLocation: ABC Manufacturing Co.' }]
    };
  }
  
  return {
    answer: 'Based on the document analysis, the shipment LD53657 is a Full Truckload (FTL) movement from ABC Manufacturing Co. to XYZ Distribution Center. The carrier rate is $2,450.00 USD for a 53\' Dry Van equipment type.',
    confidence_score: 0.92,
    guardrail_triggered: false,
    demo_mode: true,
    sources: [
      { rank: 1, similarity_score: 0.95, page_number: 1, content: 'RATE CONFIRMATION\nLoad #: LD53657\nCarrier: FastFreight Logistics LLC\nRate: $2,450.00 USD' },
      { rank: 2, similarity_score: 0.88, page_number: 1, content: 'Shipper: ABC Manufacturing Co.\nConsignee: XYZ Distribution Center\nEquipment: 53\' Dry Van' }
    ]
  };
}

function getDemoExtractionResponse() {
  return {
    data: {
      shipment_id: 'LD53657',
      shipper: 'ABC Manufacturing Co.',
      consignee: 'XYZ Distribution Center',
      pickup_datetime: '2026-02-24T08:00:00',
      delivery_datetime: '2026-02-26T14:00:00',
      equipment_type: '53\' Dry Van',
      mode: 'FTL (Full Truckload)',
      rate: '$2,450.00',
      currency: 'USD',
      weight: '42,000 lbs',
      carrier_name: 'FastFreight Logistics LLC'
    },
    metadata: {
      extraction_confidence: 0.94,
      fields_extracted: 11,
      fields_total: 11
    },
    demo_mode: true
  };
}

function getDemoGraphResponse() {
  return {
    answer: 'The knowledge graph shows that FastFreight Logistics LLC (Carrier) is transporting shipment LD53657 from ABC Manufacturing Co. (Shipper) to XYZ Distribution Center (Consignee). The shipment uses a 53\' Dry Van and follows an FTL mode.',
    confidence: 0.89,
    entities_found: 6,
    relationships_found: 5,
    reasoning: '1. Identified carrier entity: FastFreight Logistics LLC\n2. Found shipper-consignee relationship: ABC Manufacturing -> XYZ Distribution\n3. Mapped equipment and mode attributes',
    demo_mode: true
  };
}

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Node.js API Gateway running on http://localhost:${PORT}`);
  console.log(`📡 Proxying to FastAPI at ${FASTAPI_URL}`);
});
