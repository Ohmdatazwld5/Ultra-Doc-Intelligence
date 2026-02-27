import axios from 'axios';

const API_BASE = '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000
});

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  const response = await api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
};

export const askQuestion = async (documentId, question, minConfidence = 0.4, useReasoning = true) => {
  const response = await api.post('/ask', {
    document_id: documentId,
    question,
    min_confidence: minConfidence,
    use_reasoning: useReasoning
  });
  return response.data;
};

export const extractData = async (documentId) => {
  const response = await api.post('/extract', {
    document_id: documentId,
    use_llm: true
  });
  return response.data;
};

export const indexGraph = async (documentId) => {
  const response = await api.post('/graph/index', {
    document_id: documentId
  });
  return response.data;
};

export const queryGraph = async (query, maxEntities = 20) => {
  const response = await api.post('/graph/query', {
    query,
    max_entities: maxEntities
  });
  return response.data;
};

export const getGraphStats = async () => {
  const response = await api.get('/graph/stats');
  return response.data;
};

export default api;
