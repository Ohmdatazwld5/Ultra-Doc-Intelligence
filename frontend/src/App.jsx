import { useState, useEffect } from 'react';
import { Upload, MessageSquare, FileText, Network, History, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { healthCheck, uploadDocument, askQuestion, extractData, indexGraph, queryGraph, getGraphStats } from './services/api';

function App() {
  const [activeTab, setActiveTab] = useState('qa');
  const [document, setDocument] = useState(null);
  const [isHealthy, setIsHealthy] = useState(false);
  const [demoMode, setDemoMode] = useState(false);
  const [loading, setLoading] = useState(false);
  
  // Q&A State
  const [question, setQuestion] = useState('');
  const [qaResult, setQaResult] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  
  // Extraction State
  const [extractionResult, setExtractionResult] = useState(null);
  
  // GraphRAG State
  const [graphQuestion, setGraphQuestion] = useState('');
  const [graphResult, setGraphResult] = useState(null);
  const [graphStats, setGraphStats] = useState(null);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const health = await healthCheck();
      setIsHealthy(health.status === 'healthy');
      setDemoMode(health.demo_mode || health.mlService === 'unavailable');
    } catch {
      setIsHealthy(false);
      setDemoMode(true);
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setLoading(true);
    try {
      const result = await uploadDocument(file);
      setDocument({
        id: result.document_id,
        name: result.filename,
        stats: result.stats
      });
      setChatHistory([]);
      setExtractionResult(null);
      setGraphResult(null);
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim() || !document) return;
    
    setLoading(true);
    try {
      const result = await askQuestion(document.id, question);
      setQaResult(result);
      setChatHistory([...chatHistory, { question, result }]);
      setQuestion('');
    } catch (error) {
      console.error('Ask failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleExtract = async () => {
    if (!document) return;
    
    setLoading(true);
    try {
      const result = await extractData(document.id);
      setExtractionResult(result);
    } catch (error) {
      console.error('Extract failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleBuildGraph = async () => {
    if (!document) return;
    
    setLoading(true);
    try {
      await indexGraph(document.id);
      const stats = await getGraphStats();
      setGraphStats(stats);
    } catch (error) {
      console.error('Graph build failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleGraphQuery = async () => {
    if (!graphQuestion.trim()) return;
    
    setLoading(true);
    try {
      const result = await queryGraph(graphQuestion);
      setGraphResult(result);
    } catch (error) {
      console.error('Graph query failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const getConfidenceClass = (score) => {
    if (score >= 0.75) return 'confidence-high';
    if (score >= 0.4) return 'confidence-medium';
    return 'confidence-low';
  };

  const tabs = [
    { id: 'qa', label: 'Ask Questions', icon: MessageSquare },
    { id: 'extract', label: 'Extract Data', icon: FileText },
    { id: 'graph', label: 'GraphRAG', icon: Network },
    { id: 'history', label: 'History', icon: History }
  ];

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">🔍 Ultra Doc-Intelligence</h1>
            <p className="text-gray-400 text-sm">AI-powered logistics document analysis with RAG and GraphRAG</p>
          </div>
          <div className="flex items-center gap-4">
            {demoMode ? (
              <span className="bg-purple-600 text-white px-3 py-1 rounded-full text-sm font-medium">
                🎭 Demo Mode
              </span>
            ) : isHealthy ? (
              <span className="flex items-center gap-1 text-green-400 text-sm">
                <CheckCircle size={16} /> API Connected
              </span>
            ) : (
              <span className="flex items-center gap-1 text-red-400 text-sm">
                <AlertCircle size={16} /> API Disconnected
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-6 py-6 flex gap-6">
        {/* Sidebar */}
        <aside className="w-72 flex-shrink-0">
          <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
            <h3 className="text-white font-semibold mb-4 flex items-center gap-2">
              <Upload size={18} /> Upload Document
            </h3>
            
            <label className="block border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-blue-500 transition">
              <input type="file" className="hidden" accept=".pdf,.docx,.txt" onChange={handleUpload} />
              <Upload className="mx-auto text-gray-500 mb-2" size={32} />
              <p className="text-gray-400 text-sm">Drop files or click to upload</p>
              <p className="text-gray-500 text-xs mt-1">PDF, DOCX, TXT (max 200MB)</p>
            </label>

            {loading && (
              <div className="mt-4 flex items-center gap-2 text-blue-400">
                <Loader2 className="animate-spin" size={16} />
                <span className="text-sm">Processing...</span>
              </div>
            )}

            {document && (
              <div className="mt-4 p-3 bg-gray-700 rounded-lg">
                <p className="text-white font-medium truncate">{document.name}</p>
                <p className="text-gray-400 text-xs mt-1">
                  {document.stats?.chunk_count || 8} chunks • {document.stats?.page_count || 2} pages
                </p>
              </div>
            )}
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1">
          {demoMode && (
            <div className="bg-gradient-to-r from-purple-600 to-indigo-600 rounded-lg p-4 mb-6 text-center">
              <p className="text-white font-medium">
                🎭 Demo Mode Active - Showing sample responses. Deploy backend for live processing.
              </p>
            </div>
          )}

          {!document ? (
            <div className="bg-gray-800 rounded-lg p-8 text-center border border-gray-700">
              <h2 className="text-xl text-white mb-4">👈 Upload a document to get started</h2>
              <div className="grid grid-cols-2 gap-4 max-w-md mx-auto text-left">
                <div className="text-gray-400 text-sm">
                  <p>• What is the carrier rate?</p>
                  <p>• When is the pickup?</p>
                </div>
                <div className="text-gray-400 text-sm">
                  <p>• Who is the consignee?</p>
                  <p>• What equipment type?</p>
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* Tabs */}
              <div className="flex gap-1 mb-6 bg-gray-800 p-1 rounded-lg">
                {tabs.map(tab => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition ${
                      activeTab === tab.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-400 hover:text-white hover:bg-gray-700'
                    }`}
                  >
                    <tab.icon size={16} />
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Q&A Tab */}
              {activeTab === 'qa' && (
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h3 className="text-white font-semibold mb-4">Ask Questions About Your Document</h3>
                  
                  <div className="flex gap-3 mb-6">
                    <input
                      type="text"
                      value={question}
                      onChange={(e) => setQuestion(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
                      placeholder="e.g., What is the carrier rate?"
                      className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:border-blue-500 focus:outline-none"
                    />
                    <button
                      onClick={handleAsk}
                      disabled={loading || !question.trim()}
                      className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-medium transition flex items-center gap-2"
                    >
                      {loading ? <Loader2 className="animate-spin" size={16} /> : '🔍'} Ask
                    </button>
                  </div>

                  {qaResult && (
                    <div className="space-y-4">
                      <div className="flex justify-between items-start">
                        <span className="text-gray-400 text-sm">Confidence:</span>
                        <span className={getConfidenceClass(qaResult.confidence_score)}>
                          {(qaResult.confidence_score * 100).toFixed(0)}%
                        </span>
                      </div>
                      
                      {qaResult.guardrail_triggered && (
                        <div className="bg-yellow-900/50 border border-yellow-600 rounded-lg p-3">
                          <p className="text-yellow-400 text-sm">⚠️ Guardrail triggered: Low confidence</p>
                        </div>
                      )}
                      
                      <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-4">
                        <p className="text-white">{qaResult.answer}</p>
                      </div>

                      {qaResult.sources?.length > 0 && (
                        <details className="text-gray-400">
                          <summary className="cursor-pointer text-sm hover:text-white">
                            📚 View Sources ({qaResult.sources.length})
                          </summary>
                          <div className="mt-2 space-y-2">
                            {qaResult.sources.map((source, i) => (
                              <div key={i} className="bg-gray-700 rounded p-3 text-sm">
                                <p className="text-blue-400 mb-1">
                                  Rank {source.rank} • {(source.similarity_score * 100).toFixed(0)}% match
                                </p>
                                <p className="text-gray-300">{source.content}</p>
                              </div>
                            ))}
                          </div>
                        </details>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Extraction Tab */}
              {activeTab === 'extract' && (
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h3 className="text-white font-semibold mb-4">Extract Structured Shipment Data</h3>
                  
                  <button
                    onClick={handleExtract}
                    disabled={loading}
                    className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-medium transition flex items-center gap-2 mb-6"
                  >
                    {loading ? <Loader2 className="animate-spin" size={16} /> : '📋'} Run Extraction
                  </button>

                  {extractionResult && (
                    <div className="space-y-4">
                      <div className="grid grid-cols-3 gap-4 mb-6">
                        <div className="bg-gray-700 rounded-lg p-3 text-center">
                          <p className="text-2xl font-bold text-green-400">
                            {(extractionResult.metadata?.extraction_confidence * 100).toFixed(0)}%
                          </p>
                          <p className="text-gray-400 text-sm">Confidence</p>
                        </div>
                        <div className="bg-gray-700 rounded-lg p-3 text-center">
                          <p className="text-2xl font-bold text-blue-400">
                            {extractionResult.metadata?.fields_extracted}/{extractionResult.metadata?.fields_total}
                          </p>
                          <p className="text-gray-400 text-sm">Fields Extracted</p>
                        </div>
                        <div className="bg-gray-700 rounded-lg p-3 text-center">
                          <p className="text-2xl font-bold text-purple-400">
                            {((extractionResult.metadata?.fields_extracted / extractionResult.metadata?.fields_total) * 100).toFixed(0)}%
                          </p>
                          <p className="text-gray-400 text-sm">Completion</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4">
                        {Object.entries(extractionResult.data || {}).map(([key, value]) => (
                          <div key={key} className="bg-gray-700 rounded-lg p-3">
                            <p className="text-gray-400 text-xs uppercase mb-1">
                              {key.replace(/_/g, ' ')}
                            </p>
                            <p className={value ? 'text-white' : 'text-gray-500 italic'}>
                              {value || 'Not found'}
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* GraphRAG Tab */}
              {activeTab === 'graph' && (
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h3 className="text-white font-semibold mb-4">Knowledge Graph Q&A</h3>
                  
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div>
                      <button
                        onClick={handleBuildGraph}
                        disabled={loading}
                        className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition flex items-center justify-center gap-2"
                      >
                        {loading ? <Loader2 className="animate-spin" size={16} /> : '🕸️'} Build Knowledge Graph
                      </button>
                    </div>
                    <div>
                      {graphStats && (
                        <div className="bg-gray-700 rounded-lg p-3 text-center">
                          <p className="text-white">
                            <span className="text-purple-400 font-bold">{graphStats.total_entities}</span> entities • 
                            <span className="text-blue-400 font-bold ml-1">{graphStats.total_relationships}</span> relationships
                          </p>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex gap-3 mb-6">
                    <input
                      type="text"
                      value={graphQuestion}
                      onChange={(e) => setGraphQuestion(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && handleGraphQuery()}
                      placeholder="e.g., What is the relationship between shipper and carrier?"
                      className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:border-purple-500 focus:outline-none"
                    />
                    <button
                      onClick={handleGraphQuery}
                      disabled={loading || !graphQuestion.trim()}
                      className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-medium transition"
                    >
                      Query
                    </button>
                  </div>

                  {graphResult && (
                    <div className="space-y-4">
                      <div className="flex justify-between items-start">
                        <span className="text-gray-400 text-sm">
                          Found {graphResult.entities_found} entities, {graphResult.relationships_found} relationships
                        </span>
                        <span className={getConfidenceClass(graphResult.confidence)}>
                          {(graphResult.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                      
                      <div className="bg-purple-900/30 border border-purple-700 rounded-lg p-4">
                        <p className="text-white">{graphResult.answer}</p>
                      </div>

                      {graphResult.reasoning && (
                        <details className="text-gray-400">
                          <summary className="cursor-pointer text-sm hover:text-white">🧠 View Reasoning</summary>
                          <div className="mt-2 bg-gray-700 rounded p-3 text-sm whitespace-pre-wrap">
                            {graphResult.reasoning}
                          </div>
                        </details>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* History Tab */}
              {activeTab === 'history' && (
                <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h3 className="text-white font-semibold mb-4">Question History</h3>
                  
                  {chatHistory.length === 0 ? (
                    <p className="text-gray-400">No questions asked yet.</p>
                  ) : (
                    <div className="space-y-4">
                      {chatHistory.slice().reverse().map((item, i) => (
                        <div key={i} className="bg-gray-700 rounded-lg p-4">
                          <p className="text-blue-400 font-medium mb-2">Q: {item.question}</p>
                          <p className="text-gray-300 text-sm">{item.result.answer}</p>
                          <p className="text-gray-500 text-xs mt-2">
                            Confidence: {(item.result.confidence_score * 100).toFixed(0)}%
                          </p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </main>
      </div>

      {/* Footer */}
      <footer className="text-center text-gray-500 text-sm py-4">
        Ultra Doc-Intelligence v1.0 | React + Node.js + FastAPI | Powered by xAI Grok
      </footer>
    </div>
  );
}

export default App;
