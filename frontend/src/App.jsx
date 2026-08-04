import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('search'); // 'search' | 'upload'
  
  // --- Search State ---
  const [appState, setAppState] = useState('idle'); // 'idle' | 'fetching' | 'simulating' | 'results'
  const [realLogs, setRealLogs] = useState([]);
  const [displayedLogs, setDisplayedLogs] = useState([]);
  const [results, setResults] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  
  // --- Upload State ---
  const [uploadStatus, setUploadStatus] = useState('idle'); // 'idle' | 'uploading' | 'success' | 'error'
  const [uploadLogs, setUploadLogs] = useState([]);
  
  const fileInputRef = useRef(null);
  const adminFileInputRef = useRef(null);

  // Handle real-time line-by-line simulation for Search
  useEffect(() => {
    if (appState === 'simulating' && realLogs.length > 0) {
      let currentIndex = 0;
      
      const interval = setInterval(() => {
        if (currentIndex < realLogs.length) {
          const logToAdd = realLogs[currentIndex];
          setDisplayedLogs(prev => {
            if (prev.includes(logToAdd)) return prev;
            return [...prev, logToAdd];
          });
          currentIndex += 1;
        } else {
          clearInterval(interval);
          setTimeout(() => {
            setAppState('results');
          }, 800);
        }
      }, 600);
      
      return () => clearInterval(interval);
    }
  }, [appState, realLogs]);

  // --- Drag & Drop Handlers ---
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = async (e, mode) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      if (mode === 'search') processFile(e.dataTransfer.files[0]);
      if (mode === 'upload') handleAdminUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e, mode) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      if (mode === 'search') processFile(e.target.files[0]);
      if (mode === 'upload') handleAdminUpload(e.target.files[0]);
    }
  };

  // --- Search Pipeline ---
  const processFile = async (file) => {
    const objectUrl = URL.createObjectURL(file);
    setUploadedImage(objectUrl);
    setAppState('fetching');
    setError(null);
    setDisplayedLogs(["> [LOCAL] Uploading image payload to AWS EC2..."]);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('/api/recommend?num_results=5', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error('API Request Failed');
      
      const data = await response.json();
      setResults(data.recommendations);
      
      if (data.logs) {
        setRealLogs(data.logs.map(log => "> " + log));
        setAppState('simulating');
      } else {
        setRealLogs([
          "> [EC2] WARNING: Backend returned old format.",
          "> [EC2] Showing fallback results..."
        ]);
        setAppState('simulating');
      }
    } catch (err) {
      console.error(err);
      setError("Failed to connect to the AWS AI Server.");
      setAppState('idle');
    }
  };

  // --- Admin Upload Pipeline ---
  const handleAdminUpload = async (file) => {
    const objectUrl = URL.createObjectURL(file);
    setUploadedImage(objectUrl);
    setUploadStatus('uploading');
    setError(null);
    setDisplayedLogs(["> [LOCAL] Initiating secure S3 upload & ChromaDB injection..."]);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch('/api/add_to_catalog', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) throw new Error('Upload Failed');
      
      const data = await response.json();
      if (data.logs) {
        setUploadLogs(data.logs.map(log => "> " + log));
      }
      setUploadStatus('success');
    } catch (err) {
      console.error(err);
      setError("Failed to push to AWS backend. Check IAM Secrets.");
      setUploadStatus('error');
    }
  };

  const resetApp = () => {
    setAppState('idle');
    setUploadStatus('idle');
    setResults([]);
    setError(null);
    setDisplayedLogs([]);
    setUploadLogs([]);
    setUploadedImage(null);
  };

  return (
    <div className="app-container">
      
      {/* Navigation Bar */}
      <nav className="nav-tabs glass-panel" style={{ display: 'flex', gap: '20px', padding: '10px 20px', borderRadius: '30px', marginBottom: '40px' }}>
        <button 
          onClick={() => { setActiveTab('search'); resetApp(); }}
          style={{ background: activeTab === 'search' ? 'rgba(167, 139, 250, 0.4)' : 'transparent', border: 'none', color: '#fff', padding: '10px 20px', borderRadius: '20px', cursor: 'pointer', fontWeight: 'bold' }}>
          Search AI
        </button>
        <button 
          onClick={() => { setActiveTab('upload'); resetApp(); }}
          style={{ background: activeTab === 'upload' ? 'rgba(167, 139, 250, 0.4)' : 'transparent', border: 'none', color: '#fff', padding: '10px 20px', borderRadius: '20px', cursor: 'pointer', fontWeight: 'bold' }}>
          Admin Database
        </button>
      </nav>

      <header>
        {activeTab === 'search' ? (
          <>
            <h1>Discover Your Style <span className="text-gradient">Powered by AI</span></h1>
            <p>Upload a photo of any clothing item. Our ResNet50 neural network will find visually identical matches from 44,000+ items in milliseconds.</p>
          </>
        ) : (
          <>
            <h1>Dynamic <span className="text-gradient">Dataset Injection</span></h1>
            <p>Upload new fashion items directly to AWS S3. The AI will extract features and dynamically inject them into ChromaDB without rebuilding.</p>
          </>
        )}
      </header>

      {error && <div style={{color: '#ef4444', marginBottom: '20px', padding: '15px', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '8px'}}>{error}</div>}

      {/* ----------------- SEARCH TAB ----------------- */}
      {activeTab === 'search' && (
        <>
          {appState === 'idle' && (
            <div 
              className={`upload-zone glass-panel ${dragActive ? "drag-active" : ""}`}
              onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag}
              onDrop={(e) => handleDrop(e, 'search')}
              onClick={() => fileInputRef.current.click()}
            >
              <input ref={fileInputRef} type="file" accept="image/*" onChange={(e) => handleChange(e, 'search')} style={{ display: "none" }} />
              <div className="upload-icon">✦</div>
              <div className="upload-text">Drag & Drop an image to Search</div>
            </div>
          )}

          {appState !== 'idle' && uploadedImage && (
            <div style={{ width: '100%', maxWidth: '700px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
              <div className="target-image-container glass-panel" style={{ padding: '15px', borderRadius: '12px', width: '200px' }}>
                <h3 style={{ fontSize: '1rem', color: '#a78bfa', marginBottom: '10px', textAlign: 'center' }}>Target Image</h3>
                <img src={uploadedImage} alt="Target" style={{ width: '100%', height: 'auto', borderRadius: '8px', objectFit: 'contain', background: '#fff' }} />
              </div>

              <div className="terminal-container" style={{ margin: '0', width: '100%' }}>
                <div className="terminal-header">
                  <div className="terminal-dots"><span className="dot red"></span><span className="dot yellow"></span><span className="dot green"></span></div>
                  <div className="terminal-title">bash - aws-ec2-ubuntu</div>
                </div>
                <div className="terminal-body">
                  {displayedLogs.map((log, index) => <div key={index} className="terminal-line">{log}</div>)}
                  {appState === 'simulating' && <div className="terminal-cursor">_</div>}
                </div>
              </div>
            </div>
          )}

          {appState === 'results' && (
            <div className="results-container" style={{ marginTop: '40px' }}>
              <div className="results-header">
                <h2>AI Recommendations</h2>
                <button className="reset-btn" onClick={resetApp}>Start Over</button>
              </div>
              <div className="image-grid smaller-grid">
                {results.map((item, index) => {
                  const url = typeof item === 'string' ? item : item.url;
                  const match = typeof item === 'string' ? null : item.match;
                  return (
                    <div key={index} className="image-card glass-panel progressive-image">
                      {match && <div className="match-badge">{match}</div>}
                      <img src={url} alt={`Recommendation ${index + 1}`} loading="lazy" />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </>
      )}

      {/* ----------------- ADMIN TAB ----------------- */}
      {activeTab === 'upload' && (
        <>
          {uploadStatus === 'idle' && (
            <div 
              className={`upload-zone glass-panel ${dragActive ? "drag-active" : ""}`}
              onDragEnter={handleDrag} onDragLeave={handleDrag} onDragOver={handleDrag}
              onDrop={(e) => handleDrop(e, 'upload')}
              onClick={() => adminFileInputRef.current.click()}
              style={{ border: '2px dashed #a78bfa' }}
            >
              <input ref={adminFileInputRef} type="file" accept="image/*" onChange={(e) => handleChange(e, 'upload')} style={{ display: "none" }} />
              <div className="upload-icon" style={{ color: '#a78bfa' }}>⚙️</div>
              <div className="upload-text">Upload new Image to ChromaDB & S3</div>
            </div>
          )}

          {uploadStatus !== 'idle' && uploadedImage && (
            <div style={{ width: '100%', maxWidth: '700px', margin: '0 auto', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
              <div className="target-image-container glass-panel" style={{ padding: '15px', borderRadius: '12px', width: '200px' }}>
                <h3 style={{ fontSize: '1rem', color: '#a78bfa', marginBottom: '10px', textAlign: 'center' }}>Database Injection</h3>
                <img src={uploadedImage} alt="Upload" style={{ width: '100%', height: 'auto', borderRadius: '8px', objectFit: 'contain', background: '#fff' }} />
              </div>

              <div className="terminal-container" style={{ margin: '0', width: '100%', border: '1px solid #a78bfa' }}>
                <div className="terminal-header">
                  <div className="terminal-dots"><span className="dot red"></span><span className="dot yellow"></span><span className="dot green"></span></div>
                  <div className="terminal-title" style={{ color: '#a78bfa' }}>bash - root@admin-console</div>
                </div>
                <div className="terminal-body">
                  <div className="terminal-line">{displayedLogs[0]}</div>
                  {uploadLogs.map((log, index) => <div key={index} className="terminal-line">{log}</div>)}
                  {uploadStatus === 'uploading' && <div className="terminal-cursor">_</div>}
                  {uploadStatus === 'success' && (
                    <div style={{ marginTop: '20px', color: '#27c93f' }}>Successfully added to live dataset!</div>
                  )}
                </div>
              </div>
              
              {uploadStatus === 'success' && (
                <button className="reset-btn" onClick={resetApp} style={{ marginTop: '20px', background: '#a78bfa' }}>
                  Add Another Item
                </button>
              )}
            </div>
          )}
        </>
      )}

    </div>
  );
}

export default App;
