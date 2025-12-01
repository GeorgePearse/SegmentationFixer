import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [proposal, setProposal] = useState(null);
  const [ws, setWs] = useState(null);
  const [status, setStatus] = useState('Disconnected');
  const [viewBox, setViewBox] = useState(null);
  const [progress, setProgress] = useState({ current: 0, total: 100 });
  const [isDone, setIsDone] = useState(false);
  const [selectedVariation, setSelectedVariation] = useState(null);
  const [fullscreenVariation, setFullscreenVariation] = useState(null);

  useEffect(() => {
    const socket = new WebSocket('ws://localhost:3000/ws');
    
    socket.onopen = () => setStatus('Connected');
    socket.onclose = () => setStatus('Disconnected');
    
    socket.onmessage = (event) => {
      try {
          const msg = JSON.parse(event.data);
          
          if (msg.type === 'Progress') {
              setProgress(msg.payload);
          } else if (msg.type === 'Proposal') {
              setProposal(msg.payload);
          } else if (msg.type === 'Done') {
              setIsDone(true);
              setProposal(null);
          }
      } catch (e) {
          console.error("Message parse error", e);
      }
    };
    
    setWs(socket);
    
    return () => socket.close();
  }, []);

  // Set initial zoom when proposal changes
  useEffect(() => {
    if (proposal) {
        if (proposal.focus_rect) {
            const { x, y, width, height } = proposal.focus_rect;
            // Ensure square aspect ratio or at least consistent with container?
            // Actually, keep strictly to the focus rect for max zoom
            setViewBox({ x, y, width, height });
        } else {
            setViewBox({ x: 0, y: 0, width: proposal.width, height: proposal.height });
        }
    }
  }, [proposal]);



  useEffect(() => {
    const handleKeyDown = (e) => {
      // Handle Escape to close fullscreen
      if (e.key === 'Escape' && fullscreenVariation !== null) {
        setFullscreenVariation(null);
        return;
      }

      if (!proposal || !ws) return;

      // Handle arrow keys for cycling variations (works in both normal and fullscreen mode)
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        if (proposal.variations && proposal.variations.length > 0) {
          const cycleVariation = (prev) => {
            if (prev === null || prev === 0) {
              return proposal.variations.length - 1;
            }
            return prev - 1;
          };

          if (fullscreenVariation !== null) {
            // In fullscreen: cycle fullscreen variation
            setFullscreenVariation(cycleVariation);
            setSelectedVariation(cycleVariation);
          } else {
            // Normal mode: cycle selected variation
            setSelectedVariation(cycleVariation);
          }
        }
        return;
      }

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        if (proposal.variations && proposal.variations.length > 0) {
          const cycleVariation = (prev) => {
            if (prev === null) {
              return 0;
            }
            return (prev + 1) % proposal.variations.length;
          };

          if (fullscreenVariation !== null) {
            // In fullscreen: cycle fullscreen variation
            setFullscreenVariation(cycleVariation);
            setSelectedVariation(cycleVariation);
          } else {
            // Normal mode: cycle selected variation
            setSelectedVariation(cycleVariation);
          }
        }
        return;
      }

      // Accept/Reject actions (also work in fullscreen)
      if (e.key === 'i' || e.key === 'I') {
        const variationToAccept = fullscreenVariation !== null ? fullscreenVariation : selectedVariation;
        if (variationToAccept === null) {
          alert('Please select a variation first');
          return;
        }
        ws.send(JSON.stringify({
          type: 'ACCEPT',
          annotation_id: proposal.annotation_id,
          variation_index: variationToAccept
        }));
        setProposal(null);
        setSelectedVariation(null);
        setFullscreenVariation(null);
      } else if (e.key === 'j' || e.key === 'J') {
        ws.send(JSON.stringify({
          type: 'REJECT',
          annotation_id: proposal.annotation_id
        }));
        setProposal(null);
        setSelectedVariation(null);
        setFullscreenVariation(null);
      } else if (e.key === 'ArrowLeft') {
        ws.send(JSON.stringify({
          type: 'REJECT',
          annotation_id: proposal.annotation_id
        }));
        setProposal(null);
        setSelectedVariation(null);
        setFullscreenVariation(null);
      } else if (e.key === 'ArrowRight') {
        const variationToAccept = fullscreenVariation !== null ? fullscreenVariation : selectedVariation;
        if (variationToAccept !== null) {
          ws.send(JSON.stringify({
            type: 'ACCEPT',
            annotation_id: proposal.annotation_id,
            variation_index: variationToAccept
          }));
        } else {
          ws.send(JSON.stringify({
            type: 'REJECT',
            annotation_id: proposal.annotation_id
          }));
        }
        setProposal(null);
        setSelectedVariation(null);
        setFullscreenVariation(null);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [proposal, ws, selectedVariation, fullscreenVariation]);

  const save = async () => {
    try {
        await fetch('http://localhost:3000/save', { method: 'POST' });
        console.log("Saved");
    } catch (e) {
        console.error('Error saving:', e);
    }
  };

  // Zoom / Pan Logic
  const handleWheel = (e) => {
    if (!viewBox) return;
    
    const scaleFactor = 1.1;
    const direction = e.deltaY > 0 ? 1 : -1;
    
    const newWidth = direction > 0 ? viewBox.width * scaleFactor : viewBox.width / scaleFactor;
    const newHeight = direction > 0 ? viewBox.height * scaleFactor : viewBox.height / scaleFactor;
    
    const dx = (viewBox.width - newWidth) / 2;
    const dy = (viewBox.height - newHeight) / 2;
    
    setViewBox({
        x: viewBox.x + dx,
        y: viewBox.y + dy,
        width: newWidth,
        height: newHeight
    });
  };

  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const handleMouseDown = (e) => {
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e) => {
      if (!isDragging || !viewBox) return;
      const svgElement = e.target.closest('svg');
      if (!svgElement) return;
      
      const { width: svgWidth, height: svgHeight } = svgElement.getBoundingClientRect();
      const dxScreen = e.clientX - dragStart.x;
      const dyScreen = e.clientY - dragStart.y;
      
      const dx = dxScreen * (viewBox.width / svgWidth);
      const dy = dyScreen * (viewBox.height / svgHeight);
      
      setViewBox(prev => ({
          ...prev,
          x: prev.x - dx,
          y: prev.y - dy
      }));
      
      setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseUp = () => {
      setIsDragging(false);
  };

  const renderPoly = (seg, color, dash, strokeWidth = 1.5) => {
    if (!seg) return null;
    const polygons = Array.isArray(seg) ? seg : (seg.Polygon || []);
    
    return polygons.map((points, idx) => {
        const pts = [];
        for(let i=0; i<points.length; i+=2) {
            pts.push(`${points[i]},${points[i+1]}`);
        }
        return (
            <polygon 
                key={idx} 
                points={pts.join(' ')} 
                fill="none" 
                stroke={color} 
                strokeWidth={strokeWidth} 
                strokeDasharray={dash ? "4,4" : "none"} 
                vectorEffect="non-scaling-stroke"
            />
        );
    });
  };

  if (!proposal) {
    const pct = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;
    
    return (
      <div className="container empty-state">
        <div className="status-badge">{status}</div>
        
        {isDone ? (
            <>
                <h1>All Done! 🎉</h1>
                <p>All images have been processed.</p>
                <button className="btn-primary" onClick={save}>Final Save</button>
            </>
        ) : (
            <>
                <h1>Processing Images...</h1>
                <p>The backend is scanning for corrections.</p>
                <div className="progress-bar-container">
                    <div className="progress-bar" style={{width: `${pct}%`}}></div>
                </div>
                <div className="progress-text">
                    {progress.current} / {progress.total} images
                </div>
                <button className="btn-primary" onClick={save}>Force Save</button>
            </>
        )}
      </div>
    );
  }

  const imageUrl = `http://localhost:3000/images/${proposal.file_name}`;
  const viewBoxStr = viewBox ? `${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}` : "0 0 100 100";
  const pct = progress.total > 0 ? (progress.current / progress.total) * 100 : 0;

  const renderVariationPoly = (variationPoints) => {
    const pts = [];
    for(let i=0; i<variationPoints.length; i+=2) {
        pts.push(`${variationPoints[i]},${variationPoints[i+1]}`);
    }
    return (
        <polygon 
            points={pts.join(' ')} 
            fill="none" 
            stroke="#00ff88" 
            strokeWidth={2} 
            vectorEffect="non-scaling-stroke"
        />
    );
  };

  return (
    <div className="app-layout">
      <header className="toolbar">
        <div className="brand">
            <span className="logo-icon">✨</span> Segmentation Fixer
        </div>
        <div className="mini-progress">
             <div className="mini-progress-bar" style={{width: `${pct}%`}}></div>
        </div>
        <div className="status-indicator">
            <span className={`dot ${status.toLowerCase()}`}></span> {status}
        </div>
        <div className="actions">
            <button className="btn-secondary" onClick={save}>Save Progress</button>
        </div>
      </header>

      <main className="workspace">
        {/* Top: Original + Selected Variation Comparison */}
        <div className="original-view"
             onWheel={handleWheel}
             onMouseDown={handleMouseDown}
             onMouseMove={handleMouseMove}
             onMouseUp={handleMouseUp}
             onMouseLeave={handleMouseUp}
        >
            <div className="view-label">
              {selectedVariation !== null && proposal.variations
                ? `Comparing: Original (red) vs ${proposal.variations[selectedVariation].name} (green)`
                : 'Original Segmentation (use ↑↓ to compare options)'}
            </div>
            <svg
                className="zoom-svg"
                viewBox={viewBoxStr}
                preserveAspectRatio="xMidYMid meet"
            >
                <image
                    href={imageUrl}
                    x="0"
                    y="0"
                    width={proposal.width}
                    height={proposal.height}
                />
                {renderPoly(proposal.original_segmentation, "#ff3e3e", false, 2)}
                {selectedVariation !== null && proposal.variations &&
                  renderVariationPoly(proposal.variations[selectedVariation].new_points)}
            </svg>
        </div>

        {/* Bottom: Variation Grid */}
        <div className="variations-container">
            <div className="variations-header">
                <h3>Choose Your Preferred Snapping Option</h3>
                <p>Click to select, <kbd>Shift</kbd>+Click for fullscreen. Press <kbd>I</kbd> to accept or <kbd>J</kbd> to reject</p>
            </div>
            <div className="variations-grid">
                {proposal.variations && proposal.variations.map((variation, idx) => (
                    <div
                        key={idx}
                        className={`variation-card ${selectedVariation === idx ? 'selected' : ''}`}
                        onClick={(e) => {
                            if (e.shiftKey) {
                                setFullscreenVariation(idx);
                            } else {
                                setSelectedVariation(idx);
                            }
                        }}
                    >
                        <div className="variation-preview">
                            <svg 
                                viewBox={viewBoxStr}
                                preserveAspectRatio="xMidYMid meet"
                                style={{ width: '100%', height: '100%' }}
                            >
                                <image 
                                    href={imageUrl} 
                                    x="0" 
                                    y="0" 
                                    width={proposal.width} 
                                    height={proposal.height} 
                                />
                                {renderPoly(proposal.original_segmentation, "#ff3e3e", true, 1)}
                                {renderVariationPoly(variation.new_points)}
                            </svg>
                        </div>
                        <div className="variation-name">{variation.name}</div>
                        {selectedVariation === idx && <div className="selected-badge">✓ Selected</div>}
                    </div>
                ))}
            </div>
        </div>

        <div className="overlay-instructions">
            <div className="key-hint"><kbd>↑</kbd><kbd>↓</kbd> Cycle options</div>
            <div className="key-hint"><kbd>←</kbd> Skip</div>
            <div className="key-hint"><kbd>→</kbd> Next {selectedVariation !== null && `(Accept ${proposal.variations[selectedVariation].name})`}</div>
            <div className="key-hint"><kbd>I</kbd> Accept</div>
            <div className="key-hint"><kbd>J</kbd> Skip</div>
        </div>
      </main>

      <footer className="info-bar">
        <div className="info-item">
            <label>Image</label>
            <span>{proposal.file_name}</span>
        </div>
        <div className="info-item">
            <label>Annotation ID</label>
            <span>{proposal.annotation_id}</span>
        </div>
        <div className="info-item">
            <label>Progress</label>
            <span>{progress.current} / {progress.total}</span>
        </div>
        <div className="info-item">
            <label>Options</label>
            <span>{proposal.variations ? proposal.variations.length : 0}</span>
        </div>
      </footer>

      {/* Fullscreen Modal */}
      {fullscreenVariation !== null && proposal.variations && (
        <div
          className="fullscreen-modal"
          onClick={() => setFullscreenVariation(null)}
        >
          <div className="fullscreen-content" onClick={(e) => e.stopPropagation()}>
            <button
              className="fullscreen-close"
              onClick={() => setFullscreenVariation(null)}
            >
              ✕
            </button>
            <div className="fullscreen-title">
              {proposal.variations[fullscreenVariation].name}
            </div>
            <svg
              viewBox={viewBoxStr}
              preserveAspectRatio="xMidYMid meet"
              className="fullscreen-svg"
            >
              <image
                href={imageUrl}
                x="0"
                y="0"
                width={proposal.width}
                height={proposal.height}
              />
              {renderPoly(proposal.original_segmentation, "#ff3e3e", true, 1.5)}
              {renderVariationPoly(proposal.variations[fullscreenVariation].new_points)}
            </svg>
            <div className="fullscreen-hint">
              <kbd>↑</kbd><kbd>↓</kbd> Cycle options | <kbd>I</kbd> Accept | <kbd>J</kbd> Skip | <kbd>Esc</kbd> Close
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
