import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [pendingImages, setPendingImages] = useState([]);
  const [currentImage, setCurrentImage] = useState(null);
  const [loading, setLoading] = useState(false);

  // Backend URL - assume running on port 8000
  const API_URL = 'http://localhost:8000';

  const fetchPending = async () => {
    try {
      const res = await fetch(`${API_URL}/api/pending`);
      const data = await res.json();
      setPendingImages(data);
    } catch (err) {
      console.error("Failed to fetch pending images:", err);
    }
  };

  useEffect(() => {
    fetchPending();
  }, []);

  useEffect(() => {
    if (pendingImages.length > 0) {
      setCurrentImage(pendingImages[0]);
    } else {
      setCurrentImage(null);
    }
  }, [pendingImages]);

  useEffect(() => {
    const handleKeyDown = (event) => {
      // Ignore if loading or no image
      if (loading || !currentImage) return;

      if (event.key === 'j' || event.key === 'J') {
        handleDecision('accept');
      } else if (event.key === 'i' || event.key === 'I') {
        handleDecision('reject');
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    // Cleanup
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [loading, currentImage]); // Dependencies ensure we have fresh state

  const handleDecision = async (decision) => {
    if (!currentImage) return;
    setLoading(true);

    try {
      const endpoint = decision === 'accept' ? 'accept' : 'reject';
      const res = await fetch(`${API_URL}/api/${endpoint}/${currentImage.filename}`, {
        method: 'POST'
      });
      
      if (res.ok) {
        // Remove processed image from list
        setPendingImages(prev => prev.slice(1));
      } else {
        console.error("Action failed");
      }
    } catch (err) {
      console.error("Error submitting decision:", err);
    } finally {
      setLoading(false);
    }
  };

  if (pendingImages.length === 0 && !currentImage) {
    return (
      <div className="container">
        <h1>Segmentation Fixer</h1>
        <div className="empty">No images pending review! Great job.</div>
      </div>
    );
  }

  return (
    <div className="container">
      <h1>Segmentation Fixer Review</h1>
      
      {currentImage && (
        <div className="image-container">
          <h3>Reviewing: {currentImage.stem}</h3>
          <div className="metadata">
            <span><strong>SAM Confidence:</strong> {(currentImage.score * 100).toFixed(1)}%</span>
            <span><strong>Difference (IoU):</strong> {currentImage.iou.toFixed(3)}</span>
          </div>
          <img 
            src={`${API_URL}/images/${currentImage.filename}`} 
            alt="Comparison" 
          />
        </div>
      )}

      <div className="controls">
        <button 
          className="accept" 
          onClick={() => handleDecision('accept')}
          disabled={loading || !currentImage}
        >
          {loading ? 'Processing...' : 'Accept'}
        </button>
        <button 
          className="reject" 
          onClick={() => handleDecision('reject')}
          disabled={loading || !currentImage}
        >
          {loading ? 'Processing...' : 'Reject'}
        </button>
      </div>
      
      <div className="status">
        Pending: {pendingImages.length}
      </div>
    </div>
  )
}

export default App
