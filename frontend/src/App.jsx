import { useState } from 'react'
import './App.css'

function App() {
  const [question, setQuestion] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const API_URL = import.meta.env.VITE_API_URL
  const askQuestion = async () => {
    if (!question.trim()) {
      alert("Please type a question")
      return
    }

    setLoading(true)
    setError('')
    setResults([])

    try {
      const response = await fetch(`${API_URL}/ask?q=${encodeURIComponent(question)}`)

      if (!response.ok) {
        throw new Error("Failed to get response from server")
      }

      const data = await response.json()
      setResults(data.matches || [])
    } catch (err) {
      setError("Failed to connect to the server. Please try again later.")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      askQuestion()
    }
  }

  return (
    <div className="container">
      <div className="header">
        <h1>Indian Banking FAQ Bot</h1>
        <p>Ask any question related to banking in India</p>
      </div>

      <div className="chat-area">
        <div className="input-area">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="e.g. What is UPI?"
          />
          <button onClick={askQuestion} disabled={loading}>
            {loading ? "Searching..." : "Ask"}
          </button>
        </div>

        {error && <div className="error">{error}</div>}

        <div className="results">
          {results.map((match, index) => (
            <div className="result" key={index}>
              <div className="score">Match {index + 1} • Score: {match.score}</div>
              <div className="matched-q">{match.matched_question}</div>
              <div className="answer">{match.answer}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default App