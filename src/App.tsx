import React, { useState } from 'react'
import UploadZone from './components/UploadZone'
import ResultCard from './components/ResultCard'
import LoadingSpinner from './components/LoadingSpinner'
import { Baby, Heart, Sparkles } from 'lucide-react'

interface UploadedFile {
  file: File
  preview: string
}

function App() {
  const [momPhoto, setMomPhoto] = useState<UploadedFile | null>(null)
  const [dadPhoto, setDadPhoto] = useState<UploadedFile | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [result, setResult] = useState<{ imageUrl: string; processingTime: string } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleMomUpload = (file: File) => {
    const preview = URL.createObjectURL(file)
    setMomPhoto({ file, preview })
    setError(null)
  }

  const handleDadUpload = (file: File) => {
    const preview = URL.createObjectURL(file)
    setDadPhoto({ file, preview })
    setError(null)
  }

  const handleGenerate = async () => {
    if (!momPhoto || !dadPhoto) return

    setIsGenerating(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('mom_photo', momPhoto.file)
      formData.append('dad_photo', dadPhoto.file)

      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const response = await fetch(`${apiUrl}/api/generate-baby`, {
        method: 'POST',
        body: formData,
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || 'Failed to generate baby face')
      }

      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setIsGenerating(false)
    }
  }

  const handleReset = () => {
    setMomPhoto(null)
    setDadPhoto(null)
    setResult(null)
    setError(null)
  }

  const canGenerate = momPhoto && dadPhoto && !isGenerating

  return (
    <div className="min-h-screen bg-gradient-to-br from-baby-pink to-baby-blue">
      {/* Header */}
      <header className="text-center py-8 px-4">
        <div className="flex items-center justify-center gap-3 mb-4">
          <Baby className="w-8 h-8 text-purple-700" />
          <h1 className="text-3xl md:text-4xl font-bold text-purple-800 font-cute">
            AI Baby Face Generator
          </h1>
          <Sparkles className="w-8 h-8 text-purple-700" />
        </div>
        <p className="text-purple-700 text-lg max-w-md mx-auto">
          Upload photos of mom and dad to see what your future baby might look like! 👶
        </p>
      </header>

      <main className="container mx-auto px-4 pb-8">
        {!result ? (
          <div className="max-w-2xl mx-auto">
            {/* Upload Section */}
            <div className="grid md:grid-cols-2 gap-6 mb-8">
              <div>
                <h2 className="text-xl font-semibold text-purple-800 mb-4 text-center">
                  Upload Mom's Photo 📸
                </h2>
                <UploadZone
                  label="Mom's Photo"
                  onUpload={handleMomUpload}
                  uploaded={!!momPhoto}
                />
              </div>
              <div>
                <h2 className="text-xl font-semibold text-purple-800 mb-4 text-center">
                  Upload Dad's Photo 📸
                </h2>
                <UploadZone
                  label="Dad's Photo"
                  onUpload={handleDadUpload}
                  uploaded={!!dadPhoto}
                />
              </div>
            </div>

            {/* Generate Button */}
            <div className="text-center mb-8">
              <button
                onClick={handleGenerate}
                disabled={!canGenerate}
                className="btn-primary text-xl px-12 py-4 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isGenerating ? (
                  <div className="flex items-center gap-3">
                    <LoadingSpinner />
                    Creating your baby... 🍼
                  </div>
                ) : (
                  <>
                    Generate Our Baby! 👶
                    <Heart className="w-6 h-6 ml-2 inline" />
                  </>
                )}
              </button>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-xl text-center">
                {error}
              </div>
            )}
          </div>
        ) : (
          <ResultCard
            result={result}
            onTryAgain={handleReset}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="text-center py-6 px-4">
        <p className="text-purple-600 text-sm">
          Made with <Heart className="w-4 h-4 inline text-red-500" /> for fun — not a genetic prediction.
        </p>
        <p className="text-purple-500 text-xs mt-2">
          For entertainment purposes only. Photos are automatically deleted after processing.
        </p>
      </footer>
    </div>
  )
}

export default App
