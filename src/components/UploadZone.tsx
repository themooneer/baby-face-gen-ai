import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, CheckCircle, Image as ImageIcon } from 'lucide-react'
import Resizer from 'react-image-file-resizer'

interface UploadZoneProps {
  label: string
  onUpload: (file: File) => void
  uploaded: boolean
}

export default function UploadZone({ label, onUpload, uploaded }: UploadZoneProps) {
  const [preview, setPreview] = useState<string | null>(null)

  const resizeFile = (file: File): Promise<File> =>
    new Promise((resolve) => {
      Resizer.imageFileResizer(
        file,
        800, // max width
        800, // max height
        'JPEG',
        80, // quality
        0, // rotation
        (uri) => {
          // Convert data URL to File
          fetch(uri as string)
            .then(res => res.blob())
            .then(blob => {
              const resizedFile = new File([blob], file.name, {
                type: 'image/jpeg',
                lastModified: Date.now(),
              })
              resolve(resizedFile)
            })
        },
        'base64'
      )
    })

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (file) {
      try {
        // Resize the image to optimize for processing
        const resizedFile = await resizeFile(file)
        onUpload(resizedFile)

        // Create preview
        const reader = new FileReader()
        reader.onload = () => setPreview(reader.result as string)
        reader.readAsDataURL(resizedFile)
      } catch (error) {
        console.error('Error processing image:', error)
        // Fallback to original file
        onUpload(file)
        const reader = new FileReader()
        reader.onload = () => setPreview(reader.result as string)
        reader.readAsDataURL(file)
      }
    }
  }, [onUpload])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/jpeg': ['.jpg', '.jpeg'],
      'image/png': ['.png'],
      'image/webp': ['.webp']
    },
    maxSize: 5 * 1024 * 1024, // 5MB
    multiple: false
  })

  return (
    <div
      {...getRootProps()}
      className={`upload-zone ${isDragActive ? 'active' : ''} ${uploaded ? 'border-green-400 bg-green-50' : ''}`}
    >
      <input {...getInputProps()} />

      {preview ? (
        <div className="space-y-3">
          <img
            src={preview}
            alt={label}
            className="mx-auto w-24 h-24 object-cover rounded-full border-4 border-white shadow-lg"
          />
          <div className="flex items-center justify-center gap-2 text-green-600">
            <CheckCircle className="w-5 h-5" />
            <span className="font-semibold">✅ {label} Uploaded</span>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="text-6xl">
            {isDragActive ? '📷' : <ImageIcon className="w-16 h-16 mx-auto text-purple-600" />}
          </div>
          <div>
            <p className="text-lg font-semibold text-purple-700 mb-2">
              {isDragActive ? `Drop ${label} here` : `Upload ${label}`}
            </p>
            <p className="text-sm text-purple-600">
              Drag & drop or click to select
            </p>
            <p className="text-xs text-purple-500 mt-1">
              JPG, PNG, WebP • Max 5MB
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
