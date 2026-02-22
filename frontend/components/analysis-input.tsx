"use client"

import { useState, useRef, useCallback } from "react"
import { FileText, Mic, ArrowRight, Upload, X, File, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"

interface AnalysisInputFormProps {
  onSubmit: (formData: FormData) => void
  isLoading: boolean
}

function FileDropZone({
  accept,
  label,
  description,
  icon: Icon,
  file,
  onFile,
  onClear,
  id,
}: {
  accept: string
  label: string
  description: string
  icon: React.ElementType
  file: File | null
  onFile: (file: File) => void
  onClear: () => void
  id: string
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragOver, setIsDragOver] = useState(false)

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleDragIn = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(true)
  }, [])

  const handleDragOut = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragOver(false)
      const droppedFile = e.dataTransfer.files?.[0]
      if (droppedFile) {
        const ext = droppedFile.name.split(".").pop()?.toLowerCase()
        const acceptExts = accept.split(",").map((a) => a.trim().replace(".", ""))
        if (ext && acceptExts.includes(ext)) {
          onFile(droppedFile)
        }
      }
    },
    [accept, onFile]
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const selected = e.target.files?.[0]
      if (selected) onFile(selected)
    },
    [onFile]
  )

  function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  if (file) {
    return (
      <div className="rounded-lg border border-primary/30 bg-primary/5 p-4 flex items-center gap-4">
        <div className="h-10 w-10 rounded-md bg-primary/10 flex items-center justify-center shrink-0">
          <File className="h-5 w-5 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-foreground truncate">{file.name}</p>
          <p className="text-xs text-muted-foreground font-mono">
            {formatSize(file.size)}
          </p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onClear}
          className="shrink-0 h-8 w-8 text-muted-foreground hover:text-destructive"
          aria-label={`Remove ${file.name}`}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    )
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onDragEnter={handleDragIn}
      onDragLeave={handleDragOut}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") inputRef.current?.click()
      }}
      className={`
        rounded-lg border-2 border-dashed p-8 flex flex-col items-center gap-3 cursor-pointer transition-colors
        ${isDragOver ? "border-primary bg-primary/5" : "border-border hover:border-muted-foreground/40 hover:bg-secondary/50"}
      `}
    >
      <input
        ref={inputRef}
        id={id}
        type="file"
        accept={accept}
        onChange={handleFileSelect}
        className="sr-only"
        aria-label={label}
      />
      <div className={`h-12 w-12 rounded-full flex items-center justify-center transition-colors ${isDragOver ? "bg-primary/10" : "bg-secondary"}`}>
        <Icon className={`h-6 w-6 ${isDragOver ? "text-primary" : "text-muted-foreground"}`} />
      </div>
      <div className="text-center">
        <p className="text-sm text-foreground">
          <span className="text-primary font-medium">Click to upload</span>{" "}
          or drag and drop
        </p>
        <p className="text-xs text-muted-foreground mt-1 font-mono">{description}</p>
      </div>
      <div className="flex items-center gap-2 mt-1">
        <Upload className="h-3.5 w-3.5 text-muted-foreground" />
        <span className="text-xs text-muted-foreground font-mono uppercase tracking-wider">{label}</span>
      </div>
    </div>
  )
}

export function AnalysisInputForm({ onSubmit, isLoading }: AnalysisInputFormProps) {
  const [textFile, setTextFile] = useState<File | null>(null)
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [reportFile, setReportFile] = useState<File | null>(null)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!textFile || !audioFile || !reportFile) return

    const formData = new FormData()
    formData.append("text_file", textFile)
    formData.append("audio_file", audioFile)
    formData.append("report_file", reportFile)

    onSubmit(formData)
  }

  const canSubmit = textFile && audioFile && reportFile && !isLoading

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="flex flex-col gap-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground font-mono flex items-center gap-2">
            <FileText className="h-3.5 w-3.5" />
            Text Data
          </Label>
          <FileDropZone
            id="text-upload"
            accept=".txt"
            label="TXT files accepted"
            description="Text transcripts (.txt)"
            icon={FileText}
            file={textFile}
            onFile={setTextFile}
            onClear={() => setTextFile(null)}
          />
        </div>

        <div className="flex flex-col gap-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground font-mono flex items-center gap-2">
            <Mic className="h-3.5 w-3.5" />
            Audio Data
          </Label>
          <FileDropZone
            id="audio-upload"
            accept=".mp3,.csv"
            label="MP3 / CSV accepted"
            description="Earnings call audio (.mp3)"
            icon={Mic}
            file={audioFile}
            onFile={setAudioFile}
            onClear={() => setAudioFile(null)}
          />
        </div>

        <div className="flex flex-col gap-2">
          <Label className="text-xs uppercase tracking-wider text-muted-foreground font-mono flex items-center gap-2">
            <BookOpen className="h-3.5 w-3.5" />
            Quarterly Report
          </Label>
          <FileDropZone
            id="report-upload"
            accept=".pdf"
            label="PDF files accepted"
            description="Company quarterly report (.pdf)"
            icon={BookOpen}
            file={reportFile}
            onFile={setReportFile}
            onClear={() => setReportFile(null)}
          />
        </div>
      </div>

      <Button
        type="submit"
        disabled={!canSubmit}
        size="lg"
        className="w-full bg-primary text-primary-foreground hover:bg-primary/90 font-mono text-xs uppercase tracking-widest gap-2 h-12"
      >
        {isLoading ? (
          <span className="flex items-center gap-2">
            Analyzing Fraud Signals...
          </span>
        ) : (
          <>
            Run Fraud Signal Analysis
            <ArrowRight className="h-4 w-4" />
          </>
        )}
      </Button>
    </form>
  )
}
