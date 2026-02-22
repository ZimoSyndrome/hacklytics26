"use client"

import type { AnalysisResult } from "@/lib/types"
import { Activity, Mic, FileText, Layers } from "lucide-react"

const MODALITY_LABELS: Record<string, { label: string; Icon: React.ElementType }> = {
  text:        { label: "Text-only",   Icon: FileText },
  audio:       { label: "Audio-only",  Icon: Mic },
  late_fusion: { label: "Late Fusion", Icon: Layers },
}

interface ModelInsightsProps {
  result: AnalysisResult
}

export function ModelInsights({ result }: ModelInsightsProps) {
  const { topFeatures = [], modelInfo } = result

  if (!topFeatures.length && !modelInfo) return null

  const modality = MODALITY_LABELS[modelInfo?.modality ?? ""] ?? { label: modelInfo?.modality ?? "—", Icon: Activity }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      {/* Top fraud signals */}
      {topFeatures.length > 0 && (
        <div className="rounded-lg border border-border bg-card p-5 flex flex-col gap-4">
          <div>
            <h3 className="text-xs uppercase tracking-wider text-muted-foreground font-mono">
              Top Fraud Signals
            </h3>
            <p className="text-sm font-semibold text-foreground mt-0.5">
              Voice &amp; language indicators (gain importance)
            </p>
          </div>

          <ul className="flex flex-col gap-3">
            {topFeatures.map((f, idx) => (
              <li key={f.name} className="flex flex-col gap-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-foreground leading-tight">
                    <span className="text-muted-foreground mr-1.5">{idx + 1}.</span>
                    {f.label || f.name}
                  </span>
                  <span className="text-xs font-mono text-muted-foreground ml-2 shrink-0">
                    {(f.importance * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="h-1.5 rounded-full bg-secondary overflow-hidden">
                  <div
                    className="h-full rounded-full bg-primary transition-all duration-500"
                    style={{ width: `${f.importance * 100}%` }}
                  />
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Model performance */}
      {modelInfo && (
        <div className="rounded-lg border border-border bg-card p-5 flex flex-col gap-4">
          <div>
            <h3 className="text-xs uppercase tracking-wider text-muted-foreground font-mono">
              Model Performance
            </h3>
            <p className="text-sm font-semibold text-foreground mt-0.5">
              Trained on SEC AAER enforcement data
            </p>
          </div>

          {/* Modality badge */}
          <div className="flex items-center gap-2">
            <modality.Icon className="h-4 w-4 text-primary" />
            <span className="text-xs font-mono text-foreground">{modality.label}</span>
            <span className="ml-auto text-xs font-mono text-muted-foreground">
              {modelInfo.horizon}Q look-ahead
            </span>
          </div>

          <div className="flex flex-col gap-3">
            {/* PR-AUC */}
            <div className="rounded-md border border-border bg-secondary/40 px-4 py-3">
              <div className="flex items-baseline justify-between">
                <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
                  PR-AUC
                </span>
                <span className="text-xl font-mono font-bold text-foreground tabular-nums">
                  {modelInfo.pr_auc.toFixed(4)}
                </span>
              </div>
              <p className="text-xs text-muted-foreground font-mono mt-1">
                95% CI&nbsp;
                [{modelInfo.pr_ci_lower.toFixed(4)} – {modelInfo.pr_ci_upper.toFixed(4)}]
              </p>
              <p className="text-xs text-muted-foreground font-mono mt-0.5">
                Baseline (random) ≈ 0.014 (fraud rate)
              </p>
            </div>

            {/* Brier score */}
            <div className="rounded-md border border-border bg-secondary/40 px-4 py-3">
              <div className="flex items-baseline justify-between">
                <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
                  Brier Score
                </span>
                <span className="text-xl font-mono font-bold text-foreground tabular-nums">
                  {modelInfo.brier.toFixed(4)}
                </span>
              </div>
              <p className="text-xs text-muted-foreground font-mono mt-1">
                Calibrated probability error (lower = better)
              </p>
            </div>

            {/* Methodology note */}
            <p className="text-xs text-muted-foreground font-mono leading-relaxed">
              LightGBM + isotonic calibration · 7-stat temporal features · ticker-cluster bootstrap CI · temporal company-grouped split
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
