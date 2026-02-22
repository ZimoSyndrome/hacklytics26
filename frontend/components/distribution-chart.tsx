"use client"

import {
  ComposedChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  type TooltipProps,
} from "recharts"
import type { AnalysisResult, DistributionPoint } from "@/lib/types"

const riskColors: Record<string, string> = {
  critical: "hsl(0 72% 51%)",
  high:     "hsl(38 92% 50%)",
  elevated: "hsl(45 93% 47%)",
  moderate: "hsl(142 71% 45%)",
  low:      "hsl(142 71% 45%)",
}

const horizonYears: Record<string, string> = {
  "4Q Ahead":  "1 year",
  "8Q Ahead":  "2 years",
  "16Q Ahead": "4 years",
}

function probToRiskColor(p: number): string {
  if (p >= 0.65) return riskColors.high
  if (p >= 0.50) return riskColors.elevated
  if (p >= 0.30) return riskColors.moderate
  return riskColors.low
}

function CustomTooltip({ active, payload, label }: TooltipProps<number, string>) {
  if (!active || !payload?.length) return null
  const value = payload[0]?.value ?? 0
  return (
    <div className="rounded-md border border-border bg-popover px-3 py-2 shadow-md">
      <p className="text-xs font-mono font-semibold text-foreground mb-1">{label}</p>
      <p className="text-xs font-mono text-primary">
        Fraud Probability: {(value * 100).toFixed(1)}%
      </p>
    </div>
  )
}

interface DistributionChartProps {
  result: AnalysisResult
}

export function DistributionChart({ result }: DistributionChartProps) {
  const riskColor = riskColors[result.riskLevel] ?? "hsl(var(--primary))"

  return (
    <div className="rounded-lg border border-border bg-card p-6 flex flex-col gap-5">
      {/* Header */}
      <div>
        <h3 className="text-xs uppercase tracking-wider text-muted-foreground font-mono">
          Fraud Probability by Horizon
        </h3>
        <p className="text-lg font-semibold font-mono text-foreground mt-1">
          Peak:{" "}
          <span style={{ color: riskColor }}>
            {(Math.max(...result.distribution.map((d) => d.probability)) * 100).toFixed(1)}%
          </span>
          <span className="text-xs text-muted-foreground ml-2 uppercase">
            {result.riskLevel}
          </span>
        </p>
      </div>

      {/* Per-horizon stat tiles */}
      <div className="grid grid-cols-3 gap-3">
        {result.distribution.map((d: DistributionPoint) => {
          const color = probToRiskColor(d.probability)
          const year  = horizonYears[d.quarter] ?? ""
          return (
            <div
              key={d.quarter}
              className="rounded-md border border-border bg-secondary/40 px-4 py-3 flex flex-col gap-0.5"
            >
              <span className="text-xs font-mono text-muted-foreground uppercase tracking-wider">
                {d.quarter}
              </span>
              <span
                className="text-3xl font-mono font-bold tabular-nums"
                style={{ color }}
              >
                {(d.probability * 100).toFixed(1)}%
              </span>
              {year && (
                <span className="text-xs font-mono text-muted-foreground">{year}</span>
              )}
            </div>
          )
        })}
      </div>

      {/* Chart */}
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={result.distribution}
            margin={{ top: 12, right: 16, left: 0, bottom: 8 }}
          >
            <defs>
              <linearGradient id="gradProb" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={riskColor} stopOpacity={0.35} />
                <stop offset="95%" stopColor={riskColor} stopOpacity={0.02} />
              </linearGradient>
            </defs>

            <CartesianGrid
              strokeDasharray="3 3"
              stroke="hsl(var(--border))"
              opacity={0.35}
            />

            <XAxis
              dataKey="quarter"
              tick={{ fontSize: 11, fill: "hsl(var(--muted-foreground))", fontFamily: "monospace" }}
              tickLine={false}
              axisLine={{ stroke: "hsl(var(--border))" }}
              interval={0}
            />

            <YAxis
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "hsl(var(--muted-foreground))", fontFamily: "monospace" }}
              tickLine={false}
              axisLine={{ stroke: "hsl(var(--border))" }}
              width={48}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />

            <Tooltip
              content={<CustomTooltip />}
              cursor={{ stroke: "hsl(var(--muted-foreground))", strokeWidth: 1, strokeDasharray: "3 3" }}
            />

            <Area
              type="monotone"
              dataKey="probability"
              stroke={riskColor}
              strokeWidth={2.5}
              fill="url(#gradProb)"
              dot={{ r: 5, fill: riskColor, stroke: "hsl(var(--background))", strokeWidth: 2 }}
              activeDot={{ r: 6 }}
              isAnimationActive={true}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
