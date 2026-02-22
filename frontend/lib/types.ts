import { z } from "zod"

export const distributionPointSchema = z.object({
  quarter: z.string(),      // e.g. "4Q Ahead"
  probability: z.number(),  // fraud likelihood 0–1
})

export const featureImportanceSchema = z.object({
  name: z.string(),
  importance: z.number(),   // 0–1 normalised
  label: z.string(),        // human-readable display label
})

export const modelInfoSchema = z.object({
  horizon: z.number(),      // quarters ahead
  modality: z.string(),     // "text" | "audio" | "late_fusion"
  pr_auc: z.number(),
  pr_ci_lower: z.number(),
  pr_ci_upper: z.number(),
  brier: z.number(),
})

export const analysisResultSchema = z.object({
  overallRiskScore: z.number(),  // 0–100
  riskLevel: z.enum(["critical", "high", "elevated", "moderate", "low"]),
  distribution: z.array(distributionPointSchema),
  topFeatures: z.array(featureImportanceSchema).optional().default([]),
  modelInfo: modelInfoSchema.nullable().optional(),
})

export type DistributionPoint  = z.infer<typeof distributionPointSchema>
export type FeatureImportance  = z.infer<typeof featureImportanceSchema>
export type ModelInfo          = z.infer<typeof modelInfoSchema>
export type AnalysisResult     = z.infer<typeof analysisResultSchema>
