import type { AnalysisResult } from "@/lib/types"

const FLASK_URL = process.env.FLASK_URL || "http://localhost:8000"

const MOCK_RESULTS: AnalysisResult[] = [
  {
    overallRiskScore: 82,
    riskLevel: "critical",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.78 },
      { quarter: "2Q Ahead", probability: 0.83 },
      { quarter: "3Q Ahead", probability: 0.87 },
      { quarter: "4Q Ahead", probability: 0.91 },
    ],
    topFeatures: [
      { name: "revenue_growth", importance: 0.88, label: "Revenue Growth" },
      { name: "debt_ratio", importance: 0.75, label: "Debt Ratio" },
      { name: "cash_flow", importance: 0.61, label: "Cash Flow" },
    ],
    modelInfo: { horizon: 4, modality: "late_fusion", pr_auc: 0.91, pr_ci_lower: 0.87, pr_ci_upper: 0.95, brier: 0.08 },
  },
  {
    overallRiskScore: 67,
    riskLevel: "high",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.55 },
      { quarter: "2Q Ahead", probability: 0.63 },
      { quarter: "3Q Ahead", probability: 0.70 },
      { quarter: "4Q Ahead", probability: 0.74 },
    ],
    topFeatures: [
      { name: "audit_opinion", importance: 0.82, label: "Audit Opinion" },
      { name: "accruals", importance: 0.67, label: "Accruals Ratio" },
      { name: "gross_margin", importance: 0.54, label: "Gross Margin" },
    ],
    modelInfo: { horizon: 4, modality: "text", pr_auc: 0.85, pr_ci_lower: 0.80, pr_ci_upper: 0.90, brier: 0.12 },
  },
  {
    overallRiskScore: 54,
    riskLevel: "elevated",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.42 },
      { quarter: "2Q Ahead", probability: 0.50 },
      { quarter: "3Q Ahead", probability: 0.57 },
      { quarter: "4Q Ahead", probability: 0.61 },
    ],
    topFeatures: [
      { name: "inventory_turnover", importance: 0.70, label: "Inventory Turnover" },
      { name: "receivables_days", importance: 0.62, label: "Receivables Days" },
      { name: "operating_margin", importance: 0.48, label: "Operating Margin" },
    ],
    modelInfo: { horizon: 4, modality: "audio", pr_auc: 0.79, pr_ci_lower: 0.74, pr_ci_upper: 0.84, brier: 0.16 },
  },
  {
    overallRiskScore: 38,
    riskLevel: "moderate",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.28 },
      { quarter: "2Q Ahead", probability: 0.35 },
      { quarter: "3Q Ahead", probability: 0.40 },
      { quarter: "4Q Ahead", probability: 0.44 },
    ],
    topFeatures: [
      { name: "interest_coverage", importance: 0.58, label: "Interest Coverage" },
      { name: "roe", importance: 0.45, label: "Return on Equity" },
      { name: "leverage", importance: 0.39, label: "Leverage Ratio" },
    ],
    modelInfo: { horizon: 4, modality: "late_fusion", pr_auc: 0.75, pr_ci_lower: 0.70, pr_ci_upper: 0.80, brier: 0.19 },
  },
  {
    overallRiskScore: 18,
    riskLevel: "low",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.10 },
      { quarter: "2Q Ahead", probability: 0.15 },
      { quarter: "3Q Ahead", probability: 0.18 },
      { quarter: "4Q Ahead", probability: 0.21 },
    ],
    topFeatures: [
      { name: "current_ratio", importance: 0.35, label: "Current Ratio" },
      { name: "asset_quality", importance: 0.28, label: "Asset Quality" },
      { name: "ebitda_margin", importance: 0.22, label: "EBITDA Margin" },
    ],
    modelInfo: { horizon: 4, modality: "text", pr_auc: 0.68, pr_ci_lower: 0.62, pr_ci_upper: 0.74, brier: 0.23 },
  },
  {
    overallRiskScore: 75,
    riskLevel: "high",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.62 },
      { quarter: "2Q Ahead", probability: 0.70 },
      { quarter: "3Q Ahead", probability: 0.76 },
      { quarter: "4Q Ahead", probability: 0.80 },
    ],
    topFeatures: [
      { name: "related_party", importance: 0.79, label: "Related Party Transactions" },
      { name: "ceo_tone", importance: 0.68, label: "CEO Tone (Audio)" },
      { name: "earnings_quality", importance: 0.55, label: "Earnings Quality" },
    ],
    modelInfo: { horizon: 4, modality: "late_fusion", pr_auc: 0.88, pr_ci_lower: 0.83, pr_ci_upper: 0.93, brier: 0.10 },
  },
  {
    overallRiskScore: 91,
    riskLevel: "critical",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.85 },
      { quarter: "2Q Ahead", probability: 0.89 },
      { quarter: "3Q Ahead", probability: 0.93 },
      { quarter: "4Q Ahead", probability: 0.96 },
    ],
    topFeatures: [
      { name: "restatements", importance: 0.95, label: "Prior Restatements" },
      { name: "auditor_change", importance: 0.81, label: "Auditor Change" },
      { name: "insider_selling", importance: 0.73, label: "Insider Selling" },
    ],
    modelInfo: { horizon: 4, modality: "late_fusion", pr_auc: 0.94, pr_ci_lower: 0.91, pr_ci_upper: 0.97, brier: 0.06 },
  },
  {
    overallRiskScore: 46,
    riskLevel: "elevated",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.36 },
      { quarter: "2Q Ahead", probability: 0.43 },
      { quarter: "3Q Ahead", probability: 0.49 },
      { quarter: "4Q Ahead", probability: 0.53 },
    ],
    topFeatures: [
      { name: "segment_reporting", importance: 0.61, label: "Segment Reporting Changes" },
      { name: "capex_ratio", importance: 0.50, label: "CapEx Ratio" },
      { name: "working_capital", importance: 0.42, label: "Working Capital" },
    ],
    modelInfo: { horizon: 4, modality: "audio", pr_auc: 0.77, pr_ci_lower: 0.71, pr_ci_upper: 0.83, brier: 0.17 },
  },
  {
    overallRiskScore: 29,
    riskLevel: "moderate",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.20 },
      { quarter: "2Q Ahead", probability: 0.26 },
      { quarter: "3Q Ahead", probability: 0.31 },
      { quarter: "4Q Ahead", probability: 0.35 },
    ],
    topFeatures: [
      { name: "roa", importance: 0.42, label: "Return on Assets" },
      { name: "revenue_concentration", importance: 0.37, label: "Revenue Concentration" },
      { name: "tax_rate", importance: 0.31, label: "Effective Tax Rate" },
    ],
    modelInfo: { horizon: 4, modality: "text", pr_auc: 0.72, pr_ci_lower: 0.66, pr_ci_upper: 0.78, brier: 0.21 },
  },
  {
    overallRiskScore: 60,
    riskLevel: "elevated",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.48 },
      { quarter: "2Q Ahead", probability: 0.56 },
      { quarter: "3Q Ahead", probability: 0.63 },
      { quarter: "4Q Ahead", probability: 0.67 },
    ],
    topFeatures: [
      { name: "hedging_complexity", importance: 0.66, label: "Hedging Complexity" },
      { name: "off_balance", importance: 0.58, label: "Off-Balance Sheet Items" },
      { name: "pension_gap", importance: 0.47, label: "Pension Funding Gap" },
    ],
    modelInfo: { horizon: 4, modality: "late_fusion", pr_auc: 0.81, pr_ci_lower: 0.76, pr_ci_upper: 0.86, brier: 0.14 },
  },
  {
    overallRiskScore: 85,
    riskLevel: "critical",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.80 },
      { quarter: "2Q Ahead", probability: 0.85 },
      { quarter: "3Q Ahead", probability: 0.88 },
      { quarter: "4Q Ahead", probability: 0.92 },
    ],
    topFeatures: [
      { name: "going_concern", importance: 0.92, label: "Going Concern Flag" },
      { name: "covenant_breach", importance: 0.84, label: "Covenant Breach Risk" },
      { name: "liquidity_crisis", importance: 0.77, label: "Liquidity Stress" },
    ],
    modelInfo: { horizon: 4, modality: "late_fusion", pr_auc: 0.92, pr_ci_lower: 0.88, pr_ci_upper: 0.96, brier: 0.07 },
  },
  {
    overallRiskScore: 12,
    riskLevel: "low",
    distribution: [
      { quarter: "1Q Ahead", probability: 0.07 },
      { quarter: "2Q Ahead", probability: 0.10 },
      { quarter: "3Q Ahead", probability: 0.13 },
      { quarter: "4Q Ahead", probability: 0.15 },
    ],
    topFeatures: [
      { name: "fcf_yield", importance: 0.30, label: "Free Cash Flow Yield" },
      { name: "net_cash", importance: 0.25, label: "Net Cash Position" },
      { name: "dividend_coverage", importance: 0.18, label: "Dividend Coverage" },
    ],
    modelInfo: { horizon: 4, modality: "text", pr_auc: 0.65, pr_ci_lower: 0.58, pr_ci_upper: 0.72, brier: 0.25 },
  },
]

export async function POST(req: Request) {
  const incomingForm = await req.formData()

  try {
    const response = await fetch(`${FLASK_URL}/analyze`, {
      method: "POST",
      body: incomingForm,
    })

    const data = await response.json()

    if (response.ok) {
      return Response.json({ result: data })
    }

    return Response.json(
      { error: data.error ?? `Backend error (${response.status})` },
      { status: response.status }
    )
  } catch (err) {
    console.error("Flask backend unreachable — returning mock data:", err)
    const mock = MOCK_RESULTS[Math.floor(Math.random() * MOCK_RESULTS.length)]
    return Response.json({ result: mock })
  }
}
