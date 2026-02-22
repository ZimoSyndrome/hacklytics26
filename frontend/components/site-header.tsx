import { Shield } from "lucide-react"

export function SiteHeader() {
  return (
    <header className="border-b border-border bg-card">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <Shield className="h-4 w-4 text-primary" />
          <span className="text-sm font-mono font-semibold text-foreground tracking-wider uppercase">
            Fraud Signal Analysis
          </span>
        </div>
      </div>
    </header>
  )
}
