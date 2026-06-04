import type { ReactNode } from 'react'

interface CardProps {
  title: string
  subtitle?: string
  children: ReactNode
  className?: string
  accent?: boolean
  processing?: boolean
}

export function Card({
  title,
  subtitle,
  children,
  className = '',
  accent = false,
  processing = false,
}: CardProps) {
  return (
    <section
      className={`card-shadow relative overflow-hidden rounded-2xl border bg-white transition-all duration-300 hover:shadow-md ${
        accent ? 'border-brand-200' : 'border-emerald-100/80'
      } ${processing ? 'shimmer-surface border-brand-300' : ''} ${className}`}
    >
      {processing && (
        <div className="absolute inset-x-0 top-0 h-0.5 overflow-hidden bg-brand-100">
          <div className="progress-indeterminate h-full w-1/3 rounded-full bg-brand-500" />
        </div>
      )}
      <div className="p-5">
        <div className="mb-4">
          <h2 className="text-sm font-semibold tracking-tight text-brand-800">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs text-muted">{subtitle}</p>}
        </div>
        {children}
      </div>
    </section>
  )
}

interface BadgeProps {
  children: ReactNode
  variant?: 'default' | 'success' | 'running' | 'muted' | 'error'
}

export function Badge({ children, variant = 'default' }: BadgeProps) {
  const styles = {
    default: 'bg-brand-50 text-brand-700 border-brand-200',
    success: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    running: 'bg-brand-100 text-brand-800 border-brand-300 pulse-ring-active',
    muted: 'bg-slate-50 text-slate-600 border-slate-200',
    error: 'bg-red-50 text-red-700 border-red-200',
  }
  return (
    <span
      className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-medium uppercase tracking-wide transition-colors duration-300 ${styles[variant]}`}
    >
      {children}
    </span>
  )
}

export function ShimmerLines({ rows = 3 }: { rows?: number }) {
  return (
    <div className="space-y-2.5">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className="shimmer-bar h-3 rounded-full"
          style={{ width: `${88 - i * 12}%`, animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  )
}
