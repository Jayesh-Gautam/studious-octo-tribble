import { Bar, BarChart, CartesianGrid, LabelList, Line, LineChart, XAxis, YAxis } from 'recharts'
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'

export const SERIES = ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)', 'var(--series-5)']

type Datum = Record<string, string | number | null>

/** Horizontal bars, one or more series; categories on the y axis (good for long model names). */
export function HBarChart({ data, category, series, height, max, format = (v: number) => v.toFixed(3) }: {
  data: Datum[]; category: string; series: { key: string; label: string }[]; height?: number; max?: number
  format?: (v: number) => string
}) {
  const config: ChartConfig = Object.fromEntries(series.map((s, i) => [s.key, { label: s.label, color: SERIES[i] }]))
  return (
    <ChartContainer config={config} className="w-full" style={{ height: height ?? Math.max(160, data.length * (28 + 14 * series.length)) }}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 48 }} barGap={2}>
        <CartesianGrid horizontal={false} />
        <XAxis type="number" domain={[0, max ?? 'auto']} tickFormatter={(v) => format(Number(v))} tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey={category} width={210} tickLine={false} axisLine={false} tick={{ fontSize: 12 }} />
        <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
        {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
        {series.map((s) => (
          <Bar key={s.key} isAnimationActive={false} dataKey={s.key} fill={`var(--color-${s.key})`} radius={4} maxBarSize={22}>
            <LabelList dataKey={s.key} position="right" className="fill-foreground" fontSize={11}
                       formatter={(v: unknown) => (typeof v === 'number' ? format(v) : '')} />
          </Bar>
        ))}
      </BarChart>
    </ChartContainer>
  )
}

/** Vertical grouped bars (few categories, e.g. train/test family combinations). */
export function VBarChart({ data, category, series, height = 240, format = (v: number) => v.toFixed(3), domain }: {
  data: Datum[]; category: string; series: { key: string; label: string }[]; height?: number
  format?: (v: number) => string; domain?: [number, number]
}) {
  const config: ChartConfig = Object.fromEntries(series.map((s, i) => [s.key, { label: s.label, color: SERIES[i] }]))
  return (
    <ChartContainer config={config} className="w-full" style={{ height }}>
      <BarChart data={data} margin={{ top: 20 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey={category} tickLine={false} axisLine={false} tick={{ fontSize: 12 }} />
        <YAxis domain={domain ?? [0, 'auto']} tickFormatter={(v) => format(Number(v))} tickLine={false} axisLine={false} width={44} />
        <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
        {series.length > 1 && <ChartLegend content={<ChartLegendContent />} />}
        {series.map((s) => (
          <Bar key={s.key} isAnimationActive={false} dataKey={s.key} fill={`var(--color-${s.key})`} radius={4} maxBarSize={48}>
            <LabelList dataKey={s.key} position="top" className="fill-foreground" fontSize={11}
                       formatter={(v: unknown) => (typeof v === 'number' ? format(v) : '')} />
          </Bar>
        ))}
      </BarChart>
    </ChartContainer>
  )
}

export function LineChartCard({ data, x, series, height = 240, domain, format = (v: number) => v.toFixed(3) }: {
  data: Datum[]; x: string; series: { key: string; label: string }[]; height?: number
  domain?: [number, number]; format?: (v: number) => string
}) {
  const config: ChartConfig = Object.fromEntries(series.map((s, i) => [s.key, { label: s.label, color: SERIES[i] }]))
  return (
    <ChartContainer config={config} className="w-full" style={{ height }}>
      <LineChart data={data} margin={{ top: 12, right: 16 }}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey={x} tickLine={false} axisLine={false} />
        <YAxis domain={domain ?? ['auto', 'auto']} tickFormatter={(v) => format(Number(v))} tickLine={false} axisLine={false} width={48} />
        <ChartTooltip content={<ChartTooltipContent />} />
        <ChartLegend content={<ChartLegendContent />} />
        {series.map((s) => (
          <Line key={s.key} isAnimationActive={false} dataKey={s.key} stroke={`var(--color-${s.key})`} strokeWidth={2} dot={{ r: 4 }} type="monotone" />
        ))}
      </LineChart>
    </ChartContainer>
  )
}
