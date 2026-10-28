import { useEffect, useState } from 'react'
import { CalendarRange, CopyMinus, Database, Plane } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from '@/components/ui/chart'
import { Skeleton } from '@/components/ui/skeleton'
import { HBarChart } from '@/components/charts'
import { BarList, StatCard } from '@/components/shared'
import { api, type Dataset } from '@/lib/api'

const fmt = (n: number) => n.toLocaleString('en-US')
const title = (s: string) => s.toLowerCase().replace(/(^|[\s/-])\w/g, (m) => m.toUpperCase())

export default function DatasetPage() {
  const [d, setD] = useState<Dataset | null>(null)
  useEffect(() => { api.dataset().then(setD).catch(() => setD(null)) }, [])
  if (!d) return <div className="grid gap-4 md:grid-cols-4">{Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="h-32" />)}</div>

  const c = d.counts as Record<string, number>
  const split = d.split_sizes
  const yearConfig: ChartConfig = { B737: { label: 'Boeing 737', color: 'var(--series-1)' }, A320: { label: 'A320 family', color: 'var(--series-2)' } }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard label="Reports used" icon={<Database className="size-3.5" />} value={fmt(c.final_rows)}
                  hint={`from ${fmt(c.filtered_rows)} A320 + 737 reports, 1995-2025`} />
        <StatCard label="Duplicates removed" icon={<CopyMinus className="size-3.5" />} value={fmt(c.dropped_duplicates)}
                  hint={`${fmt(c.dropped_cross_year_duplicates)} were later-year copies that would leak into testing`} />
        <StatCard label="Time-based split" icon={<CalendarRange className="size-3.5" />}
                  value={`${fmt(split.train)} / ${fmt(split.val)} / ${fmt(split.test)}`}
                  hint="train ≤ 2023 · validation 2024 · test 2025" />
        <StatCard label="Fleets" icon={<Plane className="size-3.5" />} value="A320 family · Boeing 737"
                  hint="A319, A320, A321 (ceo/neo) · 737 Classic, NG, MAX" />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Reports per year</CardTitle>
          <CardDescription>After cleaning and deduplication. Models train on 1995-2023, are tuned on 2024 and tested on 2025.</CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={yearConfig} className="h-72 w-full">
            <BarChart data={d.per_year}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="file_year" tickLine={false} axisLine={false} interval={2} />
              <YAxis tickLine={false} axisLine={false} width={48} tickFormatter={(v) => `${Number(v) / 1000}k`} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent />} />
              <Bar isAnimationActive={false} dataKey="B737" stackId="a" fill="var(--color-B737)" />
              <Bar isAnimationActive={false} dataKey="A320" stackId="a" fill="var(--color-A320)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Top systems (ATA chapters)</CardTitle>
            <CardDescription>Share of each fleet's training reports. Structure (53 fuselage) dominates, which is why macro-F1 is the headline metric.</CardDescription>
          </CardHeader>
          <CardContent>
            <HBarChart data={d.ata_share.map((r) => ({ chapter: `${r.ata2} ${title(String(r.title))}`, B737: Number(r.B737), A320: Number(r.A320) }))}
                       category="chapter" format={(v) => `${(v * 100).toFixed(0)}%`}
                       series={[{ key: 'B737', label: 'Boeing 737' }, { key: 'A320', label: 'A320 family' }]} height={620} />
          </CardContent>
        </Card>
        <div className="grid gap-6">
          <Card>
            <CardHeader><CardTitle>Most common part conditions</CardTitle><CardDescription>All years.</CardDescription></CardHeader>
            <CardContent>
              <BarList format={fmt} items={d.conditions.map((r) => ({ label: title(String(r.condition)), value: Number(r.reports) }))} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Most cited manual references</CardTitle>
              <CardDescription>Generic structural references dominate, so AeroFix ranks references by specificity.</CardDescription>
            </CardHeader>
            <CardContent>
              <BarList color="var(--series-3)" format={fmt}
                       items={d.references.map((r) => ({ label: <span className="font-mono">{String(r.reference)}</span>, value: Number(r.citations) }))} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
