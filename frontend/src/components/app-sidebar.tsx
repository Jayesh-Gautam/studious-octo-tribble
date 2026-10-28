import { ArrowLeftRight, BarChart3, Database, Info, MessageSquareText, Plane, Wrench } from 'lucide-react'
import {
  Sidebar, SidebarContent, SidebarFooter, SidebarGroup, SidebarGroupContent, SidebarGroupLabel, SidebarHeader,
  SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarRail,
} from '@/components/ui/sidebar'

export type PageId = 'advisor' | 'performance' | 'dataset' | 'feedback' | 'about'

export const PAGES: { id: PageId; title: string; subtitle: string; icon: typeof Wrench; group: string }[] = [
  { id: 'advisor', title: 'Fault advisor', subtitle: 'Describe a defect, get the likely system, fixes, references and evidence', icon: Wrench, group: 'Operations' },
  { id: 'feedback', title: 'Feedback log', subtitle: 'Technician confirmations and corrections used to improve the models', icon: MessageSquareText, group: 'Operations' },
  { id: 'performance', title: 'Model performance', subtitle: 'How every component scores on the held-out 2025 reports', icon: BarChart3, group: 'Insights' },
  { id: 'dataset', title: 'Dataset', subtitle: 'FAA Service Difficulty Reports, A320 family and Boeing 737, 1995-2025', icon: Database, group: 'Insights' },
  { id: 'about', title: 'About AeroFix', subtitle: 'How the system works, data sources and limitations', icon: Info, group: 'Project' },
]

/** Pages shown in the clean production app (served at /production). */
export const PRODUCTION_PAGES: PageId[] = ['advisor', 'feedback']

export function AppSidebar({ page, onNavigate, production }: {
  page: PageId; onNavigate: (p: PageId) => void; production: boolean
}) {
  const visible = PAGES.filter((p) => !production || PRODUCTION_PAGES.includes(p.id))
  const groups = [...new Set(visible.map((p) => p.group))]
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" onClick={() => onNavigate('advisor')}>
              <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                <Plane className="size-4" />
              </div>
              <div className="grid flex-1 text-left leading-tight">
                <span className="truncate font-semibold">AeroFix</span>
                <span className="truncate text-xs text-muted-foreground">{production ? 'Maintenance assistant' : 'Maintenance fault advisor'}</span>
              </div>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        {groups.map((g) => (
          <SidebarGroup key={g}>
            {!production && <SidebarGroupLabel>{g}</SidebarGroupLabel>}
            <SidebarGroupContent>
              <SidebarMenu>
                {visible.filter((p) => p.group === g).map((p) => (
                  <SidebarMenuItem key={p.id}>
                    <SidebarMenuButton isActive={page === p.id} tooltip={p.title} onClick={() => onNavigate(p.id)}>
                      <p.icon />
                      <span>{p.title}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                ))}
              </SidebarMenu>
            </SidebarGroupContent>
          </SidebarGroup>
        ))}
      </SidebarContent>
      <SidebarFooter>
        {!production && (
          <div className="rounded-md border bg-sidebar-accent/40 p-2 text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">
            <p className="font-medium text-foreground">Fleet: A320 family · Boeing 737</p>
            <p>540,008 FAA reports · local models only</p>
          </div>
        )}
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip={production ? 'Presentation console' : 'Production app'}>
              <a href={production ? '/' : '/production'}>
                <ArrowLeftRight />
                <span>{production ? 'Open presentation console' : 'Open production app'}</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
