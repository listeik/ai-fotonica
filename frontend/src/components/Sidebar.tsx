import { Atom, ChartColumn, Images, Layers3, LayoutDashboard, SlidersHorizontal } from 'lucide-react'
import { views, type View } from '../workspace'

const navigation = [
  { id: 'overview', icon: LayoutDashboard, short: 'Обзор' },
  { id: 'targets', icon: Images, short: 'Цели' },
  { id: 'settings', icon: SlidersHorizontal, short: 'Расчёт' },
  { id: 'restoration', icon: Layers3, short: 'Результаты' },
  { id: 'comparison', icon: ChartColumn, short: 'Сравнение' },
] as const

export function Sidebar({ active, device, failed }: { active: View; device?: string; failed: boolean }) {
  const status = device ? 'Подключён' : failed ? 'Недоступен' : 'Подключение…'
  const state = device ? 'online' : failed ? 'offline' : 'pending'

  return <aside className="sidebar">
    <div className="sidebar-top">
      <a href="#overview" className="brand" aria-label="AI Photonica · Обзор">
        <span className="brand-icon"><Atom size={23} strokeWidth={1.75} /></span>
        <span>AI Photonica<small>ЛАБОРАТОРИЯ ДОЭ</small></span>
      </a>
      <span className={'mobile-status ' + state} role="status" aria-label={'API · ' + status} title={status}>
        <i className="status-dot" /> API
      </span>
    </div>
    <div className="sidebar-label">Рабочая область</div>
    <nav className="navigation" aria-label="Разделы">
      {navigation.map(({ id, icon: Icon, short }) => <a key={id} href={'#' + id}
        className="navigation-link" aria-label={views[id].label} aria-current={active === id ? 'page' : undefined}>
        <Icon size={18} strokeWidth={1.8} aria-hidden="true" />
        <span className="navigation-label">{views[id].label}</span>
        <span className="navigation-short" aria-hidden="true">{short}</span>
        <span className="navigation-index" aria-hidden="true">{views[id].number}</span>
      </a>)}
    </nav>
    <div className={'sidebar-status ' + state} role="status">
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold">Статус API</span>
        {device && <span className="device-chip">{device.toUpperCase()}</span>}
      </div>
      <span className="mt-2 flex items-center gap-2 text-xs"><i className="status-dot" />{status}</span>
    </div>
  </aside>
}
