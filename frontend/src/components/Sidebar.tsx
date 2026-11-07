import { Atom, LayoutDashboard } from 'lucide-react'

export function Sidebar({ device, failed }: { device?: string; failed: boolean }) {
  const status = device ? 'Подключён' : failed ? 'Недоступен' : 'Подключение…'
  const state = device ? 'online' : failed ? 'offline' : 'pending'

  return <aside className="sidebar">
    <div className="sidebar-top">
      <a href="#workspace" className="brand" aria-label="AI Photonica · Синтез ДОЭ">
        <span className="brand-icon"><Atom size={23} strokeWidth={1.75} /></span>
        <span>AI Photonica<small>ЛАБОРАТОРИЯ ДОЭ</small></span>
      </a>
      <span className={'mobile-status ' + state} role="status" aria-label={'API · ' + status} title={status}>
        <i className="status-dot" /> API
      </span>
    </div>
    <div className="sidebar-label">Рабочая область</div>
    <nav className="navigation" aria-label="Разделы">
      <a href="#workspace" className="navigation-link" aria-current="page">
        <LayoutDashboard size={18} strokeWidth={1.8} aria-hidden="true" />
        <span>Синтез ДОЭ</span>
      </a>
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
