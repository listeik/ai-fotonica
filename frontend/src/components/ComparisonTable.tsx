import { ArrowRight, ChartColumn, FolderArchive, LoaderCircle } from 'lucide-react'
import type { Method, Run } from '../api'
import { format, formatTime, methods } from '../workspace'

function Meter({ value }: { value: number }) {
  const width = Math.min(100, Math.abs(value) * 100)
  return <span className="meter">
    <span>{format(value)}</span>
    <span className="meter-track" aria-hidden="true">
      <i className={value < 0 ? 'negative' : ''} style={{ width: width + '%' }} />
    </span>
  </span>
}

export function ComparisonTable({ runs, active, busy, exporting, onSelect, onExport }: {
  runs: Run[]
  active: Method | null
  busy: boolean
  exporting: boolean
  onSelect: (method: Method) => void
  onExport: () => void
}) {
  if (!runs.length) return <section className="panel empty-state">
    <span className="icon-tile"><ChartColumn size={25} /></span>
    <h2>Пока нет расчётов для сравнения</h2>
    <p>Запустите синтез на текущем наборе целей. Каждый метод получит свою строку.</p>
    <a className="button secondary" href="#settings">К настройкам <ArrowRight size={16} /></a>
  </section>

  return <div className="panel comparison-panel">
    <div className="comparison-toolbar">
      <p className="muted text-sm">Выберите метод в таблице, чтобы увидеть его восстановление.</p>
      <button type="button" className="button primary" disabled={busy} onClick={onExport}>
        {exporting ? <LoaderCircle size={17} className="spin" /> : <FolderArchive size={17} />}
        {exporting ? 'Сборка архива…' : 'Экспорт ZIP'}
      </button>
    </div>
    <div className="table-scroll" tabIndex={0} role="region" aria-label="Таблица сравнения методов">
      <table>
        <thead><tr>
          <th scope="col">Метод</th><th scope="col">PCC, средняя</th><th scope="col">SSIM, средняя</th>
          <th scope="col">PCC, мин.</th><th scope="col">SSIM, мин.</th>
          <th scope="col">Итерации</th><th scope="col">Синтез и восстановление</th>
        </tr></thead>
        <tbody>{runs.map(run => <tr key={run.method} className={active === run.method ? 'selected-row' : ''}>
          <th scope="row"><button className="chip selectable" type="button" aria-pressed={active === run.method}
            onClick={() => onSelect(run.method)}>{run.smart ? 'Smart · ' : ''}{methods[run.method].name}</button></th>
          <td><Meter value={run.metrics.mean_pcc} /></td><td><Meter value={run.metrics.mean_ssim} /></td>
          <td>{format(run.metrics.worst_pcc)}</td><td>{format(run.metrics.worst_ssim)}</td>
          <td>{run.iterations}</td><td>{formatTime(run.seconds)}</td>
        </tr>)}</tbody>
      </table>
    </div>
  </div>
}
