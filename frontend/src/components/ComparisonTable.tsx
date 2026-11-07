import { ArrowRight, ChartColumn, FolderArchive, LoaderCircle } from 'lucide-react'
import { pngUrl, type Method, type Run } from '../api'
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

  const preview = runs.find(run => run.method === active)
  return <div className="grid gap-5 sm:gap-6">
    <section className="panel comparison-panel">
      <div className="comparison-toolbar">
        <div><h2>Метрики и время</h2><p className="muted mt-1 text-sm">Выберите метод в таблице, чтобы сменить предпросмотр.</p></div>
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
            <th scope="col">Итерации</th><th scope="col">Время</th>
          </tr></thead>
          <tbody>{runs.map(run => <tr key={run.method} className={active === run.method ? 'selected-row' : ''}>
            <th scope="row"><button className="chip selectable" type="button" aria-pressed={active === run.method}
              onClick={() => onSelect(run.method)}>{methods[run.method].name}</button></th>
            <td><Meter value={run.metrics.mean_pcc} /></td><td><Meter value={run.metrics.mean_ssim} /></td>
            <td>{format(run.metrics.worst_pcc)}</td><td>{format(run.metrics.worst_ssim)}</td>
            <td>{run.iterations}</td><td>{formatTime(run.seconds)}</td>
          </tr>)}</tbody>
        </table>
      </div>
    </section>
    {preview && <section className="panel comparison-panel">
      <div className="comparison-toolbar">
        <div><h2>Предпросмотр · {methods[preview.method].name}</h2><p className="muted mt-1 text-sm">Фазовая карта и четыре восстановленных изображения.</p></div>
        <a className="button secondary" href="#restoration">Открыть результаты <ArrowRight size={16} /></a>
      </div>
      <div className="preview-strip">
        <figure><img src={pngUrl(preview.phase_png)} alt="Фазовая карта метода" /><figcaption>Фазовый ДОЭ</figcaption></figure>
        {preview.reconstruction_pngs.map((png, index) => <figure key={index}>
          <img src={pngUrl(png)} alt={`Амплитуда · плоскость ${index + 1}`} />
          <figcaption>{preview.metrics.planes[index]?.distance_mm.toFixed(1)} мм</figcaption>
        </figure>)}
      </div>
    </section>}
  </div>
}
