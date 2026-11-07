import { ArrowDownToLine, ArrowRight, Clock3, Layers3, LoaderCircle } from 'lucide-react'
import { decodeBase64, downloadFile, pngUrl, type Method, type Run } from '../api'
import { useObjectUrl } from '../hooks/useObjectUrl'
import { format, formatTime, methods } from '../workspace'

function PlaneResult({ index, metrics, target, reconstruction }: {
  index: number
  metrics: Run['metrics']['planes'][number]
  target: File | null
  reconstruction: string
}) {
  const targetUrl = useObjectUrl(target)

  return <article className="panel plane-result">
    <div className="flex items-center justify-between gap-2">
      <span className="step-number">{String(index + 1).padStart(2, '0')}</span>
      <h3>z = {metrics.distance_mm.toFixed(1)} мм</h3>
    </div>
    <div className="plane-pair">
      <figure><div>{targetUrl && <img src={targetUrl} alt={`Цель · плоскость ${index + 1}`} />}</div><figcaption>Цель</figcaption></figure>
      <figure><div><img src={pngUrl(reconstruction)} alt={`Восстановление · плоскость ${index + 1}`} /></div><figcaption>Восстановление</figcaption></figure>
    </div>
    <dl className="plane-scores">
      <div><dt title="Коэффициент корреляции Пирсона">PCC</dt><dd>{format(metrics.pcc)}</dd></div>
      <div><dt title="Индекс структурного сходства">SSIM</dt><dd>{format(metrics.ssim)}</dd></div>
    </dl>
  </article>
}

export function ResultsView({ run, runs, targets, running, onSelect }: {
  run?: Run
  runs: Run[]
  targets: File[]
  running: boolean
  onSelect: (method: Method) => void
}) {
  if (!run) return <section className="panel empty-state">
    <span className="icon-tile">{running ? <LoaderCircle size={25} className="spin" /> : <Layers3 size={25} />}</span>
    <h2>{running ? 'Выполняется синтез' : 'Результаты появятся после расчёта'}</h2>
    <p>{running ? 'Рассчитываем фазу и восстановление.' : 'Запустите синтез, чтобы получить фазовый ДОЭ и четыре восстановленных изображения.'}</p>
    {!running && <a className="button secondary" href="#settings">К настройкам <ArrowRight size={16} /></a>}
  </section>

  const summary = [
    ['Средняя PCC', format(run.metrics.mean_pcc)],
    ['Средняя SSIM', format(run.metrics.mean_ssim)],
    ['Минимальная PCC', format(run.metrics.worst_pcc)],
    ['Минимальная SSIM', format(run.metrics.worst_ssim)],
    ['Время синтеза', formatTime(run.seconds)],
    ['Итерации', run.iterations],
  ]
  return <div className="grid gap-5 sm:gap-6">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex flex-wrap gap-2" role="group" aria-label="Показать результат метода">
        {runs.map(item => <button key={item.method} className="chip selectable" type="button"
          aria-pressed={run.method === item.method} onClick={() => onSelect(item.method)}>
          {methods[item.method].name}
        </button>)}
      </div>
      <a className="text-link flex items-center gap-1.5" href="#comparison">Сравнение методов <ArrowRight size={15} /></a>
    </div>
    <div className="result-summary">
      <section className="panel phase-panel">
        <h2 className="flex items-center gap-2"><Layers3 size={18} /> Фазовая карта</h2>
        <div className="phase-preview"><img src={pngUrl(run.phase_png)} alt="Фазовая карта ДОЭ" /></div>
        <div className="phase-scale"><span>−π</span><span className="scale" /><span>+π</span></div>
        <button type="button" className="button secondary" onClick={() =>
          downloadFile(decodeBase64(run.phase_png), 'doe-' + run.method + '.png', 'image/png')}>
          <ArrowDownToLine size={16} /> Скачать ДОЭ
        </button>
      </section>
      <section className="panel summary-panel">
        <h2 className="flex items-center gap-2"><Clock3 size={18} /> Сводка расчёта</h2>
        <dl className="metric-grid">
          {summary.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
        </dl>
      </section>
    </div>
    <section className="grid gap-3" aria-labelledby="plane-results-title">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="plane-results-title">Восстановление по плоскостям</h2>
        <span className="muted text-xs">Амплитуда поля из 8-битного ДОЭ</span>
      </div>
      <div className="plane-results">
        {run.metrics.planes.map((metrics, index) => <PlaneResult key={metrics.distance_mm}
          index={index} metrics={metrics} target={targets[index] ?? null} reconstruction={run.reconstruction_pngs[index] ?? ''} />)}
      </div>
    </section>
  </div>
}
