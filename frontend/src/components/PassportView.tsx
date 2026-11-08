import { useState } from 'react'
import { ArrowDownToLine, ClipboardList, LoaderCircle } from 'lucide-react'
import { pngUrl } from '../api'
import { exportPassport } from '../export'
import { useObjectUrl } from '../hooks/useObjectUrl'
import { format, formatTime, methods, passportLabel, type Passport } from '../workspace'

function PassportTarget({ file, index }: { file: File | undefined; index: number }) {
  const url = useObjectUrl(file ?? null)
  return <figure><div>{url && <img src={url} alt={`Цель ${index + 1}`} />}</div><figcaption>Цель</figcaption></figure>
}

export function PassportView({ entries, selected, onSelect }: {
  entries: Passport[]
  selected: Passport | undefined
  onSelect: (id: string) => void
}) {
  const [exporting, setExporting] = useState(false)
  const [error, setError] = useState('')

  if (!selected) return <section className="panel detail-panel">
    <p>Паспорт создаётся для каждого готового ДОЭ. <a href="#settings" className="text-link">Перейти к синтезу</a></p>
  </section>

  const { run } = selected
  const parameters = run.parameters
  const summary = [
    ['Сценарий', run.smart ? 'Smart Optimization · эксперимент' : 'Обычный синтез'],
    ['Метод', methods[run.method].name],
    ['Модель', run.checkpoint?.id ?? 'Classical · Adam'],
    ['Архитектура', run.checkpoint?.architecture ?? '—'],
    ['Версия checkpoint', run.model_version ?? '—'],
    ['Длина волны', `${parameters.wavelength_nm} нм`],
    ['Шаг модулятора', `${parameters.pixel_pitch_um} мкм`],
    ['Фазовая сетка', `${parameters.size_px} × ${parameters.size_px} · 8 бит · −π…+π`],
    ['Плоскости', parameters.distances_mm.map(distance => distance.toFixed(1)).join(' / ') + ' мм'],
    ['Реконструкция', parameters.reconstruction],
    ['Вычисление', parameters.device],
    ['Seed', parameters.seed],
    ['Итерации', run.iterations],
    ['Learning rate', parameters.learning_rate ?? '—'],
    ['Синтез и восстановление', formatTime(run.seconds)],
    ['PCC средняя / мин.', `${format(run.metrics.mean_pcc)} / ${format(run.metrics.worst_pcc)}`],
    ['SSIM средняя / мин.', `${format(run.metrics.mean_ssim)} / ${format(run.metrics.worst_ssim)}`],
  ]
  if (run.smart) {
    summary.push(
      ['Дополнительные Neural-запуски', run.smart.forward_count],
      ['Время поиска', formatTime(run.smart.total_seconds)],
      ['Цели для метрик', run.smart.mode === 'adapt' ? 'Адаптированные' : 'Оригинальные'],
      ['PCC по оригиналам', format(run.smart.original_metrics.mean_pcc)],
      ['Global SSIM по оригиналам', format(run.smart.original_metrics.mean_ssim)],
    )
  }

  async function download(entry: Passport) {
    setExporting(true)
    setError('')
    try {
      await exportPassport(entry)
    } catch {
      setError('Не удалось собрать архив паспорта.')
    } finally {
      setExporting(false)
    }
  }

  return <div className="detail-stack">
    <section className="panel detail-panel">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <label className="number-field report-selector">Расчёт
          <select value={selected.id} onChange={event => onSelect(event.currentTarget.value)}>
            {entries.map(entry => <option key={entry.id} value={entry.id}>{passportLabel(entry)}</option>)}
          </select>
        </label>
        <button type="button" className="button primary" disabled={exporting} onClick={() => void download(selected)}>
          {exporting ? <LoaderCircle size={16} className="spin" /> : <ArrowDownToLine size={16} />} Скачать паспорт ZIP
        </button>
      </div>
      {error && <p className="banner error" role="alert">{error}</p>}
      <p className="muted text-sm">Паспорта доступны в текущей сессии. ZIP содержит паспорт в JSON, фазовый PNG, четыре цели и четыре реконструкции.</p>
    </section>
    <div className="result-summary">
      <section className="panel phase-panel">
        <h2>Фазовая карта</h2>
        <div className="phase-preview"><img src={pngUrl(run.phase_png)} alt="Фазовая карта ДОЭ" /></div>
        <div className="phase-scale"><span>−π</span><span className="scale" /><span>+π</span></div>
      </section>
      <section className="panel detail-panel">
        <h2 className="flex items-center gap-2"><ClipboardList size={18} /> Параметры и метрики</h2>
        <dl className="report-grid">{summary.map(([label, value]) =>
          <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
      </section>
    </div>
    <div className="plane-results">
      {run.metrics.planes.map(({ distance_mm, pcc, ssim }, index) => <article className="panel plane-result" key={distance_mm}>
        <h3>z = {distance_mm.toFixed(1)} мм</h3>
        <div className="plane-pair">
          <PassportTarget file={selected.targets[index]} index={index} />
          <figure><div><img src={pngUrl(run.reconstruction_pngs[index] ?? '')} alt={`Реконструкция ${index + 1}`} /></div>
            <figcaption>Реконструкция</figcaption></figure>
        </div>
        <dl className="plane-scores"><div><dt>PCC</dt><dd>{format(pcc)}</dd></div><div><dt>SSIM</dt><dd>{format(ssim)}</dd></div></dl>
      </article>)}
    </div>
  </div>
}
