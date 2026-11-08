import { useEffect, useState } from 'react'
import { FlaskConical, LoaderCircle, ScanLine } from 'lucide-react'
import { decodeBase64, pngUrl, request, targetForm, type Diagnostics, type ResearchOption, type Run } from '../api'
import { format, formatTime, passportLabel, type Passport } from '../workspace'

function Maps({ report }: { report: Diagnostics }) {
  const [low, high] = report.power_db_range
  return <>
    <p className="muted text-sm">Ошибка рассчитана по 8-битным PNG. Карты интенсивности |U|² показаны на общей логарифмической шкале {low}…{high} дБ относительно единичной входной интенсивности. Доли мощности рассчитаны отдельно для каждой плоскости. Маска цели включает пиксели ярче {Math.round(report.support_threshold * 255)} из 255.</p>
    <div className="plane-results">
      {report.planes.map(plane => <article className="panel plane-result" key={plane.distance_mm}>
        <div><h3>z = {plane.distance_mm.toFixed(1)} мм</h3><p className="muted text-xs">Шаг сетки {plane.output_pitch_um.toFixed(2)} мкм</p></div>
        <div className="plane-pair">
          <figure><div><img src={pngUrl(plane.error_png)} alt={`Абсолютная ошибка на ${plane.distance_mm} мм`} /></div><figcaption>Ошибка · 0–1</figcaption></figure>
          <figure><div><img src={pngUrl(plane.intensity_png)} alt={`Интенсивность на ${plane.distance_mm} мм`} /></div><figcaption>Интенсивность · дБ</figcaption></figure>
        </div>
        <div className="phase-scale"><span>{low} дБ</span><span className="scale" /><span>{high} дБ</span></div>
        <dl className="plane-scores">
          <div><dt>MAE</dt><dd>{format(plane.mae)}</dd></div>
          <div><dt>Мощность в цели</dt><dd>{(plane.power_in_target * 100).toFixed(1)}%</dd></div>
        </dl>
        <div className="power-track" role="meter" aria-label="Доля мощности в области цели"
          aria-valuemin={0} aria-valuemax={100} aria-valuenow={+(plane.power_in_target * 100).toFixed(1)}>
          <span style={{ width: `${plane.power_in_target * 100}%` }} />
        </div>
      </article>)}
    </div>
  </>
}

export function ResearchView({ targets, entries, active, disabled, onBusy, onRuns }: {
  targets: File[]
  entries: Passport[]
  active: boolean
  disabled: boolean
  onBusy: (busy: boolean) => void
  onRuns: (runs: Run[], targets: File[]) => Passport[]
}) {
  const [options, setOptions] = useState<ResearchOption[] | null>(null)
  const [optionError, setOptionError] = useState('')
  const [retry, setRetry] = useState(0)
  const [architectures, setArchitectures] = useState<string[]>([])
  const [comparison, setComparison] = useState<{ targets: File[]; entries: Passport[] } | null>(null)
  const [sourceId, setSourceId] = useState('')
  const [report, setReport] = useState<{ id: string; value: Diagnostics } | null>(null)
  const [pending, setPending] = useState<'compare' | 'maps' | null>(null)
  const [comparisonError, setComparisonError] = useState('')
  const [mapError, setMapError] = useState('')
  const source = entries.find(entry => entry.id === sourceId) ?? entries[0]
  const canCompare = targets.length === 4 && architectures.length >= 2 && !disabled

  useEffect(() => {
    if (!active || options) return
    const controller = new AbortController()
    request<ResearchOption[]>('/api/research/options', { signal: controller.signal }).then(items => {
      if (controller.signal.aborted) return
      setOptions(items)
      setArchitectures(items.map(item => item.id))
      setOptionError('')
    }).catch(failure => {
      if (!controller.signal.aborted) setOptionError(failure.message)
    })
    return () => controller.abort()
  }, [active, options, retry])

  async function compare() {
    if (!canCompare) return
    setComparisonError('')
    setPending('compare')
    onBusy(true)
    try {
      const body = targetForm(targets)
      for (const architecture of architectures) body.append('architectures', architecture)
      const runs = await request<Run[]>('/api/research/compare', { method: 'POST', body })
      setComparison({ targets, entries: onRuns(runs, targets) })
    } catch (failure) {
      setComparisonError(failure instanceof Error ? failure.message : 'Не удалось выполнить сравнение.')
    } finally {
      setPending(null)
      onBusy(false)
    }
  }

  async function calculateMaps() {
    if (!source || disabled) return
    setMapError('')
    setPending('maps')
    onBusy(true)
    try {
      const body = targetForm(source.targets)
      body.append('phase', new Blob([decodeBase64(source.run.phase_png)], { type: 'image/png' }), 'doe.png')
      const value = await request<Diagnostics>('/api/research/diagnostics', { method: 'POST', body })
      setReport({ id: source.id, value })
    } catch (failure) {
      setMapError(failure instanceof Error ? failure.message : 'Не удалось рассчитать карты.')
    } finally {
      setPending(null)
      onBusy(false)
    }
  }

  return <div className="detail-stack">
    <section className="panel detail-panel">
      <h2 className="flex items-center gap-2"><FlaskConical size={18} /> Сравнение архитектур</h2>
      <p className="muted text-sm">Выберите минимум две архитектуры. Каждая обработает один и тот же текущий набор целей в режиме Neural.</p>
      {targets.length < 4 && <p><a className="text-link" href="#targets">Загрузить четыре цели</a></p>}
      {optionError
        ? <p className="banner error" role="alert">{optionError} <button className="text-link" onClick={() => { setOptionError(''); setRetry(value => value + 1) }}>Повторить</button></p>
        : !options ? <p className="hint" role="status"><LoaderCircle size={16} className="spin" /> Загружаем список архитектур…</p>
          : options.length < 2 ? <p className="hint">Для сравнения нужны два доступных checkpoint на сервере.</p>
            : <fieldset className="architecture-controls" disabled={disabled}>
              <legend className="sr-only">Архитектуры эксперимента</legend>
              {options.map(option => <label key={option.id} className="chip">
                <input type="checkbox" checked={architectures.includes(option.id)} onChange={event => {
                  const checked = event.currentTarget.checked
                  setArchitectures(previous => checked ? [...previous, option.id] : previous.filter(id => id !== option.id))
                }} /> {option.name}
              </label>)}
            </fieldset>}
      <button type="button" className="button primary justify-self-start" disabled={!canCompare} onClick={() => void compare()}>
        {pending === 'compare' ? <LoaderCircle className="spin" size={16} /> : <FlaskConical size={16} />}
        {pending === 'compare' ? 'Выполняется сравнение…' : 'Сравнить на текущих целях'}
      </button>
      {comparisonError && <p className="banner error" role="alert">{comparisonError}</p>}
      {comparison && <>
        {comparison.targets !== targets && <p className="hint">Показаны результаты для предыдущего набора целей.</p>}
        <div className="table-scroll">
          <table>
            <thead><tr><th scope="col">Архитектура и checkpoint</th><th scope="col">Средняя PCC</th><th scope="col">Средняя SSIM</th><th scope="col">Синтез и восстановление</th><th scope="col">Устройство</th><th scope="col">Карты</th></tr></thead>
            <tbody>{comparison.entries.map(({ id, number, run }) => <tr key={id}>
              <th scope="row">{run.checkpoint?.architecture}<small className="muted block">{run.model_version}</small></th>
              <td>{format(run.metrics.mean_pcc)}</td><td>{format(run.metrics.mean_ssim)}</td>
              <td>{formatTime(run.seconds)}</td><td>{run.parameters.device}</td>
              <td><button className="text-link" type="button" disabled={disabled} onClick={() => {
                setSourceId(id)
                setMapError('')
                document.getElementById('research-maps')?.scrollIntoView({ block: 'start' })
              }}>Выбрать №{number}</button></td>
            </tr>)}</tbody>
          </table>
        </div>
      </>}
    </section>
    <section className="panel detail-panel" id="research-maps">
      <h2 className="flex items-center gap-2"><ScanLine size={18} /> Ошибки и оптическая мощность</h2>
      {source
        ? <div className="flex flex-wrap items-end gap-3">
          <label className="number-field report-selector">ДОЭ из паспортов
            <select value={source.id} disabled={disabled} onChange={event => { setSourceId(event.currentTarget.value); setMapError('') }}>
              {entries.map(entry => <option key={entry.id} value={entry.id}>{passportLabel(entry)}</option>)}
            </select>
          </label>
          <button type="button" className="button secondary" disabled={disabled} onClick={() => void calculateMaps()}>
            {pending === 'maps' ? <LoaderCircle size={16} className="spin" /> : <ScanLine size={16} />}
            {pending === 'maps' ? 'Рассчитываем карты…' : 'Рассчитать карты'}
          </button>
        </div>
        : <p>Выполните <a className="text-link" href="#settings">синтез ДОЭ</a> или сравните архитектуры, чтобы получить фазовую карту.</p>}
      {mapError && <p className="banner error" role="alert">{mapError}</p>}
    </section>
    {report && report.id === source?.id && <Maps report={report.value} />}
  </div>
}
