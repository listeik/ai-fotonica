import { useState } from 'react'
import { LoaderCircle, ScanLine } from 'lucide-react'
import { decodeBase64, pngUrl, request, targetForm, type Diagnostics } from '../api'
import { format, passportLabel, type Passport } from '../workspace'

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

export function ResearchView({ entries, disabled, onBusy }: {
  entries: Passport[]
  disabled: boolean
  onBusy: (busy: boolean) => void
}) {
  const [sourceId, setSourceId] = useState('')
  const [report, setReport] = useState<{ id: string; value: Diagnostics } | null>(null)
  const [pending, setPending] = useState(false)
  const [mapError, setMapError] = useState('')
  const source = entries.find(entry => entry.id === sourceId) ?? entries[0]

  async function calculateMaps() {
    if (!source || disabled) return
    setMapError('')
    setPending(true)
    onBusy(true)
    try {
      const body = targetForm(source.targets)
      body.append('phase', new Blob([decodeBase64(source.run.phase_png)], { type: 'image/png' }), 'doe.png')
      const value = await request<Diagnostics>('/api/research/diagnostics', { method: 'POST', body })
      setReport({ id: source.id, value })
    } catch (failure) {
      setMapError(failure instanceof Error ? failure.message : 'Не удалось рассчитать карты.')
    } finally {
      setPending(false)
      onBusy(false)
    }
  }

  return <div className="detail-stack">
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
            {pending ? <LoaderCircle size={16} className="spin" /> : <ScanLine size={16} />}
            {pending ? 'Рассчитываем карты…' : 'Рассчитать карты'}
          </button>
        </div>
        : <p>Выполните <a className="text-link" href="#settings">синтез ДОЭ</a>, чтобы получить фазовую карту.</p>}
      {mapError && <p className="banner error" role="alert">{mapError}</p>}
    </section>
    {report && report.id === source?.id && <Maps report={report.value} />}
  </div>
}
