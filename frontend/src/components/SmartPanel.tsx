import { useEffect, useRef, useState } from 'react'
import { LoaderCircle, Sparkles, X } from 'lucide-react'
import { decodeBase64, pngUrl, request, targetForm, type Run, type SmartCandidate, type SmartResult, type SmartStatus } from '../api'
import { distances, format, formatTime } from '../workspace'
import { useObjectUrl } from '../hooks/useObjectUrl'

const names: Record<string, string> = {
  gamma: 'Гамма', brightness: 'Яркость', contrast: 'Адаптивный контраст',
  denoise: 'Подавление шума', frequency: 'Частотный фильтр',
}

function configLabel(candidate: SmartCandidate) {
  return candidate.config.operations.map(operation => {
    const settings = Object.entries(operation).filter(([key]) => key !== 'method')
      .map(([key, value]) => key + '=' + value).join(', ')
    return (names[operation.method] ?? operation.method) + ' · ' + settings
  }).join(' + ') || 'Исходные изображения'
}

function average(values: number[]) {
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function OriginalPreview({ file, index }: { file: File; index: number }) {
  const url = useObjectUrl(file)
  return <figure className="panel plane-result">
    {url && <img className="smart-original" src={url} alt={`Оригинал ${index + 1}`} />}
    <figcaption>{distances[index]} мм · {file.name}</figcaption>
  </figure>
}

export function SmartPanel({ open, baseline, targets, disabled, onBusy, onClose, onAccept }: {
  open: boolean
  baseline: Run | undefined
  targets: File[]
  disabled: boolean
  onBusy: (busy: boolean) => void
  onClose: () => void
  onAccept: (run: Run, targets: File[]) => void
}) {
  const [mode, setMode] = useState<'preserve' | 'adapt'>('preserve')
  const [status, setStatus] = useState<SmartStatus | null>(null)
  const [result, setResult] = useState<SmartResult | null>(null)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(0)
  const [submitting, setSubmitting] = useState(false)
  const [accepted, setAccepted] = useState(false)
  const [reload, setReload] = useState(0)
  const source = useRef<File[]>(targets)
  const pending = submitting || !!status && (status.status === 'analysis' || status.status === 'search')
  const candidate = result?.candidates[selected]
  const best = result?.candidates[result.best_index]
  const original = result?.candidates[0]

  useEffect(() => {
    if (!status || !pending) return
    const controller = new AbortController()
    let timer = 0
    async function poll() {
      try {
        const next = await request<SmartStatus>('/api/smart/' + status!.id, { signal: controller.signal })
        if (controller.signal.aborted) return
        if (next.status === 'analysis' || next.status === 'search') {
          setStatus(next)
          timer = window.setTimeout(() => void poll(), 500)
          return
        }
        const report = await request<SmartResult>('/api/smart/' + next.id + '/result', { signal: controller.signal })
        if (controller.signal.aborted) return
        setResult(report)
        setSelected(report.best_index)
        setStatus(next)
        setError(next.error ?? '')
        onBusy(false)
      } catch (failure) {
        if (controller.signal.aborted) return
        const message = failure instanceof Error ? failure.message : 'Не удалось получить состояние задачи.'
        setError(message)
        if (message.includes('Задача истекла')) {
          setStatus(previous => previous && ({ ...previous, status: 'error', error: message }))
          onBusy(false)
        } else {
          timer = window.setTimeout(() => void poll(), 1500)
        }
      }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [status?.id, pending, reload])

  async function start() {
    if (!baseline || targets.length !== 4 || disabled || pending) return
    setSubmitting(true)
    setStatus(null)
    setError('')
    setResult(null)
    setAccepted(false)
    onBusy(true)
    source.current = targets
    try {
      const body = targetForm(targets)
      body.append('phase', new Blob([decodeBase64(baseline.phase_png)], { type: 'image/png' }), 'baseline.png')
      body.append('model_version', baseline.model_version ?? '')
      body.append('baseline_seconds', String(baseline.seconds))
      body.append('mode', mode)
      const job = await request<SmartStatus>('/api/smart', { method: 'POST', body })
      setStatus(job)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Поиск не запущен.')
      onBusy(false)
    } finally {
      setSubmitting(false)
    }
  }

  async function cancel() {
    if (!status) return
    try {
      await request<SmartStatus>('/api/smart/' + status.id + '/cancel', { method: 'POST' })
      setStatus(previous => previous && ({ ...previous, cancel_requested: true }))
      setReload(value => value + 1)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Не удалось отменить задачу.')
    }
  }

  function accept() {
    if (!result || !best || !result.improved || accepted || source.current !== targets) return
    const adapted = result.mode === 'adapt'
    const nextTargets = adapted ? best.inputs.map((image, index) =>
      new File([decodeBase64(image)], 'adapted-z' + (index + 1) + '.png', { type: 'image/png' })) : targets
    onAccept({
      ...best.run,
      metrics: adapted ? best.adapted_metrics : best.run.metrics,
      smart: {
        job_id: result.id, mode: result.mode, config: best.config,
        forward_count: result.forward_count, total_seconds: result.total_seconds,
        original_metrics: best.run.metrics, adapted_metrics: best.adapted_metrics,
        originals: result.originals, inputs: best.inputs,
      },
    }, nextTargets)
    setAccepted(true)
  }

  if (!open) return null
  return <section className="panel detail-panel smart-panel" aria-labelledby="smart-heading">
    <div className="section-heading">
      <h2 id="smart-heading" className="flex items-center gap-2"><Sparkles size={18} /> Smart Optimization</h2>
      <button className="icon-button" type="button" aria-label="Закрыть панель" onClick={onClose}><X size={18} /></button>
    </div>
    <p className="muted text-sm">Экспериментальный поиск использует несколько независимых запусков Neural. Стоимость поиска показывается отдельно от одиночного синтеза.</p>
    <fieldset disabled={pending || disabled} className="architecture-controls">
      <legend className="sr-only">Режим подготовки</legend>
      <label className="chip"><input type="radio" name="smart-mode" checked={mode === 'preserve'} onChange={() => setMode('preserve')} /> Сохранить оригиналы</label>
      <label className="chip"><input type="radio" name="smart-mode" checked={mode === 'adapt'} onChange={() => setMode('adapt')} /> Адаптировать изображения</label>
    </fieldset>
    <p className="muted text-xs">{mode === 'preserve'
      ? 'Подготовка меняет только входы сети. Все официальные метрики сравниваются с оригиналами.'
      : 'Цели заменятся адаптированными версиями после принятия. Сравнение с оригиналами сохранится отдельно.'}</p>
    {!result && <div className="smart-previews">{targets.map((file, index) =>
      <OriginalPreview key={index} file={file} index={index} />)}</div>}
    <div className="flex flex-wrap gap-2">
      <button className="button primary" type="button" disabled={disabled || pending || !baseline || targets.length !== 4} onClick={() => void start()}>
        {pending ? <LoaderCircle size={17} className="spin" /> : <Sparkles size={17} />}
        {pending ? 'Ищем улучшение…' : 'Проверить варианты'}
      </button>
      {pending && <button type="button" className="button secondary" disabled={submitting || status?.cancel_requested} onClick={() => void cancel()}>
        {status?.cancel_requested ? 'Завершаем текущий вариант…' : 'Отменить поиск'}
      </button>}
    </div>
    {result && source.current !== targets && !accepted && <p className="hint">Результаты относятся к предыдущему набору целей. Запустите новый поиск для текущих изображений.</p>}
    {status && <p role="status" className="hint">
      {status.status === 'analysis' ? 'Анализ изображений'
        : status.status === 'search' ? `Проверено ${status.checked} из ${status.total} вариантов`
          : status.status === 'cancelled' ? 'Поиск отменён'
            : status.status === 'error' ? 'Ошибка поиска'
              : result?.improved ? 'Улучшение найдено' : 'Улучшение не найдено'}
    </p>}
    {error && <div className="banner error" role="alert">{error}
      {pending && <button className="text-link" onClick={() => { setError(''); setReload(value => value + 1) }}>Проверить состояние</button>}
    </div>}
    {result && <>
      <p className="muted text-sm">{result.forward_count} дополнительных Neural-запусков · поиск {formatTime(result.total_seconds)}.
        PCC и Global SSIM ниже рассчитаны по оригиналам. Порог улучшения score {result.selection_policy.minimum_gain},
        допустимое снижение по плоскости {result.selection_policy.plane_tolerance}.
        Score — среднее mean/worst PCC и mean/worst Global SSIM.</p>
      {result.analysis && <details>
        <summary className="text-link cursor-pointer">Анализ изображений и сходство</summary>
        <p className="muted text-xs">Характеристики используются для отбора преобразований. Прогноз качества требует отдельной проверки.</p>
        <div className="table-scroll"><table>
          <thead><tr><th>Плоскость</th><th>Яркость</th><th>Контраст</th><th>Энтропия</th><th>Границы</th><th>Активные пиксели</th><th>Центр FFT</th><th>ВЧ энергия</th></tr></thead>
          <tbody>{result.analysis.planes.map((plane, index) => <tr key={index}><th>{distances[index]} мм</th>
            {[plane.brightness, plane.contrast, plane.entropy_bits, plane.edge_fraction, plane.active_fraction, plane.spectral_centroid, plane.high_frequency_fraction]
              .map((value, i) => <td key={i}>{format(value)}</td>)}</tr>)}</tbody>
        </table></div>
        <p className="muted text-xs">PCC пар целей · {result.analysis.similarity.map((value, index) =>
          result.analysis.similarity_pairs[index]?.join('–') + ' = ' + format(value)).join(' / ')}</p>
      </details>}
      <div className="table-scroll"><table>
        <thead><tr><th>Вариант</th><th>Mean PCC</th><th>Worst PCC</th><th>Global SSIM</th><th>Worst SSIM</th><th>Windowed SSIM</th><th>Фоновая мощность</th><th>Синтез + восстановление</th></tr></thead>
        <tbody>{result.candidates.map(item => <tr key={item.index} className={selected === item.index ? 'selected-row' : ''}>
          <th><button className="text-link" type="button" onClick={() => setSelected(item.index)}>{item.index === 0 ? 'Baseline' : `№${item.index}`}{item.index === result.best_index ? ' · выбран' : ''}</button></th>
          <td>{format(item.run.metrics.mean_pcc)}</td><td>{format(item.run.metrics.worst_pcc)}</td>
          <td>{format(item.run.metrics.mean_ssim)}</td><td>{format(item.run.metrics.worst_ssim)}</td>
          <td>{format(average(item.windowed_ssim))}</td><td>{(average(item.background_power) * 100).toFixed(2)}%</td><td>{formatTime(item.run.seconds)}</td>
        </tr>)}</tbody>
      </table></div>
      {candidate && <>
        <p className="text-sm">{configLabel(candidate)}</p>
        <div className="smart-previews">{result.originals.map((image, index) => <article className="panel plane-result" key={index}>
          <h3>{distances[index]} мм</h3>
          <div className="plane-pair">
            <figure><div><img src={pngUrl(image)} alt={`Оригинал ${index + 1}`} /></div><figcaption>Оригинал</figcaption></figure>
            <figure><div><img src={pngUrl(candidate.inputs[index] ?? '')} alt={`Подготовка ${index + 1}`} /></div><figcaption>Вход сети</figcaption></figure>
            <figure><div><img src={pngUrl(original?.run.reconstruction_pngs[index] ?? '')} alt="Исходное восстановление" /></div><figcaption>До</figcaption></figure>
            <figure><div><img src={pngUrl(candidate.run.reconstruction_pngs[index] ?? '')} alt="Восстановление варианта" /></div><figcaption>Вариант</figcaption></figure>
          </div>
          <p className="muted text-xs">Изменение входа MAE {format(candidate.input_mae[index] ?? 0)}</p>
          <p className="text-xs">Оригинал · PCC {format(candidate.run.metrics.planes[index]?.pcc ?? 0)} · Global SSIM {format(candidate.run.metrics.planes[index]?.ssim ?? 0)}</p>
          {result.mode === 'adapt' && <p className="text-xs">Адаптированная цель · PCC {format(candidate.adapted_metrics.planes[index]?.pcc ?? 0)} · Global SSIM {format(candidate.adapted_metrics.planes[index]?.ssim ?? 0)}</p>}
        </article>)}</div>
        <div className="plane-pair smart-phase-pair">
          <figure><div><img src={pngUrl(original?.run.phase_png ?? '')} alt="Исходный ДОЭ" /></div><figcaption>Исходный ДОЭ · −π…+π</figcaption></figure>
          <figure><div><img src={pngUrl(candidate.run.phase_png)} alt="ДОЭ варианта" /></div><figcaption>ДОЭ варианта · −π…+π</figcaption></figure>
        </div>
      </>}
      {result.status === 'completed' && <div className="flex flex-wrap gap-2">
        {result.improved && <button className="button primary" type="button"
          disabled={accepted || disabled || source.current !== targets} onClick={accept}>
          {accepted ? 'Результат принят' : result.mode === 'adapt' ? 'Принять ДОЭ и адаптированные цели' : 'Принять улучшенный ДОЭ'}
        </button>}
        <button className="button secondary" type="button" onClick={onClose}>Оставить текущий результат</button>
      </div>}
    </>}
  </section>
}
