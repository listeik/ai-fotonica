import {
  Activity, ArrowDownToLine, ArrowRight, Atom, ChevronDown, CircleHelp, FlaskConical,
  FolderArchive, ImagePlus, Layers3, LoaderCircle, Play, SlidersHorizontal, X,
} from 'lucide-react'
import { useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { strToU8, zipSync } from 'fflate'
import {
  decodeBase64, downloadFile, pngUrl, request,
  type Demo, type Health, type Method, type Methods, type Model, type Run,
} from './api'
import './styles.css'

const distances = [24.1, 25.3, 26.5, 27.7]
const descriptions: Record<Method, { name: string; detail: string }> = {
  classical: { name: 'Classical', detail: 'Оптимизация фазы методом Adam' },
  neural: { name: 'Neural', detail: 'Прямой проход обученной U-Net' },
  hybrid: { name: 'Hybrid', detail: 'U-Net с уточнением фазы методом Adam' },
}
const methodIds = Object.keys(descriptions) as Method[]
const format = (value: number) => value.toFixed(3)
const formatTime = (seconds: number) => seconds < 1 ? Math.round(seconds * 1000) + ' мс' : seconds.toFixed(2) + ' с'

function OpticalDiagram() {
  return <svg className="optical-diagram" viewBox="0 0 610 235" aria-hidden="true">
    <defs>
      <linearGradient id="beam" x2="100%" y2="0%">
        <stop stopColor="#9a43bc" stopOpacity=".8" />
        <stop offset=".47" stopColor="#7650b5" stopOpacity=".65" />
        <stop offset="1" stopColor="#3c68b8" stopOpacity=".2" />
      </linearGradient>
      <linearGradient id="glass" x2="100%" y2="100%">
        <stop stopColor="#9164c9" stopOpacity=".2" />
        <stop offset="1" stopColor="#7761ab" stopOpacity="0" />
      </linearGradient>
      <pattern id="glassGrid" width="11" height="11" patternUnits="userSpaceOnUse">
        <path d="M 0 0 L 11 0 M 0 0 L 0 11" fill="none" stroke="#8870b9" strokeOpacity=".25" />
      </pattern>
    </defs>
    <path d="M 24 117 H 585" stroke="#715897" strokeDasharray="3 5" opacity=".6" />
    {[0, 1, 2, 3].map(plane => {
      const x = 295 + plane * 69
      return <g key={plane} opacity={1 - plane * .12}>
        <path d={'M ' + x + ' 56 l 48 -22 v 125 l -48 22 z'} fill="url(#glass)" stroke="#72559d" strokeWidth=".8" />
        <path d={'M ' + (x + 24) + ' 45 v 125 M ' + x + ' 118 l 48 -22'} stroke="#72559d" strokeWidth=".6" opacity=".5" />
        <circle cx={x + 24} cy="107" r="13" stroke="#7a53b2" fill="none" opacity=".65" />
        <circle cx={x + 24} cy="107" r="5" fill="#7a52b7" opacity=".8" />
        <text x={x + 5} y={203 + plane * 3} fill="#655476" fontSize="10" fontFamily="monospace">z{plane + 1}</text>
      </g>
    })}
    {Array.from({ length: 11 }, (_, index) => {
      const offset = (index - 5) * 6.4
      return <path key={index} d={'M 34 ' + (117 + offset * .39) + ' C 100 ' + (117 + offset * .39) + ', 131 ' + (117 + offset) + ', 170 ' + (117 + offset) + ' S 302 ' + (105 + offset * 1.18) + ', 557 ' + (95 + offset * 1.65)} stroke="url(#beam)" fill="none" />
    })}
    <path d="M 140 69 l 53 -25 v 116 l -53 25 z" fill="url(#glass)" stroke="#8255bd" />
    <path d="M 140 69 l 53 -25 v 116 l -53 25 z" fill="url(#glassGrid)" />
    <circle cx="34" cy="117" r="4" fill="#9a51b5" />
    <text x="131" y="211" fill="#694884" fontSize="10" fontFamily="monospace">DOE / φ</text>
  </svg>
}

function useObjectUrl(file: File | null) {
  const [url, setUrl] = useState<string | null>(null)
  useEffect(() => {
    if (!file) {
      setUrl(null)
      return
    }
    const next = URL.createObjectURL(file)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [file])
  return url
}

function TargetCard({ index, file, disabled, onChange }: {
  index: number
  file: File | null
  disabled: boolean
  onChange: (file: File | null) => void
}) {
  const url = useObjectUrl(file)
  const [size, setSize] = useState<{ url: string; width: number; height: number } | null>(null)
  return <div className="target-card">
    <div className="target-heading">
      <span className="plane-index">{String(index + 1).padStart(2, '0')}</span>
      <span>z = {distances[index]} мм</span>
    </div>
    <label className={'target-drop' + (file ? ' is-filled' : '')} onDragOver={event => event.preventDefault()}
      onDrop={event => {
        event.preventDefault()
        if (!disabled && event.dataTransfer.files[0]) onChange(event.dataTransfer.files[0])
      }}>
      <input type="file" accept=".png,image/png" disabled={disabled}
        aria-label={'Загрузить цель для плоскости ' + (index + 1)}
        onChange={event => {
          if (event.target.files?.[0]) onChange(event.target.files[0])
          event.target.value = ''
        }} />
      {file && url
        ? <img src={url} alt={'Цель ' + (index + 1)} onLoad={event => setSize({
          url, width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight,
        })} />
        : <span><ImagePlus size={22} />Выбрать PNG</span>}
    </label>
    <div className="file-meta">
      <span title={file?.name}>{file ? <>
        {file.name}
        {size && size.url === url && <small>{size.width} × {size.height} пикс.</small>}
      </> : '128 × 128 · 8 бит'}</span>
      {file && <button className="icon-button" type="button" disabled={disabled}
        onClick={() => onChange(null)} aria-label={'Удалить цель ' + (index + 1)}><X size={14} /></button>}
    </div>
  </div>
}

function ResultView({ run, targets }: { run: Run; targets: File[] }) {
  return <div className="result-content">
    <div className="result-summary">
      <div className="phase-panel">
        <div className="phase-heading"><Layers3 size={17} /> Фазовая карта</div>
        <img className="phase-image" src={pngUrl(run.phase_png)} alt="Фазовая карта ДОЭ" />
        <div className="phase-scale"><span>−π</span><span className="scale" /><span>+π</span></div>
        <button className="button secondary" onClick={() =>
          downloadFile(decodeBase64(run.phase_png), 'doe-' + run.method + '.png', 'image/png')}>
          <ArrowDownToLine size={16} /> Скачать ДОЭ
        </button>
      </div>
      <dl className="metric-grid">
        <div><dt>Средняя PCC</dt><dd>{format(run.metrics.mean_pcc)}</dd></div>
        <div><dt>Средняя SSIM</dt><dd>{format(run.metrics.mean_ssim)}</dd></div>
        <div><dt>Минимальная PCC</dt><dd>{format(run.metrics.worst_pcc)}</dd></div>
        <div><dt>Минимальная SSIM</dt><dd>{format(run.metrics.worst_ssim)}</dd></div>
        <div><dt>Время синтеза</dt><dd>{formatTime(run.seconds)}</dd></div>
        <div><dt>Итерации</dt><dd>{run.iterations}</dd></div>
      </dl>
    </div>
    <div className="plane-results">
      {run.metrics.planes.map((metrics, index) =>
        <PlaneResult key={metrics.distance_mm} metrics={metrics}
          target={targets[index] ?? null} reconstruction={run.reconstruction_pngs[index] ?? ''} />)}
    </div>
  </div>
}

function PlaneResult({ metrics, target, reconstruction }: {
  metrics: Run['metrics']['planes'][number]
  target: File | null
  reconstruction: string
}) {
  const targetUrl = useObjectUrl(target)
  return <article className="plane-result">
    <div className="plane-result-title">z = {metrics.distance_mm} мм</div>
    <div className="plane-pair">
      <figure>{targetUrl && <img src={targetUrl} alt="Целевое изображение" />}<figcaption>Цель</figcaption></figure>
      <figure><img src={pngUrl(reconstruction)} alt="Восстановленное изображение" /><figcaption>Амплитуда</figcaption></figure>
    </div>
    <div className="plane-scores">
      <span>PCC <strong>{format(metrics.pcc)}</strong></span>
      <span>SSIM <strong>{format(metrics.ssim)}</strong></span>
    </div>
  </article>
}

function Comparison({ runs, active, select, exportRuns }: {
  runs: Run[]
  active: Method | null
  select: (method: Method) => void
  exportRuns: () => void
}) {
  if (!runs.length) return <div className="empty-state small-empty">
    <Activity size={24} /><p>Здесь появятся метрики и время для каждого запуска.</p>
  </div>
  return <div className="comparison-content">
    <div className="comparison-toolbar">
      <span>Повторный запуск метода заменяет его строку.</span>
      <button className="button secondary" onClick={exportRuns}><FolderArchive size={16} /> Экспорт ZIP</button>
    </div>
    <div className="table-scroll">
      <table>
        <thead><tr>
          <th scope="col">Метод</th><th scope="col">PCC, средняя</th><th scope="col">SSIM, средняя</th>
          <th scope="col">PCC, мин.</th><th scope="col">SSIM, мин.</th>
          <th scope="col">Итерации</th><th scope="col">Время</th>
        </tr></thead>
        <tbody>{runs.map(run => <tr key={run.method}>
          <th scope="row"><button className="method-link" aria-pressed={active === run.method}
            onClick={() => select(run.method)}>{descriptions[run.method].name}</button></th>
          <td><Meter value={run.metrics.mean_pcc} /></td>
          <td><Meter value={run.metrics.mean_ssim} /></td>
          <td>{format(run.metrics.worst_pcc)}</td><td>{format(run.metrics.worst_ssim)}</td>
          <td>{run.iterations}</td><td>{formatTime(run.seconds)}</td>
        </tr>)}</tbody>
      </table>
    </div>
  </div>
}

function Meter({ value }: { value: number }) {
  return <div className="meter"><span>{format(value)}</span>
    <span className="meter-track" aria-hidden="true"><i style={{ width: Math.max(0, Math.min(100, 50 + value * 50)) + '%' }} /></span>
  </div>
}

function SectionHeading({ index, title, description }: {
  index: string
  title: string
  description: string
}) {
  return <div className="section-heading">
    <div className="section-number">{index}</div>
    <div><h2>{title}</h2><p>{description}</p></div>
  </div>
}

function App() {
  const [targets, setTargets] = useState<(File | null)[]>(Array(4).fill(null))
  const [method, setMethod] = useState<Method>('classical')
  const [steps, setSteps] = useState({ classical: 160, hybrid: 40 })
  const [runs, setRuns] = useState<Partial<Record<Method, Run>>>({})
  const [active, setActive] = useState<Method | null>(null)
  const [methods, setMethods] = useState<Methods | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [model, setModel] = useState<Model | null>(null)
  const [demos, setDemos] = useState<Demo[]>([])
  const [demoId, setDemoId] = useState<number | null>(null)
  const [busy, setBusy] = useState<'demo' | 'synthesis' | 'export' | null>(null)
  const [error, setError] = useState('')
  const [connectionFailed, setConnectionFailed] = useState(false)
  const inputCount = targets.filter(Boolean).length
  const validTargets = targets.filter((file): file is File => file !== null)
  const completed = methodIds.flatMap(id => runs[id] ? [runs[id]] : [])
  const result = active ? runs[active] : null

  async function connect(signal?: AbortSignal) {
    setConnectionFailed(false)
    try {
      const [nextHealth, nextMethods, nextModel, nextDemos] = await Promise.all([
        request<Health>('/api/health', { signal }),
        request<Methods>('/api/methods', { signal }),
        request<Model>('/api/model/info', { signal }),
        request<Demo[]>('/api/demo', { signal }),
      ])
      setHealth(nextHealth)
      setMethods(nextMethods)
      setModel(nextModel)
      setDemos(nextDemos)
    } catch {
      if (!signal?.aborted) setConnectionFailed(true)
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    void connect(controller.signal)
    return () => controller.abort()
  }, [])

  function clearResults() {
    setRuns({})
    setActive(null)
    setError('')
  }

  function changeTarget(index: number, file: File | null) {
    if (file && (!file.name.toLowerCase().endsWith('.png') || file.size > 64 * 1024)) {
      setError('Выберите файл PNG размером до 64 КиБ.')
      return
    }
    clearResults()
    setTargets(previous => previous.map((target, position) => position === index ? file : target))
    setDemoId(null)
  }

  async function loadDemo(index: number) {
    setBusy('demo')
    setError('')
    try {
      const demo = await request<Demo & { images: string[] }>('/api/demo/' + index)
      clearResults()
      setTargets(demo.images.map((image, plane) =>
        new File([decodeBase64(image)], 'demo-' + (index + 1) + '-z' + (plane + 1) + '.png', { type: 'image/png' })))
      setDemoId(index)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Не удалось загрузить набор.')
    } finally {
      setBusy(null)
    }
  }

  async function runSynthesis() {
    if (validTargets.length !== 4) return
    const data = new FormData()
    for (const target of validTargets) data.append('images', target)
    data.append('method', method)
    if (method !== 'neural') data.append('iterations', String(steps[method]))

    setError('')
    setBusy('synthesis')
    try {
      const response = await request<Run>('/api/synthesize', { method: 'POST', body: data })
      setRuns(previous => ({ ...previous, [response.method]: response }))
      setActive(response.method)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Синтез не выполнен.')
    } finally {
      setBusy(null)
    }
  }

  async function exportResults() {
    setBusy('export')
    setError('')
    try {
      const files: Record<string, Uint8Array> = {}
      for (const [index, target] of validTargets.entries()) {
        files['targets/z' + (index + 1) + '.png'] = new Uint8Array(await target.arrayBuffer())
      }
      const summaries = completed.map(run => {
        const { phase_png, reconstruction_pngs, ...summary } = run
        files[run.method + '/doe.png'] = decodeBase64(phase_png)
        reconstruction_pngs.forEach((png, index) => {
          files[run.method + '/z' + (index + 1) + '.png'] = decodeBase64(png)
        })
        return summary
      })
      files['comparison.json'] = strToU8(JSON.stringify(summaries, null, 2))
      const csv = [
        'method,distance_mm,pcc,ssim,iterations,seconds',
        ...completed.flatMap(run => run.metrics.planes.map(plane =>
          [run.method, plane.distance_mm, plane.pcc, plane.ssim, run.iterations, run.seconds].join(','))),
      ]
      files['comparison.csv'] = strToU8(csv.join('\n') + '\n')
      const archive = Uint8Array.from(zipSync(files))
      downloadFile(archive, 'photonica-results.zip', 'application/zip')
    } catch {
      setError('Не удалось собрать архив результатов.')
    } finally {
      setBusy(null)
    }
  }

  const selectedAvailable = methods?.methods.find(entry => entry.id === method)?.available
  const iterationsValid = method === 'neural' || (
    Number.isInteger(steps[method]) && steps[method] >= (method === 'classical' ? 1 : 0)
    && steps[method] <= (methods?.max_iterations ?? 2000)
  )

  return <>
    <header className="topbar">
      <div className="page-width topbar-inner">
        <a className="brand" href="#"><span className="brand-icon"><Atom size={24} /></span>
          <span>AI Photonica <small>ЛАБОРАТОРИЯ ДОЭ</small></span></a>
        <nav aria-label="Разделы"><a href="#input">Синтез</a><a href="#results">Восстановление</a><a href="#comparison">Сравнение</a></nav>
        <div className={'status' + (health ? ' online' : connectionFailed ? ' offline' : '')}>
          <span />{health ? health.device.toUpperCase() + ' · API' : connectionFailed ? 'API недоступен' : 'Подключение'}
        </div>
      </div>
    </header>
    <main className="page-width">
      <section className="hero" aria-labelledby="page-title">
        <div className="hero-content">
          <p className="eyebrow"><span /> ВЫЧИСЛИТЕЛЬНАЯ ФОТОНИКА</p>
          <h1 id="page-title">Многоплоскостной<br /> синтез <em>ДОЭ</em></h1>
          <p className="hero-description">Одна фазовая карта для четырёх заданных изображений. Сравните методы синтеза по качеству восстановления и времени расчёта.</p>
        </div>
        <OpticalDiagram />
        <div className="optics">
          <span><small>ДЛИНА ВОЛНЫ</small><strong>532 нм</strong></span>
          <span><small>ШАГ ПИКСЕЛЯ</small><strong>10 мкм</strong></span>
          <span><small>МАТРИЦА</small><strong>128 × 128</strong></span>
          <span><small>ПЛОСКОСТИ</small><strong>4</strong></span>
        </div>
      </section>
      {connectionFailed && <div className="banner" role="status">
        Не удалось подключиться к API. <button onClick={() => void connect()}>Повторить</button>
      </div>}
      <div className="work-grid" id="input">
        <section className="panel inputs" aria-label="Целевые изображения">
          <div className="panel-header inputs-header">
            <SectionHeading index="01" title="Целевые изображения" description="Один 8-битный PNG в оттенках серого на каждую плоскость." />
            <label className="demo-select">
              <FlaskConical size={15} />
              <select aria-label="Демонстрационный набор" value={demoId ?? ''} disabled={!!busy || !demos.length}
                onChange={event => { if (event.target.value !== '') void loadDemo(Number(event.target.value)) }}>
                <option value="" disabled>Демо-набор</option>
                {demos.map(demo => <option value={demo.id} key={demo.id}>{demo.name}</option>)}
              </select>
              <ChevronDown size={14} />
            </label>
          </div>
          <div className="target-grid">
            {targets.map((file, index) => <TargetCard key={index} file={file} index={index} disabled={!!busy}
              onChange={next => changeTarget(index, next)} />)}
          </div>
          <div className="input-footer"><span><i className={inputCount === 4 ? 'all-loaded' : ''} /> Загружено {inputCount} из 4</span><span>Формат · PNG 128 × 128</span></div>
        </section>
        <section className="panel controls" aria-label="Настройки синтеза">
          <div className="panel-header">
            <SectionHeading index="02" title="Настройки" description="Выберите метод синтеза." />
            <SlidersHorizontal className="header-decoration" size={18} />
          </div>
          <div className="control-body">
            <fieldset disabled={!!busy}>
              <legend className="sr-only">Метод синтеза</legend>
              {methodIds.map(id => {
                const unavailable = id !== 'classical' && !methods?.methods.find(entry => entry.id === id)?.available
                return <label key={id} className={'method-option' + (method === id ? ' selected' : '') + (unavailable ? ' unavailable' : '')}>
                  <input name="method" type="radio" checked={method === id} disabled={unavailable} onChange={() => setMethod(id)} />
                  <span><strong>{descriptions[id].name}</strong><small>{descriptions[id].detail}</small></span>
                </label>
              })}
            </fieldset>
            {model && !model.available && <p className="hint"><CircleHelp size={14} />
              {model.status === 'load_error' ? 'Не удалось загрузить веса модели.' : 'Для Neural и Hybrid нужны обученные веса.'}
            </p>}
            {method !== 'neural' && <label className="iterations">
              <span>Итерации</span>
              <input type="number" min={method === 'classical' ? 1 : 0} max={methods?.max_iterations ?? 2000}
                value={Number.isNaN(steps[method]) ? '' : steps[method]} disabled={!!busy}
                onChange={event => setSteps({ ...steps, [method]: event.target.valueAsNumber })} />
            </label>}
            <button className="button primary" disabled={inputCount !== 4 || !selectedAvailable || !!busy || !iterationsValid}
              onClick={() => void runSynthesis()}>
              {busy === 'synthesis' ? <LoaderCircle className="spin" size={17} /> : <Play size={16} />}
              {busy === 'synthesis' ? 'Выполняется синтез…' : 'Синтезировать ДОЭ'}
              {busy !== 'synthesis' && <ArrowRight size={16} className="button-trailing" />}
            </button>
          </div>
        </section>
      </div>
      {busy === 'demo' && <div className="banner" role="status">Загрузка демонстрационного набора…</div>}
      {error && <div className="banner error" role="alert">{error}</div>}
      <section className="panel results" id="results">
        <div className="panel-header">
          <SectionHeading index="03" title="Восстановление" description="Амплитуда поля после распространения из 8-битной фазовой карты." />
          {result && <span className="method-badge">{descriptions[result.method].name}</span>}
        </div>
        {result ? <ResultView run={result} targets={validTargets} /> : <div className="empty-state">
          {busy === 'synthesis' ? <LoaderCircle className="spin" size={26} /> : <Layers3 size={26} />}
          <p>{busy === 'synthesis' ? 'Рассчитываем фазу и восстановление…' : 'Запустите синтез, чтобы увидеть фазу и четыре восстановленных изображения.'}</p>
        </div>}
      </section>
      <section className="panel comparison" id="comparison">
        <div className="panel-header">
          <SectionHeading index="04" title="Сравнение методов" description="Качество восстановления и время синтеза на одном наборе целей." />
        </div>
        <Comparison runs={completed} active={active} select={id => {
          setActive(id)
          document.getElementById('results')?.scrollIntoView({ block: 'start', behavior: 'smooth' })
        }} exportRuns={() => { if (!busy) void exportResults() }} />
      </section>
    </main>
    <footer className="page-width footer">
      <span><Atom size={16} /> AI Photonica</span>
      <span>Дифракция Френеля · одно FFT · фазовый ДОЭ</span>
    </footer>
  </>
}

createRoot(document.getElementById('root')!).render(<App />)
