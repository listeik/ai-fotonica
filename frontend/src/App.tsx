import { useEffect, useState } from 'react'
import { CircleAlert, LoaderCircle } from 'lucide-react'
import { decodeBase64, request, type Demo, type Health, type Method, type Methods, type Model, type Run } from './api'
import { ComparisonTable } from './components/ComparisonTable'
import { Overview } from './components/Overview'
import { ResultsView } from './components/ResultsView'
import { SettingsPanel } from './components/SettingsPanel'
import { Sidebar } from './components/Sidebar'
import { TargetUpload } from './components/TargetUpload'
import { exportComparison } from './export'
import { methodIds } from './workspace'

async function validateTarget(file: File) {
  if (!file.name.toLowerCase().endsWith('.png') || file.size > 64 * 1024) {
    throw new Error('Выберите PNG размером до 64 КиБ.')
  }
  const header = new DataView(await file.slice(0, 26).arrayBuffer())
  if (header.byteLength < 26 || header.getUint32(0) !== 0x89504e47 || header.getUint32(4) !== 0x0d0a1a0a
    || header.getUint32(8) !== 13 || header.getUint32(12) !== 0x49484452
    || header.getUint32(16) !== 128 || header.getUint32(20) !== 128
    || header.getUint8(24) !== 8 || header.getUint8(25) !== 0) {
    throw new Error('Нужен одноканальный 8-битный PNG размером 128 × 128 пикселей.')
  }
}

export function App() {
  const [targets, setTargets] = useState<(File | null)[]>(Array(4).fill(null))
  const [method, setMethod] = useState<Method>('neural')
  const [steps, setSteps] = useState({ classical: 160, hybrid: 40 })
  const [runs, setRuns] = useState<Partial<Record<Method, Run>>>({})
  const [active, setActive] = useState<Method | null>(null)
  const [methods, setMethods] = useState<Methods | null>(null)
  const [health, setHealth] = useState<Health | null>(null)
  const [model, setModel] = useState<Model | null>(null)
  const [demos, setDemos] = useState<Demo[]>([])
  const [demoId, setDemoId] = useState<number | null>(null)
  const [busy, setBusy] = useState<'upload' | 'demo' | 'synthesis' | 'export' | null>(null)
  const [error, setError] = useState('')
  const [connectionFailed, setConnectionFailed] = useState(false)
  const validTargets = targets.filter((file): file is File => file !== null)
  const completed = methodIds.flatMap(id => runs[id] ? [runs[id]] : [])

  async function connect(signal?: AbortSignal) {
    setConnectionFailed(false)
    try {
      const [nextHealth, nextMethods, nextModel, nextDemos] = await Promise.all([
        request<Health>('/api/health', { signal }),
        request<Methods>('/api/methods', { signal }),
        request<Model>('/api/model/info', { signal }),
        request<Demo[]>('/api/demo', { signal }),
      ])
      if (signal?.aborted) return
      setHealth(nextHealth)
      setMethods(nextMethods)
      setModel(nextModel)
      setDemos(nextDemos)
      if (!nextMethods.methods.some(entry => entry.id === 'neural' && entry.available)) {
        setMethod(selected => selected === 'neural' ? 'classical' : selected)
      }
    } catch {
      if (signal?.aborted) return
      setConnectionFailed(true)
      setHealth(null)
      setMethods(null)
      setModel(null)
      setDemos([])
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

  async function changeTarget(index: number, file: File | null) {
    if (busy) return
    if (file) {
      setBusy('upload')
      try {
        await validateTarget(file)
      } catch (failure) {
        setError(failure instanceof Error ? failure.message : 'Не удалось прочитать PNG.')
        return
      } finally {
        setBusy(null)
      }
    }
    clearResults()
    setTargets(previous => previous.map((target, position) => position === index ? file : target))
    setDemoId(null)
  }

  async function loadDemo(index: number) {
    if (busy) return
    setBusy('demo')
    setError('')
    try {
      const demo = await request<Demo & { images: string[] }>('/api/demo/' + index)
      clearResults()
      setTargets(demo.images.map((image, plane) =>
        new File([decodeBase64(image)], `demo-${index + 1}-z${plane + 1}.png`, { type: 'image/png' })))
      setDemoId(index)
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Не удалось загрузить набор.')
    } finally {
      setBusy(null)
    }
  }

  async function runSynthesis() {
    if (busy || validTargets.length !== 4) return
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
      document.getElementById('restoration')?.scrollIntoView({ block: 'start' })
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Синтез не выполнен.')
    } finally {
      setBusy(null)
    }
  }

  async function exportResults() {
    if (busy) return
    setBusy('export')
    setError('')
    try {
      await exportComparison(validTargets, completed)
    } catch {
      setError('Не удалось собрать архив результатов.')
    } finally {
      setBusy(null)
    }
  }

  return <div className="app-shell">
    <Sidebar device={health?.device} failed={connectionFailed} />
    <main className="main-area" id="workspace">
      <div className="workspace">
        <header className="page-heading">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="eyebrow"><i /> Вычислительная фотоника</span>
            {busy === 'synthesis' && <span className="chip busy-chip" role="status"><LoaderCircle size={15} className="spin" /> Идёт расчёт</span>}
          </div>
          <h1>Многоплоскостной синтез ДОЭ</h1>
          <p>Загрузите четыре цели, выберите метод и сравните восстановленные изображения.</p>
        </header>
        {connectionFailed && <div className="banner" role="status">
          <CircleAlert size={18} /><span>Не удалось подключиться к API.</span>
          <button className="text-link" type="button" onClick={() => void connect()}>Повторить</button>
        </div>}
        {error && <div className="banner error" role="alert"><CircleAlert size={18} />{error}</div>}
        <div className="synthesis-workspace">
          <div className="workbench" id="input">
            <TargetUpload targets={targets} demos={demos} demoId={demoId}
              disabled={!!busy} loadingDemo={busy === 'demo'} onChange={(index, file) => void changeTarget(index, file)}
              onDemo={index => void loadDemo(index)} />
            <SettingsPanel method={method} steps={method === 'neural' ? 0 : steps[method]}
              methods={methods} model={model} targetCount={validTargets.length} busy={!!busy}
              running={busy === 'synthesis'} onMethod={setMethod}
              onSteps={(id, count) => setSteps(previous => ({ ...previous, [id]: count }))} onRun={() => void runSynthesis()} />
          </div>
          <section id="restoration" className="workspace-block" aria-labelledby="restoration-heading">
            <div className="section-heading">
              <h2 id="restoration-heading">Восстановление</h2>
              {completed.length > 0 && <a className="text-link text-xs" href="#settings">К настройкам</a>}
            </div>
            <ResultsView run={active ? runs[active] : undefined} runs={completed}
              targets={validTargets} running={busy === 'synthesis'} onSelect={setActive} />
          </section>
          <section id="comparison" className="workspace-block" aria-labelledby="comparison-heading">
            <div className="section-heading"><h2 id="comparison-heading">Сравнение и экспорт</h2></div>
            <ComparisonTable runs={completed} active={active} busy={!!busy}
              exporting={busy === 'export'} onSelect={id => {
                setActive(id)
                document.getElementById('restoration')?.scrollIntoView({ block: 'start' })
              }} onExport={() => void exportResults()} />
          </section>
          <Overview optics={health?.optics} />
        </div>
      </div>
    </main>
  </div>
}
