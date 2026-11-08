import { useEffect, useMemo, useRef, useState } from 'react'
import { CircleAlert, LoaderCircle } from 'lucide-react'
import { decodeBase64, request, targetForm, type Analysis, type Demo, type Health, type Method, type Methods, type Model, type Run } from './api'
import { AnalysisView } from './components/AnalysisView'
import { ComparisonTable } from './components/ComparisonTable'
import { Overview } from './components/Overview'
import { PassportView } from './components/PassportView'
import { ResearchView } from './components/ResearchView'
import { ResultsView } from './components/ResultsView'
import { SettingsPanel } from './components/SettingsPanel'
import { Sidebar } from './components/Sidebar'
import { TargetUpload } from './components/TargetUpload'
import { exportComparison } from './export'
import { currentView, methodIds, type Passport } from './workspace'

const headings = {
  workspace: ['Многоплоскостной синтез ДОЭ', 'Загрузите четыре цели, выберите метод и сравните восстановленные изображения.'],
  analysis: ['Анализ целей', 'Откройте этот раздел после синтеза, чтобы рассчитать характеристики текущих целей.'],
  passports: ['Паспорта синтеза', 'Технические отчёты с параметрами расчёта, фазовой картой и метриками по плоскостям.'],
  research: ['Научный режим', 'Сравните архитектуры на одном наборе и изучите поле, восстановленное из фазового ДОЭ.'],
}

export function App() {
  const [view, setView] = useState(currentView)
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
  const [busy, setBusy] = useState<'upload' | 'demo' | 'synthesis' | 'export' | 'research' | null>(null)
  const [analysis, setAnalysis] = useState<{ targets: File[]; result: Analysis | null; error: string } | null>(null)
  const [passports, setPassports] = useState<Passport[]>([])
  const [selectedPassport, setSelectedPassport] = useState('')
  const nextPassportNumber = useRef(1)
  const [error, setError] = useState('')
  const [connectionFailed, setConnectionFailed] = useState(false)
  const validTargets = useMemo(() => targets.filter((file): file is File => file !== null), [targets])
  const completed = methodIds.flatMap(id => runs[id] ? [runs[id]] : [])
  const currentAnalysis = analysis?.targets === validTargets ? analysis : null
  const hasSynthesis = passports.some(entry => entry.targets === validTargets)
  const shouldAnalyze = view === 'analysis' && hasSynthesis && !busy && !currentAnalysis
  const currentPassport = passports.find(entry => entry.id === selectedPassport) ?? passports[0]
  const activeRun = active ? runs[active] : undefined
  const activePassport = passports.find(entry => entry.run === activeRun)

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

  useEffect(() => {
    function navigate() { setView(currentView()) }
    window.addEventListener('hashchange', navigate)
    return () => window.removeEventListener('hashchange', navigate)
  }, [])

  useEffect(() => {
    const anchor = window.location.hash.slice(1)
    if (view === 'workspace' && anchor !== 'workspace' && document.getElementById(anchor)) {
      document.getElementById(anchor)?.scrollIntoView({ block: 'start' })
    } else {
      window.scrollTo({ top: 0, behavior: 'instant' })
    }
  }, [view])

  useEffect(() => {
    if (!shouldAnalyze) return
    const controller = new AbortController()
    request<Analysis>('/api/analysis', { method: 'POST', body: targetForm(validTargets), signal: controller.signal })
      .then(result => {
        if (!controller.signal.aborted) setAnalysis({ targets: validTargets, result, error: '' })
      }).catch(failure => {
        if (!controller.signal.aborted) {
          const error = failure instanceof Error ? failure.message : 'Не удалось рассчитать характеристики целей.'
          setAnalysis({ targets: validTargets, result: null, error })
        }
      })
    return () => controller.abort()
  }, [validTargets, shouldAnalyze])

  function addPassports(results: Run[], source: File[]) {
    const entries = results.map(run => ({
      id: run.id, number: nextPassportNumber.current++, targets: source, run,
    }))
    setPassports(previous => [...entries, ...previous])
    if (entries[0]) setSelectedPassport(entries[0].id)
    return entries
  }

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
        if (file.size > 16 * 1024 * 1024) throw new Error('Изображение превышает 16 МиБ.')
        const body = new FormData()
        body.append('image', file)
        const converted = await request<{ png: string; converted: boolean }>('/api/images/preprocess', { method: 'POST', body })
        if (converted.converted) {
          file = new File([decodeBase64(converted.png)], file.name.replace(/\.[^.]+$/, '') + '-128.png', { type: 'image/png' })
        }
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
    const data = targetForm(validTargets)
    data.append('method', method)
    if (method !== 'neural') data.append('iterations', String(steps[method]))

    setError('')
    setBusy('synthesis')
    try {
      const response = await request<Run>('/api/synthesize', { method: 'POST', body: data })
      setRuns(previous => ({ ...previous, [response.method]: response }))
      setActive(response.method)
      addPassports([response], validTargets)
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
    <Sidebar device={health?.device} failed={connectionFailed} view={view} />
    <main className="main-area" id={view}>
      <div className="workspace">
        <header className="page-heading">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="eyebrow"><i /> Вычислительная фотоника</span>
            {busy === 'synthesis' && <span className="chip busy-chip" role="status"><LoaderCircle size={15} className="spin" /> Идёт расчёт</span>}
          </div>
          <h1>{headings[view][0]}</h1>
          <p>{headings[view][1]}</p>
        </header>
        {connectionFailed && <div className="banner" role="status">
          <CircleAlert size={18} /><span>Не удалось подключиться к API.</span>
          <button className="text-link" type="button" onClick={() => void connect()}>Повторить</button>
        </div>}
        {view === 'analysis' && <AnalysisView result={currentAnalysis?.result ?? null} targetCount={validTargets.length}
          hasSynthesis={hasSynthesis} waiting={!!busy} error={currentAnalysis?.error ?? ''} onRetry={() => setAnalysis(null)} />}
        {view === 'passports' && <PassportView entries={passports} selected={currentPassport} onSelect={setSelectedPassport} />}
        <div hidden={view !== 'research'}>
          <ResearchView targets={validTargets} entries={passports} active={view === 'research'} disabled={!!busy}
            onBusy={pending => setBusy(pending ? 'research' : null)} onRuns={addPassports} />
        </div>
        {view === 'workspace' && <div className="synthesis-workspace">
          {error && <div className="banner error" role="alert"><CircleAlert size={18} />{error}</div>}
          <div className="workbench" id="input">
            <TargetUpload targets={targets} demos={demos} demoId={demoId}
              converting={busy === 'upload'}
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
              <div className="flex flex-wrap gap-3">
                {activePassport && <a className="text-link text-xs" href="#passports" onClick={() => setSelectedPassport(activePassport.id)}>Паспорт расчёта</a>}
                {completed.length > 0 && <a className="text-link text-xs" href="#settings">К настройкам</a>}
              </div>
            </div>
            <ResultsView run={activeRun} runs={completed}
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
        </div>}
      </div>
    </main>
  </div>
}
