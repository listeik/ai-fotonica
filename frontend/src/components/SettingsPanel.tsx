import { ArrowRight, Check, CircleHelp, Cpu, GitMerge, ImagePlus, LoaderCircle, Play, SlidersHorizontal, Sparkles } from 'lucide-react'
import type { Method, Methods, Model } from '../api'
import { methodIds, methods as descriptions } from '../workspace'

const icons = { classical: Cpu, neural: Sparkles, hybrid: GitMerge }

export function SettingsPanel({ method, steps, methods, model, targetCount, busy, running, onMethod, onSteps, onRun }: {
  method: Method
  steps: number
  methods: Methods | null
  model: Model | null
  targetCount: number
  busy: boolean
  running: boolean
  onMethod: (method: Method) => void
  onSteps: (method: 'classical' | 'hybrid', count: number) => void
  onRun: () => void
}) {
  const minimum = method === 'classical' ? 1 : 0
  const maximum = methods?.max_iterations ?? 2000
  const stepsValid = Number.isInteger(steps) && steps >= minimum && steps <= maximum
  const selectedAvailable = methods?.methods.some(entry => entry.id === method && entry.available)
  const canRun = targetCount === 4 && selectedAvailable && !busy && (method === 'neural' || stepsValid)

  return <div className="grid gap-5 sm:gap-6">
    <section className="panel settings-panel">
      <h2>Метод синтеза</h2>
      <fieldset className="method-grid" disabled={busy}>
        <legend className="sr-only">Метод синтеза</legend>
        {methodIds.map(id => {
          const Icon = icons[id]
          const unavailable = id !== 'classical' && !methods?.methods.some(entry => entry.id === id && entry.available)
          return <label className="method-card" key={id}>
            <input type="radio" name="method" checked={method === id} disabled={unavailable}
              aria-label={descriptions[id].name} aria-describedby={id + '-detail'} onChange={() => onMethod(id)} />
            <span className="flex items-start justify-between gap-2">
              <span className="icon-tile"><Icon size={20} strokeWidth={1.8} /></span>
              <span className="radio-indicator" aria-hidden="true">{method === id && <Check size={13} />}</span>
            </span>
            <strong>{descriptions[id].name}</strong>
            <small id={id + '-detail'}>{descriptions[id].detail}</small>
            {unavailable && <span className="unavailable-label">Недоступен</span>}
          </label>
        })}
      </fieldset>
      {model && !model.available && <p className="hint"><CircleHelp size={17} />
        {model.status === 'load_error' ? 'Не удалось загрузить веса модели.' : 'Для Neural и Hybrid нужны обученные веса.'}
      </p>}
    </section>
    {method !== 'neural' && <section className="panel settings-panel">
      <div className="flex items-start gap-3">
        <span className="icon-tile"><SlidersHorizontal size={18} /></span>
        <div><h2>Параметры Adam</h2><p className="muted mt-1 text-sm">Число шагов уточнения фазовой карты.</p></div>
      </div>
      <div className="iteration-controls">
        <div>
          <label htmlFor="iterations-range" className="font-semibold text-sm">Итерации</label>
          <input id="iterations-range" className="iteration-range" type="range" min={minimum} max={maximum}
            value={Number.isFinite(steps) ? Math.min(maximum, Math.max(minimum, steps)) : minimum} disabled={busy}
            onChange={event => onSteps(method, event.currentTarget.valueAsNumber)} />
          <div className="range-labels"><span>{minimum}</span><span>{maximum}</span></div>
        </div>
        <label className="number-field">
          <span>Точное значение</span>
          <input type="number" min={minimum} max={maximum} step={1} value={Number.isNaN(steps) ? '' : steps}
            disabled={busy} aria-invalid={!stepsValid} onChange={event => onSteps(method, event.currentTarget.valueAsNumber)} />
        </label>
      </div>
    </section>}
    <div className="panel action-bar">
      <div className="flex items-center gap-3">
        <span className="icon-tile"><ImagePlus size={18} /></span>
        <div><strong>Цели · {targetCount} / 4</strong>
          {targetCount !== 4
            ? <a className="text-link block text-xs" href="#targets">Загрузить недостающие изображения</a>
            : <p className="muted text-xs">{running ? 'Рассчитываем фазу и восстановление' : 'Набор готов к расчёту'}</p>}
        </div>
      </div>
      <button type="button" className="button primary" disabled={!canRun} onClick={onRun}>
        {running ? <LoaderCircle className="spin" size={17} /> : <Play size={17} />}
        {running ? 'Выполняется синтез…' : 'Синтезировать ДОЭ'}
        {!running && <ArrowRight size={16} />}
      </button>
    </div>
  </div>
}
