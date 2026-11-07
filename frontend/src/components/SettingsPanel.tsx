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

  return <section className="panel settings-panel" id="settings" aria-labelledby="settings-heading">
    <div className="section-heading">
      <h2 id="settings-heading">Настройки расчёта</h2>
      <SlidersHorizontal size={17} className="muted" />
    </div>
    <fieldset className="method-grid" disabled={busy}>
      <legend className="sr-only">Метод синтеза</legend>
      {methodIds.map(id => {
        const Icon = icons[id]
        const unavailable = id !== 'classical' && !methods?.methods.some(entry => entry.id === id && entry.available)
        return <label className="method-card" key={id}>
          <input type="radio" name="method" checked={method === id} disabled={unavailable}
            aria-label={descriptions[id].name} aria-describedby={id + '-detail'} onChange={() => onMethod(id)} />
          <span className="icon-tile"><Icon size={16} strokeWidth={1.8} /></span>
          <span className="method-text">
            <strong>{descriptions[id].name}</strong>
            <small id={id + '-detail'}>{descriptions[id].detail}</small>
            {unavailable && <span className="unavailable-label">Недоступен</span>}
          </span>
          <span className="radio-indicator" aria-hidden="true">{method === id && <Check size={13} />}</span>
        </label>
      })}
    </fieldset>
    {model && !model.available && <p className="hint"><CircleHelp size={17} />
      {model.status === 'load_error' ? 'Не удалось загрузить веса модели.' : 'Для Neural и Hybrid нужны обученные веса.'}
    </p>}
    {method !== 'neural' &&
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
      </div>}
    <div className="settings-actions">
      <button type="button" className="button primary" disabled={!canRun} onClick={onRun}>
        {running ? <LoaderCircle className="spin" size={17} /> : <Play size={17} />}
        {running ? 'Выполняется синтез…' : 'Синтезировать ДОЭ'}
        {!running && <ArrowRight size={16} />}
      </button>
      {targetCount !== 4 && <a className="text-link flex items-center gap-1.5 text-xs" href="#targets">
        <ImagePlus size={14} /> Загрузите все четыре цели
      </a>}
    </div>
  </section>
}
