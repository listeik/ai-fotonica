import { useState } from 'react'
import { ArrowRight, Check, FileImage, FlaskConical, ImagePlus, LoaderCircle, X } from 'lucide-react'
import type { Demo } from '../api'
import { useObjectUrl } from '../hooks/useObjectUrl'
import { distances } from '../workspace'

function TargetCard({ index, distance, file, disabled, onChange }: {
  index: number
  distance: number
  file: File | null
  disabled: boolean
  onChange: (file: File | null) => void
}) {
  const url = useObjectUrl(file)
  const [dragging, setDragging] = useState(false)

  return <article className="panel target-card">
    <div className="flex items-center justify-between gap-2">
      <span className="step-number">{String(index + 1).padStart(2, '0')}</span>
      <span className="plane-distance">{distance.toFixed(1)} мм</span>
    </div>
    <label className={'target-drop' + (file ? ' filled' : '') + (dragging && !disabled ? ' dragging' : '')}
      onDragOver={event => {
        event.preventDefault()
        if (!disabled) event.dataTransfer.dropEffect = 'copy'
      }}
      onDragEnter={() => setDragging(true)}
      onDragLeave={event => {
        if (!(event.relatedTarget instanceof Node) || !event.currentTarget.contains(event.relatedTarget)) {
          setDragging(false)
        }
      }}
      onDrop={event => {
        event.preventDefault()
        setDragging(false)
        if (!disabled && event.dataTransfer.files[0]) onChange(event.dataTransfer.files[0])
      }}>
      <input type="file" accept=".png,image/png" disabled={disabled}
        aria-label={`Загрузить цель для плоскости ${index + 1}`}
        onChange={event => {
          if (event.currentTarget.files?.[0]) onChange(event.currentTarget.files[0])
          event.currentTarget.value = ''
        }} />
      {file && url
        ? <img src={url} alt={`Цель ${index + 1}`} />
        : <span className="upload-prompt">
          <span className="icon-tile"><ImagePlus size={22} /></span>
          <strong>Выбрать PNG</strong>
          <small>или перетащить сюда</small>
        </span>}
    </label>
    <div className="file-meta">
      <FileImage size={17} aria-hidden="true" />
      <div>
        <strong title={file?.name}>{file?.name ?? 'Ожидает загрузки'}</strong>
        <small>128 × 128 · {file ? `${(file.size / 1024).toFixed(1)} КиБ` : '8 бит'}</small>
      </div>
      {file && <button type="button" className="icon-button" disabled={disabled}
        aria-label={`Удалить цель ${index + 1}`} onClick={() => onChange(null)}><X size={16} /></button>}
    </div>
  </article>
}

export function TargetUpload({ targets, demos, demoId, disabled, loadingDemo, onChange, onDemo }: {
  targets: (File | null)[]
  demos: Demo[]
  demoId: number | null
  disabled: boolean
  loadingDemo: boolean
  onChange: (index: number, file: File | null) => void
  onDemo: (index: number) => void
}) {
  const count = targets.filter(Boolean).length

  return <div className="grid gap-5 sm:gap-6">
    <section className="panel demo-panel">
      <div className="flex items-start gap-3">
        <span className="icon-tile"><FlaskConical size={19} /></span>
        <div>
          <h2>Демонстрационные наборы</h2>
          <p className="muted mt-1 text-sm">Один набор заполняет все четыре плоскости.</p>
        </div>
      </div>
      <div className="flex flex-wrap gap-2" aria-label="Демонстрационные наборы">
        {demos.map(demo => <button className="chip selectable" type="button" key={demo.id}
          disabled={disabled} aria-pressed={demoId === demo.id} onClick={() => onDemo(demo.id)}>
          {demo.name}
        </button>)}
        {loadingDemo && <LoaderCircle className="spin" size={19} aria-label="Загрузка демо-набора" />}
        {!demos.length && <span className="muted text-sm">Наборы доступны при подключении к API.</span>}
      </div>
    </section>
    <div className="target-grid">
      {distances.map((distance, index) => <TargetCard key={distance} index={index} distance={distance}
        file={targets[index] ?? null} disabled={disabled} onChange={file => onChange(index, file)} />)}
    </div>
    <div className="panel action-bar">
      <div className="flex items-center gap-3">
        <span className={`progress-chip ${count === 4 ? 'complete' : ''}`}>
          {count === 4 && <Check size={13} />}{count} / 4
        </span>
        <div><strong>Целевые изображения</strong><p className="muted text-xs">PNG · 128 × 128 · 8 бит · один канал</p></div>
      </div>
      <a href="#settings" className={`button ${count === 4 ? 'primary' : 'secondary'}`}>
        К настройкам <ArrowRight size={16} />
      </a>
    </div>
  </div>
}
