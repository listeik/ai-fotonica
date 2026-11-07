import { useState } from 'react'
import { ArrowRight, Focus, Grid3X3, Layers3, Waves } from 'lucide-react'
import type { Health } from '../api'
import { distances } from '../workspace'

const positions = [348, 422, 496, 570]

function OpticalDiagram({ plane, onSelect }: { plane: number; onSelect: (index: number) => void }) {
  return <svg className="optics-svg" viewBox="0 0 628 220" aria-hidden="true">
    <defs>
      <linearGradient id="optics-beam">
        <stop stopColor="#a45bc7" />
        <stop offset="1" stopColor="#6d54c8" stopOpacity=".25" />
      </linearGradient>
      <linearGradient id="optics-glass" x2="1" y2="1">
        <stop stopColor="#d5baf7" stopOpacity=".76" />
        <stop offset="1" stopColor="#f5ecff" stopOpacity=".18" />
      </linearGradient>
      <pattern id="optics-grid" width="10" height="10" patternUnits="userSpaceOnUse">
        <path d="M0 0h10M0 0v10" fill="none" stroke="#a989d1" strokeOpacity=".42" />
      </pattern>
    </defs>
    <path d="M38 105H600" stroke="#ac95ca" strokeDasharray="4 6" />
    <g>
      <rect x="23" y="91" width="64" height="28" rx="8" fill="#eee2fc" stroke="#bf9ee6" />
      <circle cx="87" cy="105" r="6" fill="#944ebe" />
      <text x="55" y="160" textAnchor="middle">Лазер</text>
    </g>
    {Array.from({ length: 9 }, (_, index) => {
      const offset = index - 4
      return <path key={index} fill="none" stroke="url(#optics-beam)" strokeWidth={index === 4 ? 2 : 1}
        d={`M87 ${105 + offset} L198 ${105 + offset * 6} Q385 ${105 + offset * 10} 592 ${105 + offset * 8}`} />
    })}
    <path d="M171 62l55-25v116l-55 25z" fill="url(#optics-glass)" stroke="#8961bc" strokeWidth="1.6" />
    <path d="M171 62l55-25v116l-55 25z" fill="url(#optics-grid)" />
    <text x="198" y="199" textAnchor="middle">ПВМС</text>
    {positions.map((x, index) => <g key={x} className={plane === index ? 'plane-node selected' : 'plane-node'}
      onClick={() => onSelect(index)}>
      <path d={`M${x - 20} 67l40-18v96l-40 18z`} fill="url(#optics-glass)" />
      <circle cx={x} cy="105" r={plane === index ? 14 : 9} fill="#efe2ff" />
      <circle cx={x} cy="105" r="4" fill="#8254b7" stroke="none" />
      <text x={x} y="188" textAnchor="middle">z{index + 1}</text>
    </g>)}
  </svg>
}

export function Overview({ optics }: { optics?: Health['optics'] }) {
  const [plane, setPlane] = useState(0)
  const values = [
    { label: 'Длина волны', value: `${optics?.wavelength_nm ?? 532} нм`, icon: Waves },
    { label: 'Шаг пикселя', value: `${optics?.pixel_pitch_um ?? 10} мкм`, icon: Focus },
    { label: 'Размер матрицы', value: `${optics?.size_px ?? 128} × ${optics?.size_px ?? 128}`, icon: Grid3X3 },
    { label: 'Фокальные плоскости', value: String(distances.length), icon: Layers3 },
  ]

  return <div className="grid gap-5 sm:gap-6">
    <section className="overview-hero panel">
      <div className="hero-copy">
        <span className="eyebrow"><i /> Вычислительная фотоника</span>
        <h2>Многоплоскостной<br className="hidden xl:block" /> синтез <em>ДОЭ</em></h2>
        <p>Синтезируйте одну фазовую карту для четырёх целевых изображений. Сравните качество и скорость восстановления разными методами.</p>
        <a className="button primary" href="#targets">Подготовить цели <ArrowRight size={17} /></a>
      </div>
      <div className="optical-path">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3>Оптическая схема</h3>
          <span className="chip">Френель · 1 FFT</span>
        </div>
        <OpticalDiagram plane={plane} onSelect={setPlane} />
        <div className="plane-picker" role="group" aria-label="Фокальная плоскость">
          {distances.map((distance, index) => <button key={distance} type="button"
            className="chip selectable" aria-pressed={index === plane} onClick={() => setPlane(index)}>
            {distance.toFixed(1)} мм
          </button>)}
        </div>
        <p className="optics-caption" aria-live="polite">Плоскость {plane + 1} · z = {distances[plane]?.toFixed(1)} мм</p>
      </div>
    </section>
    <section aria-label="Параметры оптики" className="stats-grid">
      {values.map(({ label, value, icon: Icon }) => <article className="panel stat-card" key={label}>
        <span className="icon-tile"><Icon size={19} strokeWidth={1.8} /></span>
        <div><p>{label}</p><strong>{value}</strong></div>
      </article>)}
    </section>
    <section className="panel workflow">
      <h2>Рабочий процесс</h2>
      <div className="workflow-grid">
        {[
          { id: 'targets', number: '01', name: 'Загрузите цели', detail: 'Четыре PNG или готовый демо-набор.' },
          { id: 'settings', number: '02', name: 'Запустите расчёт', detail: 'Выберите метод и число итераций.' },
          { id: 'comparison', number: '03', name: 'Сравните результаты', detail: 'Проверьте метрики и выгрузите архив.' },
        ].map(step => <a key={step.id} href={'#' + step.id} className="workflow-link">
          <span className="step-number">{step.number}</span>
          <span><strong>{step.name}</strong><small>{step.detail}</small></span>
          <ArrowRight size={15} />
        </a>)}
      </div>
    </section>
  </div>
}
