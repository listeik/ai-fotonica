import { Activity, LoaderCircle } from 'lucide-react'
import type { Analysis, ImageDescriptors } from '../api'
import { distances, format } from '../workspace'

const measures: { key: keyof ImageDescriptors; label: string; unit?: string }[] = [
  { key: 'contrast', label: 'RMS-контраст' },
  { key: 'entropy_bits', label: 'Энтропия', unit: 'бит' },
  { key: 'edge_fraction', label: 'Плотность границ', unit: '%' },
  { key: 'spectral_centroid', label: 'Центр спектра', unit: 'цикл/пикс.' },
  { key: 'high_frequency_fraction', label: 'Высокие частоты', unit: '%' },
]

function valueLabel(value: number, unit?: string) {
  return unit === '%' ? (value * 100).toFixed(1) + '%' : format(value) + (unit ? ' ' + unit : '')
}

export function AnalysisView({ result, targetCount, error, onRetry }: {
  result: Analysis | null
  targetCount: number
  error: string
  onRetry: () => void
}) {
  if (error) return <div className="banner error" role="alert">
    <Activity size={18} /> {error} <button className="text-link" onClick={onRetry}>Повторить</button>
  </div>
  if (!result) return <section className="panel detail-panel">
    {targetCount < 4
      ? <p>Загружено {targetCount} из 4 целей. <a className="text-link" href="#targets">Добавить изображения</a></p>
      : <p className="hint" role="status"><LoaderCircle size={18} className="spin" /> Вычисляем характеристики целей…</p>}
  </section>

  return <div className="detail-stack">
    <section className="panel detail-panel">
      <h2>Характеристики по плоскостям</h2>
      <div className="table-scroll">
        <table>
          <thead><tr><th scope="col">Показатель</th>{distances.map(distance => <th scope="col" key={distance}>{distance.toFixed(1)} мм</th>)}</tr></thead>
          <tbody>{measures.map(({ key, label, unit }) => <tr key={key}>
            <th scope="row">{label}</th>{result.planes.map((plane, index) => <td key={index}>{valueLabel(plane[key], unit)}</td>)}
          </tr>)}</tbody>
        </table>
      </div>
      <p className="muted text-sm">RMS-контраст — стандартное отклонение яркости в диапазоне 0–1. Границы выделены фильтром Собеля с порогом {result.edge_threshold}. Доля высоких частот считается по энергии спектра от {result.high_frequency_cutoff} цикл/пикс. без постоянной составляющей.</p>
    </section>
    <section className="panel detail-panel">
      <h2>Синтетическая выборка для сравнения</h2>
      <p className="muted text-sm">Ориентир рассчитан по {result.reference.sets} наборам из генератора обучающей выборки, seed {result.reference.seed}. Сравнение описывает свойства изображений. Прогноз SSIM требует отдельной проверки на данных.</p>
      <div className="table-scroll">
        <table>
          <thead><tr><th scope="col">Показатель</th><th scope="col">Среднее целей</th><th scope="col">Медиана выборки</th><th scope="col">P10–P90</th><th scope="col">Положение</th></tr></thead>
          <tbody>{measures.map(({ key, label, unit }) => {
            const mean = result.planes.reduce((sum, plane) => sum + plane[key], 0) / result.planes.length
            const [low, median, high] = result.reference.bands[key]
            return <tr key={key}><th scope="row">{label}</th><td>{valueLabel(mean, unit)}</td>
              <td>{valueLabel(median, unit)}</td><td>{valueLabel(low, unit)} – {valueLabel(high, unit)}</td>
              <td>{mean < low ? 'Ниже диапазона' : mean > high ? 'Выше диапазона' : 'В диапазоне'}</td></tr>
          })}</tbody>
        </table>
      </div>
    </section>
  </div>
}
