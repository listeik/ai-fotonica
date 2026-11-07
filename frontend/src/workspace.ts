import type { Method } from './api'

export const distances = [24.1, 25.3, 26.5, 27.7]

export const methods = {
  classical: { name: 'Classical', detail: 'Оптимизация фазовой карты методом Adam' },
  neural: { name: 'Neural', detail: 'Один проход обученной сети U-Net' },
  hybrid: { name: 'Hybrid', detail: 'Начальная фаза от U-Net с уточнением методом Adam' },
} satisfies Record<Method, { name: string; detail: string }>

export const methodIds = ['classical', 'neural', 'hybrid'] as const

export const views = {
  overview: { number: '01', title: 'Обзор', label: 'Обзор', description: 'Лаборатория многоплоскостного синтеза фазовых ДОЭ.' },
  targets: { number: '02', title: 'Целевые изображения', label: 'Целевые изображения', description: 'Задайте изображение для каждой из четырёх фокальных плоскостей.' },
  settings: { number: '03', title: 'Настройки расчёта', label: 'Настройки расчёта', description: 'Выберите метод синтеза и параметры оптимизации фазовой карты.' },
  restoration: { number: '04', title: 'Восстановление', label: 'Восстановление', description: 'Фазовая карта, амплитуда восстановленного поля и качество по плоскостям.' },
  comparison: { number: '05', title: 'Сравнение и экспорт', label: 'Сравнение и экспорт', description: 'Сопоставьте точность и время работы методов на одном наборе целей.' },
} as const

export type View = keyof typeof views

export function currentView(): View {
  const hash = window.location.hash.slice(1)
  if (hash === 'input') return 'targets'
  if (hash === 'results') return 'restoration'
  return Object.hasOwn(views, hash) ? hash as View : 'overview'
}

export const format = (value: number) => value.toFixed(3)
export const formatTime = (seconds: number) =>
  seconds < 1 ? Math.round(seconds * 1000) + ' мс' : seconds.toFixed(2) + ' с'
