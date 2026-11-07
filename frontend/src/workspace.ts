import type { Method } from './api'

export const distances = [24.1, 25.3, 26.5, 27.7]

export const methods = {
  classical: { name: 'Classical', detail: 'Оптимизация фазовой карты методом Adam' },
  neural: { name: 'Neural', detail: 'Один проход обученной U‑Net' },
  hybrid: { name: 'Hybrid', detail: 'Начальная фаза от U-Net с уточнением методом Adam' },
} satisfies Record<Method, { name: string; detail: string }>

export const methodIds = ['classical', 'neural', 'hybrid'] as const

export const format = (value: number) => value.toFixed(3)
export const formatTime = (seconds: number) =>
  seconds < 1 ? Math.round(seconds * 1000) + ' мс' : seconds.toFixed(2) + ' с'
