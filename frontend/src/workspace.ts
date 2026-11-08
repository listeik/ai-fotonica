import type { Method, Run } from './api'

export type View = 'workspace' | 'analysis' | 'passports' | 'research'

export interface Passport {
  id: string
  number: number
  targets: File[]
  run: Run
}

export function currentView(): View {
  const hash = window.location.hash.slice(1)
  return hash === 'analysis' || hash === 'passports' || hash === 'research' ? hash : 'workspace'
}

export const distances = [24.1, 25.3, 26.5, 27.7]

export const methods = {
  classical: { name: 'Classical', detail: 'Оптимизация фазовой карты методом Adam' },
  neural: { name: 'Neural', detail: 'Один проход обученной U‑Net' },
  hybrid: { name: 'Hybrid', detail: 'Начальная фаза от U-Net с уточнением методом Adam' },
} satisfies Record<Method, { name: string; detail: string }>

export const methodIds = ['classical', 'neural', 'hybrid'] as const

export function passportLabel({ number, run }: Passport) {
  const model = run.checkpoint ? ` · ${run.checkpoint.architecture}` : ''
  return `№${number} · ${methods[run.method].name}${model} · ${new Date(run.created_at).toLocaleString('ru-RU')}`
}

export const format = (value: number) => value.toFixed(3)
export const formatTime = (seconds: number) =>
  seconds < 1 ? Math.round(seconds * 1000) + ' мс' : seconds.toFixed(2) + ' с'
