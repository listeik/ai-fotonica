export type Method = 'classical' | 'neural' | 'hybrid'

export interface PlaneMetrics {
  distance_mm: number
  pcc: number
  ssim: number
}

export interface Run {
  method: Method
  model_version: string | null
  phase_png: string
  reconstruction_pngs: string[]
  metrics: {
    planes: PlaneMetrics[]
    mean_pcc: number
    mean_ssim: number
    worst_pcc: number
    worst_ssim: number
  }
  seconds: number
  iterations: number
  parameters: {
    seed: number
    device: string
    size_px: number
    wavelength_nm: number
    pixel_pitch_um: number
    distances_mm: number[]
    reconstruction: string
  }
}

export interface Health {
  status: 'ok'
  device: string
  optics: Omit<Run['parameters'], 'seed' | 'device'>
}

export interface Methods {
  methods: { id: Method; available: boolean; uses_iterations: boolean }[]
  max_iterations: number
}

export interface Model {
  available: boolean
  version: string | null
  status: 'ready' | 'missing' | 'load_error'
}

export interface Demo {
  id: number
  name: string
}

export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    const detail = body?.detail
    if (typeof detail === 'string') throw new Error(detail)
    if (Array.isArray(detail)) {
      throw new Error(detail.map(item => item.msg).join('; '))
    }
    throw new Error(`Сервер вернул HTTP ${response.status}.`)
  }
  return response.json()
}

export function decodeBase64(value: string): Uint8Array<ArrayBuffer> {
  const bytes = atob(value)
  return Uint8Array.from(bytes, character => character.charCodeAt(0))
}

export function pngUrl(value: string): string {
  return `data:image/png;base64,${value}`
}

export function downloadFile(data: BlobPart, filename: string, type: string): void {
  const url = URL.createObjectURL(new Blob([data], { type }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
