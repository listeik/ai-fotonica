import { strToU8, zipSync } from 'fflate'
import { decodeBase64, downloadFile, type Run } from './api'
import type { Passport } from './workspace'

export async function exportPassport({ id, targets, run }: Passport) {
  const files: Record<string, Uint8Array> = { 'doe.png': decodeBase64(run.phase_png) }
  const planes = await Promise.all(run.metrics.planes.map(async (metrics, index) => {
    const source = targets[index]
    const image = run.reconstruction_pngs[index]
    if (!source || !image) throw new Error('Паспорт содержит неполные данные.')
    const target = `targets/z${index + 1}.png`
    const reconstruction = `reconstructions/z${index + 1}.png`
    files[target] = new Uint8Array(await source.arrayBuffer())
    files[reconstruction] = decodeBase64(image)
    return { ...metrics, target, target_filename: source.name, reconstruction }
  }))
  const { phase_png: _phase, reconstruction_pngs: _reconstructions, metrics, ...summary } = run
  files['passport.json'] = strToU8(JSON.stringify({
    schema_version: 1,
    ...summary,
    phase: { file: 'doe.png', bits: 8, min_rad: -Math.PI, max_rad: Math.PI },
    metrics: { ...metrics, planes },
  }, null, 2))
  downloadFile(Uint8Array.from(zipSync(files)), `photonica-${id}.zip`, 'application/zip')
}

export async function exportComparison(targets: File[], runs: Run[]) {
  const files: Record<string, Uint8Array> = {}
  for (const [index, target] of targets.entries()) {
    files[`targets/z${index + 1}.png`] = new Uint8Array(await target.arrayBuffer())
  }
  const summaries = runs.map(run => {
    const { phase_png, reconstruction_pngs, ...summary } = run
    files[`${run.method}/doe.png`] = decodeBase64(phase_png)
    reconstruction_pngs.forEach((png, index) => {
      files[`${run.method}/z${index + 1}.png`] = decodeBase64(png)
    })
    return summary
  })
  files['comparison.json'] = strToU8(JSON.stringify(summaries, null, 2))
  const rows = [
    'method,distance_mm,pcc,ssim,iterations,seconds',
    ...runs.flatMap(run => run.metrics.planes.map(plane =>
      [run.method, plane.distance_mm, plane.pcc, plane.ssim, run.iterations, run.seconds].join(','))),
  ]
  files['comparison.csv'] = strToU8(rows.join('\n') + '\n')
  downloadFile(Uint8Array.from(zipSync(files)), 'photonica-results.zip', 'application/zip')
}
