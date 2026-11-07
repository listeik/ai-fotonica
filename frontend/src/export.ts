import { strToU8, zipSync } from 'fflate'
import { decodeBase64, downloadFile, type Run } from './api'

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
