# Carrier-conditioned portrait DOE experiment

2026-11-07. Selected model: `fresnel-unet-carrier-v1:6d0a3dcefcf7`.

## Result

Conditioning the first encoder on the fixed optical carrier produced a small portrait SSIM gain. Its frozen weighted held-out score is **0.398208**, versus **0.395209** for the original and **0.395295** for the paired zero-carrier control. This supports a modest average-quality tradeoff; it does not establish statistical significance or uniform improvement.

The validation winner was frozen before test evaluation. It met the predeclared deployment rule: improve the weighted held-out score over original and matched control, while keeping every group mean PCC/global/local SSIM drop below 0.003. The largest such drops were 0.000814 versus original and 0.001595 versus matched control.

| Test group | Quartets | Original PCC / global SSIM / local SSIM | Carrier PCC / global SSIM / local SSIM |
|---|---:|---|---|
| people_test | 10 | 0.559341 / 0.461161 / 0.164511 | 0.559453 / 0.467878 / 0.165023 |
| mixed_test | 5 | 0.645530 / 0.483562 / 0.183981 | 0.645054 / 0.484711 / 0.183425 |
| broad_test | 128 | 0.442375 / 0.317843 / 0.175017 | 0.441649 / 0.317105 / 0.174203 |
| broad_ood | 64 | 0.478698 / 0.350871 / 0.221717 | 0.478510 / 0.351936 / 0.222786 |
| fresh_broad_test | 128 | 0.444859 / 0.313229 / 0.171399 | 0.444645 / 0.313817 / 0.171220 |

These results use CUDA neural inference followed by exact 8-bit phase PNG export, CPU Fresnel propagation, peak normalization, and 8-bit reconstruction PNG export. The production artifact has identical CPU phase outputs to the selected research prototype on fixed verification inputs. A separate production CPU report is retained under `runs/photo-quality-v2/production-cpu.json`.

**Tail limitation:** p10 of the weakest PCC/global SSIM decreases in all five groups. For portraits it changes from 0.357535 to 0.354409; for fresh broad, from 0.078567 to 0.077880. Broad means remain close to the original. The visible speckle and cross-plane interference remain substantial.

## Architecture and training

The deployed model is a width-16 Fresnel U-Net with **484,182 parameters** and six learned physical projection layers. Encoder input is now 12 channels: four targets plus sine and cosine of the four fixed phase carriers. The previous model had 483,030 parameters. New input-channel weights started at zero, preserving the original phase prediction exactly; the remaining weights came from `doe-photo-projection-v1.pt`.

The winning run used the original optical objective: 1−PCC + 1−global SSIM, a 0.25 weakest-plane term, 0.1 amplitude MSE and 0.05 background-power penalty. Phase and image quantization use straight-through gradients. Optimized phase labels were used in separate unsuccessful distillation arms. The winner learned from target images and the differentiable physical reconstruction loss.

Winning run: seed 20261113, AdamW, LR 0.0002, weight decay 0.00001, cosine LR down to 10%, 16 epochs × 64 batches × 16 quartets. Best checkpoint: epoch 11. Each batch contained eight sampled new fixed training quartets, four new combinations of four distinct training people, and four broad replay quartets with random plane assignments. Fixed quartets comprise 256 new portrait combinations and 256 broad training stacks. Portrait bank: 32 training people × 10 views; broad replay cache: 16,000 quartets.

## Controlled comparisons

Ten training runs compared original-loss replay, reconstruction-teacher transfer, dense tone/peak penalties, combined penalties, warm-start phase-teacher transfer, width-32 expansion, and zero/real carrier input. All starts used the same original portrait checkpoint; paired ablations shared seed, batch sequence, optimizer, training budget and validation rule.

| Variant | Best epoch | Validation score |
|---|---:|---:|
| Original | 0 | 0.395934 |
| control | 7 | 0.398762 |
| teacher | 3 | 0.397626 |
| tone | 3 | 0.398891 |
| combined | 10 | 0.396751 |
| replay_control | 16 | 0.398723 |
| phase_teacher | 15 | 0.397255 |
| wide_control | 16 | 0.396962 |
| wide_phase | 8 | 0.396703 |
| carrier_zero | 16 | 0.398233 |
| carrier_guided | 11 | 0.399483 |

Offline teachers for 512 train-only combinations were optimized for 240 Adam steps from the frozen neural prediction. Every saved teacher improved its per-quartet selection score. They raised training portrait global SSIM from 0.47985 to 0.58722, but neither direct phase transfer nor reconstruction transfer delivered a consistent validation advantage over matched ordinary training. Width 32 also failed to improve the result.

Validation score: 0.5 people + 0.2 mixed + 0.3 broad. Inside each group: 0.6 mean(PCC, global SSIM) + 0.25 mean minimum of the eight PCC/global-SSIM plane values + 0.15 mean local SSIM. Held-out promotion score uses the same formula and replaces broad validation with fresh broad test.

## Data and uncertainty

Preparation verified 65,280 cached broad planes by pixel SHA-256, 150 Mixed parent links, disjoint portrait identities and broad origins across splits, and exact reconstruction of 120 existing teachers from exported phases. Fresh broad excludes previously evaluated original/Fourier broad test sources and Mixed test source origins. It has 128 quartets and 504 unique origins across 512 planes.

Portrait test has 10 quartets containing the same four people. Mixed-5 reuses sources from the portrait and broad test collections. Fresh broad is a new subset of the existing source-disjoint test distribution. Bootstrap intervals resample quartets, do not reflect four-person identity uncertainty, and are not corrected for adaptive validation or multiple comparisons. Only one training seed was tested. The portrait global-SSIM interval versus the paired carrier-zero control includes zero.

Global SSIM uses population moments over each complete image. Local SSIM uses uniform 7×7 windows, sample covariance correction 49/48, data_range=1 and constants 0.01²/0.03². No metric definition was changed to generate the gain.

## Inference and deployment

Neural mode calls `model.forward` once and performs zero optimizer/refinement iterations after it. Inside that forward remain six fixed learned optical projection layers. A requirement that also prohibits an unrolled projection structure needs separate interpretation; this experiment does not resolve that ambiguity.

Production checkpoint: `artifacts/doe-photo-carrier-v1.pt`. Standard loader supports the new explicit `fresnel-unet-carrier-v1` architecture and strict carrier metadata. Existing checkpoint behavior is preserved. 35 focused model, training, synthesis and API tests pass, including a carrier API test that counts one forward and verifies zero refinement iterations.

Compose project `miffi2026_2` uses image `miffi2026_2-backend:photo-carrier-v1` and a read-only checkpoint mount. The image derives from the previously running backend and replaces only the model package. Live API confirmed the new content hash, valid phase and four reconstruction PNGs, and iterations=0. One first portrait smoke request took about 0.18 s on CPU; this is not a latency benchmark.

Rollback from the original checkout: `docker compose -p miffi2026_2 -f compose.yaml -f runs/compose-photo-carrier-v1.rollback.yaml up -d --no-build`.

## Local reproduction artifacts

`runs/photo-quality-v2/` contains preparation, training, teacher, widening, carrier, evaluation and preview scripts; data manifest; all histories and checkpoints; frozen selection; full per-quartet comparison; production CPU validation and live smoke. These local run files are ignored by Git. `photo-carrier-v1-results.json` is the compact tracked result snapshot with source/checkpoint hashes. Existing source datasets are required; their absolute cache paths are retained in the local manifest.

Train/validation SHA-256: `02d940b542849ec1d04df7e0ead0ecb93b3935525e36d57a9710972eb5bf0b84`.

Test SHA-256: `033d7b4fa9fc2a87eaaeb48cb4dea9b9010935ef2ffa9e4d21bc7b3c61229d0b`.

Production checkpoint SHA-256: `6d0a3dcefcf7a4c87d014902f87f4e18a9109e3837a2c4f77542e3599a87ab1e`.
