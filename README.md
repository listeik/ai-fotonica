# AI Photonica

Синтез одного фазового дифракционного оптического элемента (ДОЭ) из четырёх изображений. Каждое изображение задаёт амплитуду на своей плоскости: 24,1; 25,3; 26,5 и 27,7 мм. Лазер — 532 нм, шаг модулятора — 10 мкм.

| Метод | Как рассчитывается фаза |
|---|---|
| Classical | Adam оптимизирует пиксели фазы с воспроизводимой случайной инициализацией. |
| Neural | Один проход Physics-Informed Multi-plane Residual Fourier U-Net. Сеть корректирует фазу с учётом комплексных невязок Френеля на четырёх плоскостях и спектральных признаков. |
| Hybrid | Начинает с фазы Neural и уточняет её тем же Adam, что и Classical. Лучший кандидат выбирается по общей функции потерь на 8-битной фазе. |

Все методы экспортируют фазу в одноканальный PNG 128×128, где 0 соответствует −π, а 255 — +π. Четыре PNG восстановления строятся из декодированной фазы. Метрики PCC и SSIM сравнивают их с загруженными целями. Подробнее — в [описании модели](docs/physics.md).

## Запуск через Docker Compose

Выполните команду из корня репозитория.

```sh
docker compose up --build
```

Compose поднимает интерфейс и API, дождавшись загрузки обученных весов. Веса уже входят в образ backend, расчёт выполняется на CPU. Если порт 5173 занят, задайте `DOE_WEB_PORT` в окружении или в файле `.env`. Для остановки используйте `docker compose down`. [Интерфейс](http://127.0.0.1:5173) · [Справка API](http://127.0.0.1:5173/docs).

## Локальная разработка

Понадобятся Python 3.11, Node.js 24 и pnpm 11.25.0. Из корня репозитория в PowerShell:

```powershell
py -3.11 -m venv .venv
$python = ".\.venv\Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
& $python -m pip install -e "./backend[dev,benchmark]"
& $python -m uvicorn doe.api:app --port 8000
```

В другом терминале:

```powershell
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend dev
```

Откройте [интерфейс](http://localhost:5173) или [справку API](http://localhost:8000/docs). Выберите готовый набор или загрузите четыре PNG (128×128, 8 бит, один канал). Для конкурсного запуска в форме по умолчанию выбран Neural, если веса загружены. На той же странице можно сравнить методы и скачать архив с фазой, реконструкциями и метриками.

Backend и CLI по умолчанию используют `artifacts/doe-residual-fourier.pt`. Это чекпойнт Residual Fourier с seed 2027 и восьмой эпохи, выбранный по минимальной валидационной ошибке 0,460969 среди трёх запусков. SHA-256 — `38cf98d0ceb539d8ca27498f83e81be9b90b04bce1c287bb47b02b6d09a1f630`. Ветка подключает новую архитектуру по запросу пользователя. В сравнительном эксперименте Projection v2 получила лучшие метрики качества, а Fourier — меньшую задержку. [Результаты исследования](docs/multiplane-study.md). Путь можно изменить через `DOE_CHECKPOINT`, устройство — через `DOE_DEVICE` (`auto`, `cpu` или `cuda`). Если веса не загружены, доступен Classical, а запросы Neural и Hybrid возвращают HTTP 503. Для GPU установите подходящую сборку по [инструкции PyTorch](https://pytorch.org/get-started/previous-versions/#v271).

## Командная строка и эксперименты

```powershell
& $python -m doe --demo 0 --method classical --output runs/example
& $python -m doe --method hybrid --output runs/example-hybrid plane1.png plane2.png plane3.png plane4.png
& $python -m doe.quality plane1.png plane2.png plane3.png plane4.png --phase runs/example-hybrid/doe.png --output runs/quality
& $python -m pytest backend
& $python -m doe.training.train --config configs/projected.json --device auto --output runs/training
& $python -m doe.benchmark --checkpoint artifacts/doe-residual-fourier.pt --device cpu --samples 32 --output runs/comparison
```

Файлы целей перечисляются в порядке роста расстояния. Обучение использует воспроизводимые синтетические наборы: геометрические фигуры, контуры, символы и полутоновые цели. Потери объединяют MSE амплитуды, PCC и локальный SSIM; градиент проходит через квантование с прямой оценкой производной и через модель Френеля. Разбиения для обучения, валидации, тестирования и демонстрации используют разные потоки генератора. Веса выбираются по валидации и сохраняются в `best.pt`; `last.pt` позволяет продолжить обучение с `--resume`. Для подключения своего чекпойнта укажите путь в `DOE_CHECKPOINT`. Для Projection v2 можно задать `DOE_CHECKPOINT=artifacts/doe-unet.pt`.

[Протокол бенчмарка](docs/benchmark.md) фиксирует одинаковые цели, нормировку и измерение времени для всех методов. См. [результаты экспериментов](docs/results.md) и [пример выходных PNG](experiments/example). Примеры и результаты моделирования не заменяют оптическую проверку на реальном модуляторе.

Для длительного итеративного поиска используйте [`doe.quality`](docs/quality.md). Он пробует несколько стартовых фаз и сохраняет лучший проверенный 8-битный ДОЭ, а при заданном эталоне — также лучший вариант, сохраняющий средний и худший PCC эталона. Передавайте только фазовые PNG, рассчитанные для тех же четырёх целей.

Для дообучения Neural с учётом оптической мощности на фоне см. [конфигурацию и результаты пилота](docs/neural-finetune.md).
