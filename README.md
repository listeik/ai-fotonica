# AI Photonica

Конкурсное решение для синтеза фазового дифракционного оптического элемента (ДОЭ). Carrier U-Net принимает четыре изображения на расстояниях 24,1 / 25,3 / 26,5 / 27,7 мм и выдаёт один ДОЭ за один прямой проход. В сети девять встроенных проекций Френеля. Режим Neural выполняет ноль итераций оптимизации.

- [Исходный код нейросети](backend/src/doe/models/unet.py)
- [Обученные веса](artifacts/doe-carrier-study-v1.pt), версия `fresnel-unet-carrier-v1:d4176bb99d8c`
- [Dockerfile](infra/render.Dockerfile) и [зависимости Python](requirements.txt)

## Запуск приложения

Из корня репозитория выполните:

```sh
docker compose up --build --wait
```

Для запуска нужны Docker и Compose. Образ сам устанавливает зависимости, собирает интерфейс и включает веса. Внешняя настройка и GPU не требуются. Откройте [интерфейс](http://127.0.0.1:5173/) или [справку API](http://127.0.0.1:5173/docs). Для остановки выполните `docker compose down`.

Загрузите четыре одноканальных PNG 128×128, 8 бит, в порядке возрастания расстояния. Выберите **Neural**, запустите синтез и нажмите **Скачать ДОЭ**. Для программного вызова отправьте четыре поля `images` и поле `method=neural` как multipart-форму на `POST /api/synthesize`. Ответ содержит `phase_png` в Base64.

## Синтез для проверки без интерфейса

Поместите `plane1.png` … `plane4.png` в папку `targets` в порядке возрастания расстояния. Команды запускаются из корня репозитория:

```sh
docker compose build backend
docker compose run --no-deps --name photonica-eval -v "${PWD}/targets:/input:ro" backend python -m doe --method neural --device cpu --output /tmp/result /input/plane1.png /input/plane2.png /input/plane3.png /input/plane4.png
docker cp photonica-eval:/tmp/result/doe.png ./doe.png
docker cp photonica-eval:/tmp/result/metrics.json ./metrics.json
docker rm photonica-eval
```

Файл `doe.png` имеет размер 128×128, глубину 8 бит и формат Grayscale. Значения 0 и 255 соответствуют −π и +π. Для каждого комплекта запускайте синтез один раз. В `metrics.json` записаны время синтеза, число итераций, SSIM и Pearson для каждой из четырёх реконструкций и их средние значения. Время не включает загрузку весов и запись выходных файлов. [Формулы расчёта Френеля и метрик](docs/physics.md).

[Рабочий сайт](https://ai-fotonica.onrender.com/) использует тот же Dockerfile и веса. [Настройка CI и автодеплоя из master](docs/render.md).
