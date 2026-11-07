# Render: хостинг и CI/CD из master

Один Docker Web Service обслуживает собранный React-интерфейс и FastAPI на общем HTTPS-адресе. Модель `fresnel-unet-carrier-v1:d4176bb99d8c` включена в образ как `artifacts/doe-carrier-study-v1.pt`; скачивание весов при запуске не требуется. Изображения обрабатываются в памяти, постоянный диск не нужен.

## Первое подключение Render

1. Авторизуйтесь в Render и подключите GitHub с доступом к приватному репозиторию `listeik/ai-fotonica`.
2. Создайте **New → Blueprint**, выберите этот репозиторий, ветку **master** и файл `render.yaml`. Также можно открыть [готовую ссылку настройки](https://render.com/deploy?repo=https://github.com/listeik/ai-fotonica/tree/master).
3. Проверьте конфигурацию: Docker, `infra/render.Dockerfile`, регион Frankfurt, тариф **Free**, health check `/api/ready`, Auto-Deploy **After CI Checks Pass**.
4. Создайте сервис. Адрес вида `https://…onrender.com` появится в панели сервиса после его создания. Для автодеплоя нужен именно подключённый Git provider, а не режим Public Git Repository URL.

Конфигурация не содержит секретов и не требует deploy hook. Доступ Render к приватному GitHub-репозиторию предоставляется через подключение аккаунта. Создать сам сервис без авторизации владельца Render невозможно.

## Автоматический деплой

```text
Изменение/merge в master
  → GitHub Actions: backend (Ruff + pytest), frontend (TypeScript + Vite)
  → сборка Render Docker-образа
  → HTTP smoke-тест контейнера с лимитом 512 МБ
  → все CI checks успешны
  → Render автоматически собирает master и развёртывает сервис
  → /api/ready подтверждает готовность нейросети
```

`render.yaml` содержит `branch: master` и `autoDeployTrigger: checksPass`. Ошибка тестов, frontend-сборки, контейнерной сборки или smoke-теста блокирует автоматический деплой. Pull requests также проверяются, но Render развёртывает только связанную ветку `master`. Действия публикации выполняет Render после CI; токен Render в GitHub Secrets не нужен.

`/api/health` проверяет процесс API. `/api/ready` отвечает HTTP 503, пока checkpoint недоступен, и HTTP 200 с версией загруженной модели, когда нейросеть готова. Именно `/api/ready` используется для Render health checks.

## Ресурсы

Используется Free Web Service: один worker Uvicorn и один CPU-поток PyTorch. Локальный smoke-тест прошёл с ограничением 512 МБ; после нейросетевого синтеза контейнер занимал около 183 МиБ. Это проверка памяти, а не гарантия производительности на CPU Render.

Бесплатный сервис засыпает после 15 минут без входящих запросов; последующий запуск может занимать около минуты. Для постоянной доступности тариф меняется отдельно владельцем аккаунта. [Ограничения Free](https://render.com/docs/free).

## Проверка и локальная сборка

Из корня репозитория:

```sh
docker build -f infra/render.Dockerfile -t ai-fotonica:render .
docker run --name ai-fotonica-render --memory=512m -p 127.0.0.1:10000:10000 ai-fotonica:render
```

В другом терминале:

```sh
python backend/scripts/smoke_deployment.py --url http://127.0.0.1:10000 --timeout 120
```

Тот же smoke-тест можно запустить против реального HTTPS-адреса Render. Он проверяет готовность модели, HTML-интерфейс, совпадение версии нейросети и нейрорежим с нулём внешних итераций, фазовым PNG и четырьмя 128×128 8-битными реконструкциями.

Для отката откройте Deploys в Render и выберите предыдущий успешный deploy. При повторяющейся ошибке новой версии сначала отключите Auto-Deploy, затем верните рабочий commit в `master` обычным revert-коммитом.

Документация: [Blueprint](https://render.com/docs/blueprint-spec), [деплой после CI](https://render.com/docs/deploys#automatic-deploys), [Docker](https://render.com/docs/docker).
