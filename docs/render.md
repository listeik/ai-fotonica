# Render и CI/CD

Сервис [AI Photonica](https://ai-fotonica.onrender.com/) использует [render.yaml](../render.yaml), [Dockerfile](../infra/render.Dockerfile) и ветку `master`. Render запускает автодеплой после успешных проверок GitHub Actions. CI проверяет backend, сборку frontend и синтез в Docker с лимитом памяти 512 МБ. Health check `/api/ready` подтверждает загрузку весов.

Для развёртывания в другом workspace подключите GitHub с доступом к репозиторию. В Render выберите **New → Blueprint**, репозиторий и ветку `master`, затем **Deploy Blueprint**. Тариф и регион заданы в Blueprint. Бесплатный экземпляр может засыпать при простое.

[История релизов и логи](https://dashboard.render.com/web/srv-dbnrofrrjlhs739mehng/deploys). Проверка работающего сервиса:

```sh
python backend/scripts/smoke_deployment.py --url https://ai-fotonica.onrender.com --timeout 120
```
