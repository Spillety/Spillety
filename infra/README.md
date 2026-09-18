# infra — data-платформа Spillety на одном хосте (k3s + Ansible)

Стек: k3s, MinIO (S3 под Iceberg), MySQL 8 (backend Hive metastore), Hive metastore 4 (thrift 9083), spark-operator, Cassandra 4.1. Pig нет (EOL), HiveServer2 нет (Iceberg ходит в metastore напрямую).

## Требования к хосту

Ubuntu 22.04/24.04, ≥16 ГБ RAM, ≥4 CPU, ≥100 ГБ диска, интернет, sudo без пароля (или `--ask-become-pass`).

## Запуск

```bash
pip install ansible kubernetes
ansible-galaxy collection install -r infra/requirements.yml
export KYT_HOST=158.160.29.228   # или -e kyt_host=... ; IP вне кода
ansible-playbook -i infra/inventory/hosts.yml infra/site.yml --private-key ~/.ssh/id_ed25519_trajectory
```

Частично: `--tags minio,mysql,metastore`. Проверка отдельно — роль `smoke` уже в конце `site.yml`.

## Что где

- `data` ns: `minio:9000` (API) / `:9001` (console), `mysql:3306`, `metastore:9083` (thrift), `cassandra:9042` (CQL).
- Бакеты: `iceberg-warehouse` (warehouse HMS `s3a://iceberg-warehouse/`), `spark-events`.
- Пароли/версии: `group_vars/all.yml` — прод: вынести секреты в `ansible-vault`.

## Пример Iceberg-таблицы (ручной запуск)

```bash
kubectl -n spark create serviceaccount spark   # один раз, если нет
kubectl apply -f infra/roles/spark/files/spark-iceberg-init.yaml
kubectl -n spark get sparkapplications -w
```

## Сверка версий

```bash
helm search repo kubeflow -l   # актуальный spark-operator chart
```

Все версии — переменные в `group_vars/all.yml`, план и решения — `TASKS/infra-ansible/`.
