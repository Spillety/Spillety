# Инфраструктура Spillety (k3s, один хост)

Живой стенд: `ubuntu@51.250.24.23` (Ubuntu 24.04, 8 CPU, 31 ГБ RAM), k3s `v1.36.4+k3s1`, single-node. Код развёртки — `infra/` (Ansible), план и решения — `TASKS/infra-ansible/`.

## Состав

| Компонент | Где | Назначение |
|---|---|---|
| MinIO | `data`, :9000/:9001 | S3-совместимое озеро; бакеты `iceberg-warehouse`, `spark-events` |
| MySQL 8 | `data`, :3306 | backend Hive metastore (БД `metastore`) |
| Hive metastore 4.0.0 | `data`, :9083 thrift | каталог Iceberg-таблиц, warehouse `s3a://iceberg-warehouse/` |
| spark-operator | `spark-operator` | запуск `SparkApplication` |
| Cassandra 4.1 ×1 | `data`, :9042 CQL |KV-хранилище (фичи/граф-смежность в будущем) |

PVC (local-path): minio 50Gi, mysql 10Gi, cassandra 20Gi. Потребление узла в простое: ~5% CPU / 13% RAM.

## Операции

Доступ: `KYT_HOST=51.250.24.23` (IP только в env, не в коде), ключ `--private-key ~/.ssh/id_ed25519_trajectory`.

```bash
# повтор/доустановка (идемпотентно)
KYT_HOST=51.250.24.23 ansible-playbook -i infra/inventory/hosts.yml infra/site.yml --private-key ~/.ssh/id_ed25519_trajectory
# только smoke
... --tags smoke
# kubeconfig с хоста
ssh -i ~/.ssh/id_ed25519_trajectory ubuntu@51.250.24.23 "sudo cat /etc/rancher/k3s/k3s.yaml"
# консоль MinIO / CQL
kubectl -n data port-forward svc/minio 9001:9001
kubectl -n data exec statefulset/cassandra -- cqlsh
```

Пароли и версии — `infra/group_vars/all.yml` (прод: перенести секреты в `ansible-vault`).

## Проверено (smoke, 2026-09-18)

Node Ready; `minio`, `hive-metastore` Available; `mysql`, `cassandra` Ready; оба бакета на месте; thrift 9083 принимает TCP; `nodetool status` — `UN`.

## Грабли деплоя (зафиксировано, чтобы не повторять)

1. `minio/minio:latest` на Docker Hub мёртв → `quay.io/minio/minio` (+ `mc`).
2. В образе `apache/hive:4.0.0` нет MySQL-драйвера → init-контейнер качает `mysql-connector-j-8.0.33.jar` в `emptyDir` + `HIVE_AUX_JARS_PATH` (и в main-контейнер тоже).
3. Entrypoint hive-образа игнорирует `args: [metastore]` и тихо выходит 0 → запуск напрямую `hive --service metastore`.
4. K8s инжектит `METASTORE_PORT=tcp://...` (Service с именем `metastore`), Hive падает с `NumberFormatException` → явный `env METASTORE_PORT=9083`.
5. `nodetool` стартует дольше 1с → пробе `timeoutSeconds: 15`.
6. SSH рвётся на длинных wait → `ServerAliveInterval=30` в `ansible.cfg`.
7. Системный pip на Ubuntu 24.04 не даёт ставить поверх debian-пакетов → все k8s-задачи идут через `/opt/ansible-venv` (`ansible_python_interpreter`), `host_base` — через `/usr/bin/python3`, facts собираются явно (`gather_facts: false`).

## Что дальше

- I2 закрыт. Кандидат на упрощение: выкинуть HMS в пользу Iceberg JDBC-каталога на Postgres (минус JVM ~1 ГБ) — отдельной задачей.
- Старый проект (`~/v2-bot`, 13 контейнеров) остановлен `docker stop`, volumes/образы на месте; диск хоста 75% — следить.
