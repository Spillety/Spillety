# data/

- `data/elliptic_raw/` — сырые csv Elliptic++ (658M, в `.gitignore`, не трекается).
- `archive.zip` — исходник автора в корне (тоже в `.gitignore`).

Ничего руками качать не нужно — один шаг:

```bash
make data
# или: python scripts/download_elliptic.py
```

Скрипт сам найдёт `archive.zip` → распакует, иначе попробует `kaggle API`.
Сырьё read-only: ноутбуки его не меняют (читают через `docs/notebooks/_elliptic_loader.py`).
