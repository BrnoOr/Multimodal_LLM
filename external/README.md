# external/

Código de terceros como submodules fijados a un commit (no se copia ni se modifica).

| Submodule | Origen | Commit |
|---|---|---|
| `open-vljepa/` | https://github.com/dion-jy/open-vljepa | `f92107fa3d188d69e2fd8ca0222c06483b09a518` |

Primera vez: `git submodule add https://github.com/dion-jy/open-vljepa external/open-vljepa && git -C external/open-vljepa checkout f92107f`.
Clones posteriores: `git submodule update --init --recursive`.
