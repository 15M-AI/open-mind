# Open Mind

**Open Mind** es un ecosistema local para entrenar, guardar y ejecutar modelos
en el formato binario propio **OMF**. Está organizado como un monorepo con tres
componentes y no descarga modelos ni utiliza GGUF.

> **Estado:** v0.1 experimental. El runtime no tiene todavía streaming,
autenticación ni TLS. Mantén el servidor ligado a `127.0.0.1` salvo que sepas
cómo proteger la red donde lo expones.

## Componentes

| Proyecto | Contenido |
|---|---|
| `open-mind-format` | Especificación OMF v0.1, lector/escritor Rust y writer PyTorch |
| `open-mind-runtime` | Inferencia CPU propia, CLI `om` y API OMA local |
| `open-mind-train` | Entrenamiento causal PyTorch, dataset mmap y checkpoints OMF |

## Requisitos

- Rust estable y Cargo.
- Python 3.10+ y PyTorch local (`pip install -e ./open-mind-train` instala sus dependencias si faltan).
- Datos de entrenamiento en un archivo de texto UTF-8.

No se descargan pesos. Para evitar escribir en el Python global, se recomienda
usar un entorno virtual:

```bash
cd ~/OpenMind
python -m venv .venv
source .venv/bin/activate
pip install -e ./open-mind-train
```

## Entrenar un modelo

El repositorio incluye `data.txt` como muestra mínima para comprobar el flujo;
no sirve para producir un modelo conversacional útil. Usa un corpus local más
amplio que tengas derecho a utilizar:

```bash
python train.py --device cpu --dataset corpus.txt --output pico.omf \
  --steps 5000 --checkpoint-every 500
```

El preset `15M` tiene 14.983.680 parámetros. En CPU, el entrenamiento usa
acumulación de gradiente y momentum SGD cuantizado a 8 bits; los pesos del
modelo permanecen en memoria durante el entrenamiento. El writer permite
checkpoints `fp16` (predeterminado), `q4_0` y `q8_0`. Los checkpoints
periódicos se guardan como `pico-v1.omf` y el final como `pico.omf`.

## Instalar y ejecutar un modelo

Compila el ejecutable local `om`:

```bash
cargo build --release
./target/release/om --help
```

Instala un OMF desde una ruta local. La instalación copia el archivo a
`~/.open-mind/models`; los pesos no se incluyen en este repositorio:

```bash
./target/release/om add pico.omf
./target/release/om list
./target/release/om show pico.omf
./target/release/om run pico.omf --prompt "Hola" --max-tokens 64
```

`om run MODEL` sin `--prompt` abre un modo interactivo; escribe una indicación
por línea y usa `Ctrl+D` para salir. Parámetros de generación:

```bash
./target/release/om run pico.omf --prompt "Hola" \
  --temperature 0.8 --top-k 40 --seed 123
```

Temperatura `0` selecciona greedy. Una semilla fija reproduce la secuencia
muestreada; cambiarla no mejora el conocimiento aprendido por el modelo.
También se puede establecer `OPEN_MIND_HOME=/ruta/local` para cambiar el
directorio donde `om add` y `om list` guardan/encuentran modelos.

## Servir la API OMA

```bash
./target/release/om serve --model pico.omf
```

Escucha por defecto en `127.0.0.1:8080`. Si hay exactamente un modelo instalado,
`om serve` puede resolverlo sin `--model`. Prueba el endpoint compatible con
OpenAI:

```bash
curl http://127.0.0.1:8080/v1/generate \
  -H 'content-type: application/json' \
  -d '{"model":"pico.omf","prompt":"Hola","max_tokens":32,"temperature":0.8,"top_k":40}'
```

También se expone `/v1/completions`. `max_tokens` tiene un máximo de 1024;
`temperature` acepta de 0 a 2. OMA v0.1 devuelve una respuesta JSON completa,
sin streaming.

## Estructura del monorepo

```text
OpenMind/
├── Cargo.toml                   # workspace Rust
├── open-mind-format/
│   ├── OMF_SPEC.md
│   ├── src/                     # lector y escritor Rust
│   └── python/src/open_mind_format/
├── open-mind-runtime/
│   └── src/main.rs              # CLI `om`, inferencia y API OMA
├── open-mind-train/
│   ├── pyproject.toml
│   └── src/open_mind/
├── train.py
└── data.txt                     # corpus de demostración, no pesos
```

## Calidad y limitaciones

Comprueba el workspace Rust y el writer/reader:

```bash
cargo test --workspace
```

El modelo es un transformer de bytes entrenado desde cero. No es un sustituto
de modelos preentrenados ni podrá responder con conocimiento que no aparezca
en sus datos de entrenamiento. El runtime ejecuta inferencia en CPU; el uso de
un preset de 15M puede ser lento en portátiles modestos.

## Publicación en GitHub

El diseño es un **monorepo**, por lo que se recomienda crear un solo repositorio:
`open-mind` (URL sugerida: `github.com/TU_USUARIO/open-mind`). Los nombres de
componente se conservan en sus carpetas:
`open-mind-format`, `open-mind-runtime` y `open-mind-train`. El publicador se
encuentra fuera de este repositorio en `~/open-mind-publisher/`.

Antes de hacer público el proyecto, elige y añade una licencia si quieres
autorizar la reutilización. No se incluye una licencia por defecto.
