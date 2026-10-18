# Open Mind Format (OMF) v0.1

OMF es un contenedor binario little-endian para modelos locales. Los lectores
deben rechazar versiones desconocidas, secciones duplicadas, longitudes
truncadas y tensores cuya forma no concuerde con sus datos. No se requiere
alineación entre registros.

## Cabecera

| Campo | Tamaño | Valor |
|---|---:|---|
| Magic | 3 bytes | ASCII `OMF` |
| Version | u16 | `1` |
| Param count | u64 | Número total de escalares almacenados en WEIGHTS |

## Secciones

Cada sección empieza con `type:u8`, seguido de `length:u64` y exactamente
`length` bytes de payload. En v0.1 el orden obligatorio es CONFIG, TOKENIZER,
WEIGHTS, METADATA; ids: `1`, `2`, `3`, `4`.

* **CONFIG (1):** objeto JSON UTF-8. Para el modelo de referencia contiene
  `hidden_size`, `layers`, `vocab`, `heads`, `intermediate_size` y `context`.
* **TOKENIZER (2):** objeto JSON UTF-8. El tokenizador de referencia declara
  `{"type":"byte","vocab":256}`; cada byte UTF-8 es un token y viceversa.
* **WEIGHTS (3):** `tensor_count:u32`; luego los registros descritos abajo.
* **METADATA (4):** objeto JSON UTF-8 (por ejemplo, modelo, paso y fecha).

Cada tensor tiene `name_len:u16`, nombre UTF-8 (`name_len` bytes), `dtype:u8`,
`rank:u8`, `rank` dimensiones `u64`, `data_len:u64` y los datos. Los pesos son
row-major, y `data_len` debe coincidir con el dtype y el número de elementos.

| dtype | Id | Codificación |
|---|---:|---|
| fp16 | 1 | IEEE-754 binary16 little-endian, 2 bytes por valor |
| q4_0 | 2 | Bloques de 32: escala binary16 + 16 bytes con dos nibbles; q = clamp(round(x/scale), -8, 7), nibble = q+8; nibble bajo corresponde al primer valor. |
| q8_0 | 3 | Bloques de 32: escala binary16 + 32 bytes int8; q = clamp(round(x/scale), -127, 127). |

Para cuantización, `scale=max(abs(block))/7` en q4_0 y `/127` en q8_0;
bloques finales incompletos se completan con ceros y al decodificar se descartan
los valores de relleno. Un bloque de ceros usa escala 0 y valores cuantizados 0
(nibble 8 en q4_0).

El lector materializa los pesos como float32 para inferencia. `param_count` es
la suma de elementos lógicos de los tensores, sin contar relleno de cuantización.
