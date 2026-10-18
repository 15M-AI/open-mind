# Contribuir

Gracias por ayudar a Open Mind. El proyecto está en fase v0.1 y el formato OMF
especificado en `open-mind-format/OMF_SPEC.md` debe seguir siendo compatible
entre el writer Python y el reader Rust.

## Cambios

- Mantén el código y los datos de prueba locales; no añadas pesos de modelos,
  credenciales, datos privados ni archivos de corpus sin permiso.
- Para cambios de formato, actualiza la especificación y ambos lados del
  intercambio; añade una prueba de round-trip o de rechazo de datos inválidos.
- Para cambios Rust, ejecuta `cargo test --workspace`.
- Para cambios Python, instala en entorno virtual y ejecuta el flujo de
  entrenamiento con el preset `tiny` y un corpus de prueba local.
- Describe el comportamiento observable y cualquier limitación añadida.

## Reportar problemas

Incluye el sistema operativo, versión de Rust/Python, comando ejecutado y un
ejemplo pequeño que no contenga datos privados ni checkpoints.
