# Seguridad

Open Mind v0.1 está pensado para ejecución local. OMA escucha en
`127.0.0.1:8080` por defecto y no implementa autenticación, autorización ni TLS.
No lo expongas a una red no confiable. No envíes rutas o prompts sensibles a
instancias que no controles.

Los archivos OMF se leen como entrada no confiable: usa modelos de fuentes de
confianza y reporta fallos de parsing o asignaciones excesivas con un caso
reproducible que no contenga pesos privados.

Para reportar una vulnerabilidad, contacta al mantenedor en privado antes de
publicar detalles explotables. No incluyas credenciales, modelos privados ni
datos personales en los reportes.
