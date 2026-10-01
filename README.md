# Unified Async LLM Client

Cliente unificado y asíncrono para trabajar con diferentes proveedores de modelos de lenguaje mediante una interfaz común.

## Objetivo

El proyecto implementa una arquitectura desacoplada para utilizar proveedores LLM diferentes sin modificar la lógica principal de la aplicación.

Actualmente se contemplan:

- OpenAI
- Anthropic

## Características

- Python 3.12+
- Arquitectura asíncrona con `async` / `await`
- Interfaz común mediante una clase abstracta
- Clientes independientes para OpenAI y Anthropic
- Streaming mediante `AsyncIterator`
- Validación de datos con Pydantic
- Configuración mediante variables de entorno
- Manejo controlado de errores de conexión, API y rate limit
- Selección del proveedor mediante configuración

## Estructura

```text
Unified Async LLM Client/
│
├── .env.example
├── .gitignore
├── anthropic_client.py
├── base_client.py
├── main.py
├── manager.py
├── openai_client.py
├── requirements.txt
├── schemas.py
└── README.md