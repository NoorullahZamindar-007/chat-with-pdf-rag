# Deployment Guide

This app is a Streamlit service with local PDF storage under `data/` and a local Chroma database under `vector_db/`.

## Local Docker

Build the image:

```powershell
docker build -t chat-with-pdf-rag .
```

Run it with your local `.env` file and persistent folders:

```powershell
docker run --env-file .env -p 8501:8501 -v ${PWD}/data:/app/data -v ${PWD}/vector_db:/app/vector_db chat-with-pdf-rag
```

Or use Compose:

```powershell
docker compose up --build
docker compose up
```

Open `http://localhost:8501`.

## Render

Use the Dockerfile deployment option. Set all `.env` values in Render's environment settings, especially `GROQ_API_KEY`. Add a persistent disk if you want to keep `data/` and `vector_db/` between deploys. For serious use, move PDFs and vector storage out of the container.

## VPS

Install Docker and Docker Compose on the server, copy the project, create `.env`, then run:

```bash
docker compose up --build
docker compose up -d
```

Put Nginx or Caddy in front of Streamlit for HTTPS and domain routing.

## Hugging Face Spaces

Create a Streamlit Space for the simplest path, or use Docker Spaces with this Dockerfile. Store secrets in Space secrets, not in the repo. Local Chroma persistence may reset depending on the Space storage setup.

## Streamlit Community Cloud

Deploy from GitHub with `app.py` as the entry point. Add secrets in Streamlit's secrets manager or environment settings. This is easiest for demos, but local files and `vector_db/` should not be treated as durable production storage.

## Production Notes

- Do not store secrets in GitHub.
- Use environment variables for API keys and provider settings.
- Add authentication before serving real users.
- Use cloud storage for PDFs instead of local `data/uploads`.
- Use PostgreSQL for document and user metadata.
- Use Qdrant for a scalable vector database.
- Add monitoring later for errors, latency, usage, and retrieval quality.

## Common Checks

- Local run: `streamlit run app.py`
- Docker build: `docker build -t chat-with-pdf-rag .`
- Docker run: `docker run --env-file .env -p 8501:8501 -v ${PWD}/data:/app/data -v ${PWD}/vector_db:/app/vector_db chat-with-pdf-rag`
- Compose run: `docker compose up --build` or `docker compose up`




