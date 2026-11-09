# Agentic RAG Enterprise Knowledge Assistant: FastAPI backend serving the built React frontend.
#
#   docker build -t knowledge-assistant .
#   docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-ant-... -v knowledge-assistant-data:/app/data knowledge-assistant
#
# Then open http://localhost:8000. Uploaded documents, the index and
# conversations live in the /app/data volume.

FROM node:24-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim
WORKDIR /app

# CPU-only PyTorch: the default Linux wheel bundles CUDA and adds gigabytes.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Download the embedding model at build time so the container starts offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY backend/ backend/
COPY --from=frontend /app/frontend/dist frontend/dist

VOLUME /app/data
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
