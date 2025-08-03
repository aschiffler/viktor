FROM python:3.11-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/HKUDS/LightRAG.git /app/LightRAG
WORKDIR /app/LightRAG
RUN git checkout ee53e43568d40c418bb4f2e34835886b9568ca38
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env # Source the cargo env for the current shell
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install --user --no-cache-dir .[api] \
    nano-vectordb networkx \
    openai ollama tiktoken \
    pypdf2 python-docx python-pptx openpyxl
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/LightRAG/lightrag ./lightrag
COPY prompt.py ./lightrag/prompt.py
ENV PATH=/root/.local/bin:$PATH
RUN mkdir -p /app/data/rag_storage /app/data/inputs
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs
EXPOSE 9621
ENTRYPOINT ["python3", "-m", "lightrag.api.lightrag_server"]
