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


COPY prompt.py /app/LightRAG/lightrag/prompt.py
COPY azure_openai.py /app/LightRAG/lightrag/llm/azure_openai.py


RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . $HOME/.cargo/env
ENV PATH="/root/.cargo/bin:${PATH}"


RUN pip install --user --no-cache-dir .[api] \
    nano-vectordb networkx \
    openai ollama tiktoken \
    pypdf2 python-docx python-pptx openpyxl

FROM python:3.11-slim
WORKDIR /app


COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/LightRAG/lightrag ./lightrag


ENV PATH=/root/.local/bin:$PATH
RUN mkdir -p /app/data/rag_storage /app/data/inputs
ENV WORKING_DIR=/app/data/rag_storage
ENV INPUT_DIR=/app/data/inputs

EXPOSE 9621
ENTRYPOINT ["python3", "-m", "lightrag.api.lightrag_server"]
