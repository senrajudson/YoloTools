FROM condaforge/mambaforge:24.3.0-0
SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Sao_Paulo

# Configura o fuso horário
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ >/etc/timezone

WORKDIR /app

# 1. Instala dependências do APT (Necessárias para OpenCV, Ultralytics e Albumentations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 tzdata \
 && rm -rf /var/lib/apt/lists/*

# 2. Instala Python 3.12 via Mamba (SEM o mamba clean para não bugar o mount do cache)
RUN --mount=type=cache,target=/opt/conda/pkgs \
    mamba install -y -c conda-forge \
    python=3.12 pip setuptools wheel

# 3. Instala o PyTorch (mantive a lógica do seu projeto anterior com CUDA 12.6, 
# essencial para o Ultralytics rodar rápido em GPU)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 \
    --extra-index-url https://download.pytorch.org/whl/cu126

# Instala o poetry
RUN pip install poetry

# Copia apenas os arquivos de configuração para aproveitar o cache de camadas
COPY pyproject.toml poetry.lock* README.md* ./

# Cria a env localmente
RUN poetry config virtualenvs.create true --local

# Instala as dependências de produção
RUN poetry install

# Copia o resto do código do projeto
COPY . .

# Comando para iniciar o treino
CMD ["poetry", "run", "python", "-m", "YOLO_tools.train.training_yolo"]