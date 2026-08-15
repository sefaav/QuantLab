# QuantLab container image.
# Build: docker build -t quantlab .
# Run the offline demo:
#   docker run --rm quantlab backtest --shipped-config demo_offline

FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

# Compilers remain confined to the builder for packages without compatible wheels.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install "uv==0.12.3"

COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
# Hatchling resolves forced includes while installing the non-editable package.
COPY configs ./configs
COPY data/raw/SPY.csv data/raw/QQQ.csv data/raw/TLT.csv data/raw/GLD.csv ./data/raw/

# Consume the same lockfile as CI and install only runtime dashboard/data extras.
RUN uv sync --locked --no-dev --extra dashboard --extra yahoo --no-editable


FROM python:3.14-slim@sha256:ce40764625a4ff50df3548277632e7f96c4e77fe75fa848aae9885476e7df5a4 AS runtime

ARG QUANTLAB_UID=1000
ARG QUANTLAB_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_PYTHON_REFRESH=quiet \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PATH=/opt/venv/bin:$PATH \
    HOME=/home/quantlab

RUN groupadd --gid "$QUANTLAB_GID" quantlab \
    && useradd --uid "$QUANTLAB_UID" --gid "$QUANTLAB_GID" \
        --create-home --shell /usr/sbin/nologin quantlab

COPY --from=builder /opt/venv /opt/venv

RUN mkdir -p \
        /home/quantlab/.quantlab/data/raw \
        /home/quantlab/.quantlab/data/processed \
        /home/quantlab/.quantlab/data/metadata \
        /home/quantlab/.quantlab/data/cache \
        /home/quantlab/.quantlab/reports/generated \
        /home/quantlab/.quantlab/reports/figures \
        /home/quantlab/.quantlab/reports/tables \
        /home/quantlab/.quantlab/logs \
    && chown -R quantlab:quantlab /home/quantlab

USER quantlab
WORKDIR /home/quantlab

EXPOSE 8501

ENTRYPOINT ["quantlab"]
CMD ["--help"]
