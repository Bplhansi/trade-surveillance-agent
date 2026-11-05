# One image for the whole demo: the C++ engine plus the Python parts
# (simulator, ML, note service, dashboard). Built in two stages so the compilers stay out
# of the final image.

# --- Stage 1: build the C++ engine ------------------------------------------------------
FROM python:3.12-slim-trixie AS engine-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake make ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY engine/ engine/
# Downloads GoogleTest and ONNX Runtime (pinned by SHA-256 in engine/CMakeLists.txt).
RUN cmake -S engine -B build -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build --target tsa_replay -j"$(nproc)" \
 && mkdir -p /out/lib \
 && cp build/tsa_replay /out/ \
 && cp -P build/_deps/onnxruntime-src/lib/libonnxruntime.so* /out/lib/

# --- Stage 2: runtime --------------------------------------------------------------------
FROM python:3.12-slim-trixie
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY --from=engine-build /out/lib/ /usr/local/lib/
COPY --from=engine-build /out/tsa_replay /usr/local/bin/tsa_replay
RUN ldconfig
COPY config/ config/
COPY sim/ sim/
COPY ml/ ml/
COPY service/ service/
COPY dashboard/ dashboard/
COPY scripts/ scripts/
COPY pyproject.toml .
# Run as a normal user, not root. data/ and models/ are filled at run time (volumes).
RUN useradd --create-home --uid 1000 tsa \
 && mkdir -p data models \
 && chown tsa:tsa data models
USER tsa
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
