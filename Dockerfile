# Hugging Face Spaces (Docker SDK): https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.13-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user \
    && mkdir -p /home/user/app \
    && chown -R user:user /home/user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install locked dependencies (app is not an installable package; use uv export + uv pip).
RUN python -m pip install --user --no-cache-dir uv

COPY --chown=user pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev -o requirements.txt \
    && uv pip install --user --no-cache-dir -r requirements.txt \
    && rm requirements.txt

COPY --chown=user . .

EXPOSE 7860

# Single worker: Dash session state; threads handle concurrent static/asset requests.
CMD ["gunicorn", "app:server", "-b", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180"]
