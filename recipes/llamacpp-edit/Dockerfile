# Editing recipe, containerised: llama.cpp built from the unmerged qwen4exp PR, plus the
# graph-reuse patch, for people who would rather not install a CUDA toolchain on the host.
#
#   docker build -t qwen38fn-edit -f recipes/llamacpp-edit/Dockerfile .
#
#   docker run -d --name qwen38fn-edit --gpus all --ipc=host \
#     -p 127.0.0.1:8000:8000 \
#     -v "$HOME/.qwen38fn/model:/model:ro" \
#     qwen38fn-edit
#
# The model is NOT baked in - it is ~105 GB. Fetch it on the host first with
# `./run.sh edit setup`, then mount it read-only as shown above.
#
# Building llama.cpp for GB10 takes a while. The runtime stage keeps only the binaries.

ARG CUDA_IMAGE=nvidia/cuda:13.0.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

ARG PR=27742
# PR #27742 merged upstream on 2026-08-27, and both patches in patches/ went in with it or
# were superseded by equivalent upstream work. REF=master builds the merged code with no
# patches; REF=pinned (the default) builds the exact commit the numbers in docs/ came from.
ARG REF=pinned
ARG PR_SHA=035e227
ARG CUDA_ARCH=121

RUN apt-get update && apt-get install -y --no-install-recommends \
      git cmake build-essential libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY patches/ /patches/
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
 && if [ "${REF}" = "master" ]; then \
      git fetch --depth 50 origin master --force && git checkout -q FETCH_HEAD \
      && echo "built from master; patches/ are upstream and were not applied"; \
    else \
      git fetch --depth 50 origin "pull/${PR}/head:qwen4exp" --force \
      && git checkout -q "${PR_SHA}" \
      && git apply --verbose /patches/canreuse-qwen4exp.patch \
      && echo "canreuse patch applied"; \
    fi

RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=ON \
 && cmake --build build --config Release -j"$(nproc)" --target llama-server llama-cli

FROM ${CUDA_IMAGE%-devel-*}-runtime-ubuntu24.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcurl4 libgomp1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=build /src/build/bin/llama-cli    /usr/local/bin/llama-cli

ENV MODEL=/model/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    CTX=262144 \
    SPEC=ngram-mod \
    ALIAS=qwen3.8-flash-next \
    PARALLEL=2 \
    MMPROJ=/model/mmproj-F16.gguf

EXPOSE 8000

# -ot per_layer_token_embd=CPU  keeps the 51.2B n-gram table off the GPU
# -lm mmap                      serves that table from NVMe via the page cache
# --parallel                    slots. CTX is divided across them: 2 means 131,072 tokens per
#                               request. See docs/parallel.md before raising it
# --mmproj                      vision, if the projector is mounted alongside the shards. Absent,
#                               the server starts text-only rather than failing.
SHELL ["/bin/bash", "-c"]
CMD llama-server \
      -m "$MODEL" \
      --alias "$ALIAS" \
      -lm mmap \
      -ot "per_layer_token_embd=CPU" \
      --n-gpu-layers 999 \
      --ctx-size "$CTX" \
      --parallel "$PARALLEL" \
      $( [ "$SPEC" != none ] && echo --spec-type "$SPEC" ) \
      $( [ -f "$MMPROJ" ] && echo --mmproj "$MMPROJ" ) \
      --temp 1.0 --top-p 0.95 --top-k 20 \
      --host 0.0.0.0 --port 8000
