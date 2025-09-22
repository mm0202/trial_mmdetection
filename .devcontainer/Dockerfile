FROM mcr.microsoft.com/devcontainers/anaconda:0.203.3-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/

RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

RUN pip install -U openmim
RUN mim install mmengine

RUN pip uninstall -y numpy pandas \
    && pip install --no-cache-dir numpy==1.24.4 pandas==1.5.3

RUN mim install "mmcv>=2.0.0"

RUN pip uninstall -y numpy pandas \
    && pip install --no-cache-dir numpy==1.24.4 pandas==1.5.3

RUN mim install mmdet