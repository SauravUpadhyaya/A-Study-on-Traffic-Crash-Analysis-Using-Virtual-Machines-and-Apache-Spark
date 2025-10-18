FROM apache/spark:3.5.1

# Install Python dependencies (like numpy).
# You may need to run as a different user since the default user in Spark images
# may not have permissions.
USER root
RUN apt-get update && apt-get install -y python3 python3-pip python3.8-venv tar grep && \
    pip3 install numpy && \
    pip install venv-pack

WORKDIR /app
COPY requirements.txt .

RUN /usr/bin/python3 -m pip install --no-cache-dir -r requirements.txt
          
# Switch back to the non-root Spark user for security
USER 185
