# Use Python 3.12 slim image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Install git
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Clone the updated pd-explain-agentic repository and install it
RUN git clone https://github.com/benayat/pd-explain-agentic.git && \
    cd pd-explain-agentic && \
    pip install --no-cache-dir -e .

# Install Jupyter Notebook
RUN pip install --no-cache-dir jupyter
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Command to run when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]