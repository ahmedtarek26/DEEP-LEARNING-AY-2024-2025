# Use an official Python image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Create a virtual environment
RUN python -m venv /app/venv

# Activate the virtual environment and install dependencies
RUN /app/venv/bin/pip install --no-cache-dir numpy torch torchvision scikit-learn matplotlib

# Set environment variables so the container uses the venv
ENV PATH="/app/venv/bin:$PATH"

# Default command
CMD ["python"]

