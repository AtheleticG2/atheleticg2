# Use the official Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and app files to the container
COPY requirements.txt /app/requirements.txt
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8501

# Set environment variables to reduce Streamlit warnings
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/streamlit_alt.py"]
