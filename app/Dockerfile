# Use official Python image
FROM python:3.11.4-bookworm

# Set working directory
WORKDIR /root/app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all app code, models, and data
COPY ./code ./code
COPY ./models ./models
COPY ./data ./data

# Expose port Dash will run on
EXPOSE 8050

# Default command to run your Dash app
CMD ["python3", "code/app.py"]
