# 1. Base Image: Use Python 3.10
FROM python:3.10-slim

# 2. Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8080

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Install system dependencies
#    --> THIS IS THE NEW SECTION <--
# This installs g++ and other tools needed to compile insightface
RUN apt-get update && apt-get install -y build-essential

# 5. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy your application code into the container
COPY . .

# 7. Define the command to run your application
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api_server:app"]