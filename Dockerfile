FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y libgomp1

# Copy Dataset
COPY data/Dataset.csv ./data/Dataset.csv

# Copy everything needed
COPY src/ ./src/
COPY run_all.sh ./run_all.sh

# Make your script executable
RUN chmod +x ./run_all.sh

# Default command: run your shell script
CMD ["./run_all.sh"]
