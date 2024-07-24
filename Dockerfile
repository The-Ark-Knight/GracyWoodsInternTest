FROM python:3.10-alpine

# Install necessary packages
RUN apk add --no-cache build-base gcc

# Set the working directory
WORKDIR /app

# Copy the application files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask pandas numpy scikit-learn

# Expose the port
EXPOSE 5000

# Start the application
CMD ["python", "app.py"]