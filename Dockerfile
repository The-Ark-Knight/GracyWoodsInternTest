# Specify the base image
FROM python:3.12

# Create a working directory
WORKDIR /app

# Copy the application's code
COPY . /app

# Install dependencies
RUN pip install -r /documentation/requirements.txt

# Expose the port for your Flask application
EXPOSE 5000

# Define the command to run your Flask app
CMD python /deployment/app.py