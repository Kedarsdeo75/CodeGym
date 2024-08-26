# Use a base image with Python installed
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire Code-Gym-main project into the container
COPY . .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that your Flask app runs on
EXPOSE 5000

# Run the app.py file when the container starts
CMD ["python", "Part 3 UI Code/app.py"]
