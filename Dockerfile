FROM python:3.8
WORKDIR /app
# Copy the requirements file to the working directory
COPY requirements.txt .
# Install the required packages
RUN pip install -r requirements.txt
# Copy the rest of the project files to the working directory
COPY . .
# Expose port 8888 for Jupyter notebooks
EXPOSE 8888
# Launch Jupyter when the container starts
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
