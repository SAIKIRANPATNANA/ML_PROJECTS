# Use an official Python runtime as a parent image
FROM python:3.8-slim

RUN apt-get update && \
    apt-get install -y git
# Set the working directory to /RuralloanApprovalBackgroundVerificationProject
WORKDIR /RuralloanApprovalBackgroundVerificationProject

# Copy the entire project into the container
COPY . /RuralloanApprovalBackgroundVerificationProject

# # Copy only the necessary files into the container
# COPY src/__init__.py src/__init__.py
# COPY src/RuralLoanApprovalBackgroundVerification/__init__.py src/RuralLoanApprovalBackgroundVerification/__init__.py
# COPY src/RuralLoanApprovalBackgroundVerification/components/__init__.py src/RuralLoanApprovalBackgroundVerification/components/__init__.py
# COPY src/RuralLoanApprovalBackgroundVerification/components/data_ingestion.py src/RuralLoanApprovalBackgroundVerification/components/data_ingestion.py
# COPY src/RuralLoanApprovalBackgroundVerification/components/data_validation.py src/RuralLoanApprovalBackgroundVerification/components/data_validation.py 
# COPY src/RuralLoanApprovalBackgroundVerification/components/data_transformation.py src/RuralLoanApprovalBackgroundVerification/components/data_transformation.py
# COPY src/RuralLoanApprovalBackgroundVerification/components/model_trainer.py src/RuralLoanApprovalBackgroundVerification/components/model_trainer.py
# COPY src/RuralLoanApprovalBackgroundVerification/components/model_evaluation.py src/RuralLoanApprovalBackgroundVerification/components/model_evaluation.py
# COPY src/RuralLoanApprovalBackgroundVerification/pipelines/__init__.py src/RuralLoanApprovalBackgroundVerification/pipelines/__init__.py
# COPY src/RuralLoanApprovalBackgroundVerification/pipelines/training_pipeline.py src/RuralLoanApprovalBackgroundVerification/pipelines/training_pipeline.py
# COPY src/RuralLoanApprovalBackgroundVerification/pipelines/prediction_pipeline.py src/RuralLoanApprovalBackgroundVerification/pipelines/prediction_pipeline.py
# COPY src/RuralLoanApprovalBackgroundVerification/logger.py src/RuralLoanApprovalBackgroundVerification/logger.py
# COPY src/RuralLoanApprovalBackgroundVerification/exception.py src/RuralLoanApprovalBackgroundVerification/exception.py
# COPY src/RuralLoanApprovalBackgroundVerification/utils/__init__.py src/RuralLoanApprovalBackgroundVerification/utils/__init__.py
# COPY src/RuralLoanApprovalBackgroundVerification/utils/utils.py src/RuralLoanApprovalBackgroundVerification/utils/utils.py
# COPY app.py app.py
# COPY requirements.txt requirements.txt
# COPY setup.py setup.py

# Copy the dataset to the /RuralloanApprovalBackgroundVerificationProject/dataset directory in the container
# COPY data/trainingData.csv dataset/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME PKS


# Run the training pipeline script with the provided dataset path when the container launches
CMD ["python", "src/RuralLoanApprovalBackgroundVerification/pipelines/training_pipeline.py"]
