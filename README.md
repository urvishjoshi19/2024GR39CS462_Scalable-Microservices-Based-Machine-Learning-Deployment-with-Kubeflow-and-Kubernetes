# Scalable Microservices-Based Machine Learning Deployment with Kubeflow and Kubernetes

## Overview

Deploying machine learning (ML) models as monolithic applications poses significant challenges in terms of scalability, maintainability, and operational efficiency. This project presents a microservices-based ML deployment architecture using Kubeflow on a Kubernetes cluster to enable flexible, resilient, and scalable deployments.

Each component of the ML pipeline—such as data ingestion, preprocessing, inference, and postprocessing—is decomposed into independent, containerized microservices. These are orchestrated using Kubeflow Pipelines and managed by Kubernetes for optimized resource allocation and lifecycle management.

## Key Features

- **Microservices Architecture**: Each stage of the ML pipeline is deployed as a standalone service.
- **Containerization**: Services are packaged as Docker containers for portability and consistency.
- **Kubernetes Orchestration**: Manages deployment, scaling, and high availability of services.
- **Kubeflow Pipelines**: Automates the orchestration and dependency management of pipeline steps.
- **Scalability**: Each microservice can scale independently based on workload requirements.
- **Resilience**: Failures in one component do not propagate through the system.
- **Continuous Integration and Deployment**: Supports CI/CD practices for rapid iteration and deployment.

## Technology Stack

- Python for machine learning logic
- Docker for containerization
- Kubernetes for orchestration and scaling
- Kubeflow for ML pipeline management


## Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/IIITV-5G-and-Edge-Computing-Activity/Scalable-ML-Kubeflow-K8s.git
cd Scalable-ML-Kubeflow-K8s

```

## Build Docker Images

- docker build -t ingestion-service ./ingestion-service
- docker build -t preprocessing-service ./preprocessing-service
- docker build -t inference-service ./inference-service
- docker build -t postprocessing-service ./postprocessing-service

## Deployment on Kubernetes

To deploy the services on your Kubernetes cluster, run the following command:

```bash
kubectl apply -f k8s-manifests/

```



