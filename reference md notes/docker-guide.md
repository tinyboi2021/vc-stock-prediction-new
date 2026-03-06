# 🐳 Docker Guide for Fake News Detection Project

## Table of Contents
1. [What is Docker?](#what-is-docker)
2. [Why Use Docker?](#why-use-docker)
3. [Docker Installation](#docker-installation)
4. [Docker Concepts](#docker-concepts)
5. [Project Setup](#project-setup)
6. [Quick Start](#quick-start)
7. [Development Workflow](#development-workflow)
8. [Team Collaboration](#team-collaboration)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## What is Docker?

Docker is a platform that uses **containerization** to package your application and all its dependencies into a portable, lightweight container. Think of it as a shipping container for your software - it contains everything needed to run your application consistently across different environments.

### Key Benefits:
- **Consistency**: "It works on my machine" → "It works everywhere"
- **Portability**: Run the same container on any system with Docker
- **Isolation**: Applications don't interfere with each other
- **Scalability**: Easy to scale up or down
- **Team Collaboration**: Share identical development environments

## Why Use Docker?

### Traditional Problems:
```
Developer A: "The code works fine on my machine!"
Developer B: "But it crashes on mine..."
Operations: "It doesn't work in production either..."
```

### Docker Solution:
```
✅ Same environment everywhere
✅ No more dependency conflicts  
✅ Easy deployment across different systems
✅ Team members work with identical setups
```

## Docker Installation

### Windows:
1. Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
2. Run the installer
3. Restart your computer
4. Open Docker Desktop and complete setup

### macOS:
1. Download Docker Desktop for Mac
2. Drag Docker to Applications folder
3. Launch Docker Desktop
4. Complete the setup wizard

### Linux (Ubuntu/Debian):
```bash
# Update package index
sudo apt update

# Install required packages
sudo apt install apt-transport-https ca-certificates curl software-properties-common

# Add Docker's GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add your user to docker group (avoid using sudo)
sudo usermod -aG docker $USER
```

### Verify Installation:
```bash
docker --version
docker compose version
```

## Docker Concepts

### 1. Images
- **What**: Blueprint/template for containers
- **Analogy**: Like a recipe for baking a cake
- **Example**: `python:3.9-slim` image contains Python 3.9 and basic tools

### 2. Containers
- **What**: Running instance of an image
- **Analogy**: The actual cake baked from the recipe
- **Example**: Your fake news detection app running in a container

### 3. Dockerfile
- **What**: Text file with instructions to build an image
- **Analogy**: Step-by-step recipe instructions
- **Example**: 
```dockerfile
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### 4. Docker Compose
- **What**: Tool to define and run multi-container applications
- **Analogy**: Orchestra conductor managing multiple musicians
- **Example**: Running your app + database + cache together

### 5. Volumes
- **What**: Persistent storage for containers
- **Analogy**: External hard drive for your computer
- **Example**: Storing ML models and data outside containers

## Project Setup

### Project Structure:
```
fake-news-detection/
├── Dockerfile                 # Instructions to build app image
├── docker-compose.yml        # Multi-container orchestration
├── docker-compose.dev.yml    # Development configuration
├── docker-compose.prod.yml   # Production configuration
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
├── .env                     # Your environment settings
├── .dockerignore           # Files to exclude from build
├── setup.sh                # Automated setup script
├── app.py                  # Main Flask application
├── logs/                   # Application logs
├── data/                   # Datasets and models
└── README.md              # Project documentation
```

## Quick Start

### Option 1: Using Setup Script (Recommended)
```bash
# Clone the project
git clone <your-project-repo>
cd fake-news-detection

# Make setup script executable
chmod +x setup.sh

# Start development environment
./setup.sh start-dev

# Your app is now running at http://localhost:8000
```

### Option 2: Manual Docker Commands
```bash
# Build the Docker image
docker build -t fake-news-detector .

# Run the container
docker run -d -p 8000:8000 --name fake-news-app fake-news-detector

# Check if it's running
docker ps

# View logs
docker logs fake-news-app
```

### Option 3: Using Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

## Development Workflow

### Daily Development:
```bash
# Start your development environment
./setup.sh start-dev

# Make code changes...
# The app will automatically reload in development mode

# View logs to debug
./setup.sh logs

# Stop when done
./setup.sh stop
```

### Testing Changes:
```bash
# Test API endpoints
./setup.sh test

# Or manually test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking: Scientists discover amazing new technology!"}'
```

### Code Updates:
```bash
# Rebuild and restart after major changes
./setup.sh restart-dev

# View container status
./setup.sh status
```

## Team Collaboration

### Sharing Your Project:

1. **Share the Repository**:
```bash
# Team member clones the repo
git clone <your-repo-url>
cd fake-news-detection

# Quick setup
./setup.sh start-dev
```

2. **Using Docker Hub (Public/Private Registry)**:
```bash
# Build and tag your image
docker build -t yourusername/fake-news-detector:v1.0 .

# Push to Docker Hub
docker push yourusername/fake-news-detector:v1.0

# Team members can pull and run
docker pull yourusername/fake-news-detector:v1.0
docker run -d -p 8000:8000 yourusername/fake-news-detector:v1.0
```

3. **Export/Import Images**:
```bash
# Export image to file
docker save fake-news-detector:latest | gzip > fake-news-detector.tar.gz

# Share the .tar.gz file with team members

# Team members import the image
gunzip -c fake-news-detector.tar.gz | docker load
```

### Environment Consistency:
- All team members use the same Docker setup
- No more "it works on my machine" problems
- Identical Python versions, libraries, and configurations
- Easy onboarding for new team members

## Troubleshooting

### Common Issues:

#### 1. Port Already in Use
```bash
# Error: Port 8000 already in use
# Solution: Stop existing containers
docker stop $(docker ps -q --filter "publish=8000")

# Or use different port
APP_PORT=8001 ./setup.sh start-dev
```

#### 2. Permission Denied
```bash
# Error: Permission denied
# Solution: Add user to docker group
sudo usermod -aG docker $USER
# Logout and login again
```

#### 3. Out of Disk Space
```bash
# Clean up unused Docker resources
docker system prune -a

# Remove specific containers/images
docker rm container_name
docker rmi image_name
```

#### 4. Container Won't Start
```bash
# Check logs for errors
docker logs container_name

# Debug by running interactively
docker run -it fake-news-detector /bin/bash
```

#### 5. Model Download Issues
```bash
# If models fail to download, check internet connection
# Set environment variables for proxy if needed
docker run -e HTTP_PROXY=http://proxy:8080 fake-news-detector
```

### Debugging Commands:
```bash
# List all containers
docker ps -a

# List all images
docker images

# Execute commands inside running container
docker exec -it container_name /bin/bash

# View resource usage
docker stats

# Inspect container configuration
docker inspect container_name
```

## Best Practices

### 1. Security
```bash
# Don't run containers as root (already handled in our Dockerfile)
# Use specific image tags, not 'latest'
FROM python:3.9-slim  # ✅ Good
FROM python:latest    # ❌ Avoid

# Don't store secrets in images
# Use environment variables or Docker secrets
```

### 2. Performance
```bash
# Use .dockerignore to exclude unnecessary files
# Use multi-stage builds for smaller images
# Cache dependencies properly by copying requirements.txt first
```

### 3. Development
```bash
# Use volumes for code during development
# Use different configurations for dev/prod
# Health checks for production deployments
```

### 4. Team Collaboration
```bash
# Always use docker-compose for multi-service apps
# Document environment variables in .env.example
# Use semantic versioning for images
# Regular cleanup of unused resources
```

## Advanced Usage

### Production Deployment:
```bash
# Use production configuration
./setup.sh start-prod

# Or with custom settings
WORKERS=4 MEMORY_LIMIT=4G ./setup.sh start-prod
```

### Scaling:
```bash
# Scale specific services
docker-compose up -d --scale fake-news-detector=3
```

### Monitoring:
```bash
# View real-time resource usage
docker stats

# Monitor logs from all services
docker-compose logs -f
```

### Backup:
```bash
# Backup volumes
docker run --rm -v fake_news_model_cache:/data -v $(pwd):/backup ubuntu tar czf /backup/models-backup.tar.gz /data
```

## Getting Help

### Resources:
- [Official Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Hub](https://hub.docker.com/)

### Support Commands:
```bash
# Get help from setup script
./setup.sh help

# Docker command help
docker --help
docker-compose --help
```

### Community:
- Docker Community Forums
- Stack Overflow (tag: docker)
- GitHub Issues for this project

---

## Summary

Docker simplifies development and deployment by:
1. **Packaging** your app with all dependencies
2. **Ensuring** consistency across environments
3. **Enabling** easy team collaboration
4. **Streamlining** deployment processes

With this setup, you and your team can:
- Start working immediately with `./setup.sh start-dev`
- Share the project with confidence it will work everywhere
- Deploy to any system that supports Docker
- Scale and maintain your application easily

Happy containerizing! 🐳