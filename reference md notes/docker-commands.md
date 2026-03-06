# Essential Docker Commands Reference

## Container Management

### Running Containers
```bash
# Run a container from an image
docker run <image_name>

# Run container in detached mode (background)
docker run -d <image_name>

# Run container with interactive terminal
docker run -it <image_name>

# Run container with port mapping
docker run -p <host_port>:<container_port> <image_name>

# Run container with volume mounting
docker run -v <host_path>:<container_path> <image_name>

# Run container with environment variables
docker run -e ENV_VAR=value <image_name>

# Run container with custom name
docker run --name <container_name> <image_name>

# Run container with automatic removal after exit
docker run --rm <image_name>
```

### Container Lifecycle
```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Start a stopped container
docker start <container_id_or_name>

# Stop a running container
docker stop <container_id_or_name>

# Restart a container
docker restart <container_id_or_name>

# Pause a container
docker pause <container_id_or_name>

# Unpause a container
docker unpause <container_id_or_name>

# Remove a container
docker rm <container_id_or_name>

# Remove all stopped containers
docker container prune
```

### Container Interaction
```bash
# Execute command in running container
docker exec <container_id_or_name> <command>

# Get interactive shell in running container
docker exec -it <container_id_or_name> /bin/bash

# View container logs
docker logs <container_id_or_name>

# Follow container logs in real-time
docker logs -f <container_id_or_name>

# Copy files between host and container
docker cp <src_path> <container_id>:<dest_path>
docker cp <container_id>:<src_path> <dest_path>

# View container resource usage
docker stats <container_id_or_name>

# Inspect container details
docker inspect <container_id_or_name>
```

## Image Management

### Image Operations
```bash
# List local images
docker images

# Pull an image from registry
docker pull <image_name>:<tag>

# Build image from Dockerfile
docker build -t <image_name>:<tag> <path_to_dockerfile>

# Build with build arguments
docker build --build-arg ARG_NAME=value -t <image_name> .

# Tag an image
docker tag <source_image> <target_image>:<tag>

# Remove an image
docker rmi <image_id_or_name>

# Remove unused images
docker image prune

# Remove all unused images (including tagged)
docker image prune -a

# View image history/layers
docker history <image_name>
```

### Registry Operations
```bash
# Login to Docker registry
docker login <registry_url>

# Push image to registry
docker push <image_name>:<tag>

# Search for images in Docker Hub
docker search <search_term>
```

## Volume Management

### Volume Commands
```bash
# Create a volume
docker volume create <volume_name>

# List volumes
docker volume ls

# Inspect volume details
docker volume inspect <volume_name>

# Remove a volume
docker volume rm <volume_name>

# Remove unused volumes
docker volume prune

# Mount volume to container
docker run -v <volume_name>:<container_path> <image_name>

# Bind mount host directory
docker run -v <host_path>:<container_path> <image_name>
```

## Network Management

### Network Commands
```bash
# List networks
docker network ls

# Create a network
docker network create <network_name>

# Create network with specific driver
docker network create --driver bridge <network_name>

# Connect container to network
docker network connect <network_name> <container_name>

# Disconnect container from network
docker network disconnect <network_name> <container_name>

# Remove network
docker network rm <network_name>

# Remove unused networks
docker network prune

# Inspect network details
docker network inspect <network_name>

# Run container on specific network
docker run --network <network_name> <image_name>
```

## Docker Compose

### Compose Commands
```bash
# Start services defined in docker-compose.yml
docker-compose up

# Start services in detached mode
docker-compose up -d

# Start specific service
docker-compose up <service_name>

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View running services
docker-compose ps

# View service logs
docker-compose logs <service_name>

# Follow service logs
docker-compose logs -f <service_name>

# Execute command in service container
docker-compose exec <service_name> <command>

# Build services
docker-compose build

# Pull service images
docker-compose pull

# Restart services
docker-compose restart <service_name>

# Scale services
docker-compose up --scale <service_name>=<number>
```

## System Management

### System Information
```bash
# Show Docker system information
docker info

# Show Docker version
docker version

# Show disk usage
docker system df

# Remove unused data (containers, networks, images, cache)
docker system prune

# Remove all unused data including volumes
docker system prune -a --volumes

# View Docker events in real-time
docker events
```

### Resource Management
```bash
# Set memory limit for container
docker run -m 512m <image_name>

# Set CPU limit for container
docker run --cpus="1.5" <image_name>

# Set CPU shares (relative weight)
docker run --cpu-shares=512 <image_name>

# Run container with resource constraints
docker run --memory=1g --cpus="2" <image_name>
```

## Dockerfile Commands

### Common Dockerfile Instructions
```dockerfile
# Base image
FROM <image>:<tag>

# Set working directory
WORKDIR <path>

# Copy files from host to container
COPY <src> <dest>

# Add files (supports URLs and tar extraction)
ADD <src> <dest>

# Run commands during build
RUN <command>

# Set environment variables
ENV <key>=<value>

# Expose ports
EXPOSE <port>

# Set default command
CMD ["executable", "param1", "param2"]

# Set entrypoint
ENTRYPOINT ["executable", "param1"]

# Create volume mount point
VOLUME ["<path>"]

# Set user for subsequent commands
USER <username>

# Add build argument
ARG <name>=<default_value>

# Add metadata labels
LABEL <key>=<value>

# Add health check
HEALTHCHECK --interval=30s --timeout=3s CMD <command>
```

## Debug and Troubleshooting

### Debugging Commands
```bash
# View container processes
docker top <container_name>

# Get low-level container information
docker inspect <container_name>

# View container filesystem changes
docker diff <container_name>

# Export container as tar archive
docker export <container_name> > container.tar

# Save image as tar archive
docker save <image_name> > image.tar

# Load image from tar archive
docker load < image.tar

# Import container from tar archive
docker import container.tar <new_image_name>

# View Docker daemon logs (varies by system)
journalctl -u docker.service

# Run container with debugging
docker run --rm -it --entrypoint /bin/sh <image_name>
```

### Performance and Monitoring
```bash
# Monitor container resource usage
docker stats

# View port mappings
docker port <container_name>

# View container IP address
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_name>

# Check if container is running
docker inspect -f '{{.State.Running}}' <container_name>
```

## Quick Reference Examples

### Complete Workflow Example
```bash
# 1. Pull base image
docker pull ubuntu:20.04

# 2. Run interactive container
docker run -it --name mycontainer ubuntu:20.04

# 3. Make changes in container, then exit

# 4. Commit changes to new image
docker commit mycontainer myapp:v1

# 5. Run new image
docker run -d -p 8080:80 --name myapp myapp:v1

# 6. Check if running
docker ps

# 7. View logs
docker logs myapp

# 8. Clean up
docker stop myapp
docker rm myapp
docker rmi myapp:v1
```

### Multi-container Application
```bash
# Create network
docker network create mynetwork

# Run database container
docker run -d --name mydb --network mynetwork \
  -e MYSQL_ROOT_PASSWORD=secret mysql:8.0

# Run application container
docker run -d --name myapp --network mynetwork \
  -p 3000:3000 -e DB_HOST=mydb myapp:latest

# View all containers
docker ps

# Clean up
docker stop myapp mydb
docker rm myapp mydb
docker network rm mynetwork
```

## Tips and Best Practices

### Performance Tips
- Use `.dockerignore` to exclude unnecessary files from build context
- Use multi-stage builds to reduce image size
- Leverage Docker layer caching by organizing Dockerfile instructions properly
- Use specific image tags instead of `latest` for reproducibility
- Clean up unused resources regularly with `docker system prune`

### Security Best Practices
- Don't run containers as root user when possible
- Use official images as base images
- Keep images updated to latest security patches
- Limit container resources (memory, CPU)
- Use secrets management for sensitive data instead of environment variables
- Scan images for vulnerabilities regularly

### Development Workflow
- Use volumes for persistent data
- Use bind mounts for development code
- Implement health checks for services
- Use docker-compose for multi-container applications
- Tag images with meaningful versions
- Use environment-specific configurations