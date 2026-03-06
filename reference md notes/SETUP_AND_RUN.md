# Informer Stock Prediction - Setup and Workflow Guide

This guide provides comprehensive instructions on how to set up, run, and troubleshoot the containerized workflows for the Informer Stock Prediction project. The project is split into two main components: the backend infrastructure (API, LLM Server, Redis) and the frontend mobile application (React Native / Expo).

---

## 📋 Prerequisites
Before you begin, ensure you have the following installed on your host machine:
- **Docker Desktop** (or Docker Engine)
- **Docker Compose**
- **NVIDIA Container Toolkit** (Optional but highly recommended for Ollama GPU acceleration)

---

## 🚀 1. Backend Infrastructure Workflow
The backend consists of three main services managed by the `docker-compose.yml` in the `app_backend` directory:
- **Ollama (`stock-ollama`)**: The LLM server running the `llama3` model for sentiment analysis.
- **FastAPI Backend (`stock-inference-api`)**: The REST API integrating model inference and the LLM.
- **Redis (`stock-redis-cache`)**: Caches news and API responses to minimize redundant requests.

### Running the Backend
1. Open a terminal and navigate to the `app_backend` directory:
   ```bash
   cd "c:\xampp\htdocs\stock Prediction latest model\app_backend"
   ```
2. Build and start the backend containers in detached mode:
   ```bash
   docker-compose up -d --build
   ```
3. To view the logs across all backend services:
   ```bash
   docker-compose logs -f
   ```

*Note: On the first run, the Ollama container will automatically download the `llama3` model. This may take a few minutes depending on your internet connection.*

### Checking Service Status
To verify the services are running and accessible:
- **Ollama**: Run `curl http://localhost:11434/` (Expected output: `Ollama is running`) or `docker exec -it stock-ollama ollama ps` to see loaded models.
- **Redis**: Run `docker exec -it stock-redis-cache redis-cli ping` (Expected output: `PONG`).
- **FastAPI**: Visit `http://localhost:8000/docs` in your browser.

### Accessing Backend Services
- **FastAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Ollama API endpoints**: `http://localhost:11434`
- **Redis Cache**: Accessible internally on port `6379`.

---

## 📱 2. Mobile App UI Workflow
The mobile application is a React Native app built with Expo, containerized for a seamless development experience.

### Running the Mobile UI
1. Open a new terminal and navigate to the `app_ui` directory:
   ```bash
   cd "c:\xampp\htdocs\stock Prediction latest model\app_ui"
   ```
2. Build and start the UI container in detached mode:
   ```bash
   docker-compose up -d --build
   ```
3. Attach to the container's shell to run Expo commands:
   ```bash
   docker exec -it informer-app /bin/bash
   ```
4. Inside the container, start the Expo development server:
   ```bash
   cd InformerAI
   npm install   # (If dependencies aren't installed yet)
   npx expo start
   ```
5. Choose your run method when prompted by Expo (e.g., `a` for Android emulator, or scan the QR code with your physical device via the Expo Go app).

### Checking Expo Status
To verify Expo is running correctly:
- Open your browser and navigate to `http://localhost:8081` (The Expo Metro bundler dashboard/status should respond).
- Or check the live logs from the host terminal:
  ```bash
  docker logs -f informer-app
  ```

#### Running Expo directly from the Host
If you want to start Expo directly from your host machine bypassing the interactive shell, you can use:
```bash
docker exec -it informer-app sh -c "cd InformerAI && REACT_NATIVE_PACKAGER_HOSTNAME=192.168.1.6 EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0 npx expo start --lan --clear"
```

---

## 🛠️ Troubleshooting & Common Issues

### Backend Troubleshooting

#### 1. Ollama GPU is Not Being Utilized (Running on CPU only)
* **Symptom**: Inference is extremely slow.
* **Fix**: Ensure the NVIDIA Container Toolkit is installed on your host system. Check if Docker recognises your GPU by running:
  ```bash
  docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
  ```
  If this fails, reinstall the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and restart Docker.

#### 2. Port Conflicts (e.g., Port 8000 or 11434 is already allocated)
* **Symptom**: `Bind for 0.0.0.0:8000 failed: port is already allocated`
* **Fix**: Another application is using the port. Find and kill the process using it, or modify the ports mapping in the `docker-compose.yml` file (e.g., change `"8000:8000"` to `"8080:8000"`).

#### 3. FastAPI Backend Exits with Native Model Errors
* **Symptom**: The API container crashes, citing missing pre-trained models or scalers.
* **Fix**: Ensure that the directories `Saved_Models`, `Saved_Scalers`, and the file `Paper_Replication_Results.csv` are present in the project root. These are mounted into the API container at runtime.

#### 4. Redis Cache Management & Troubleshooting
* **Symptom**: Unexpected or outdated news results, high memory usage, or need to manually verify cached data.
* **Fix/Commands**: You can manage and troubleshoot the Redis cache using the following commands:
  - **Flush all cache (Clear completely)**:
    ```bash
    docker exec -it stock-redis-cache redis-cli FLUSHALL
    ```
  - **Check Redis memory usage**:
    ```bash
    docker exec -it stock-redis-cache redis-cli INFO memory
    ```
  - **List all cached keys**:
    *(Use carefully if the cache is very large)*
    ```bash
    docker exec -it stock-redis-cache redis-cli KEYS "*"
    ```
  - **Delete a specific cached key**:
    *(Replace `<key_name>` with the actual key from the `KEYS` command, e.g., `news:AAPL`)*
    ```bash
    docker exec -it stock-redis-cache redis-cli DEL "<key_name>"
    ```
  - **Monitor live query activity** (Useful for seeing what keys the API is requesting):
    ```bash
    docker exec -it stock-redis-cache redis-cli MONITOR
    ```

### Mobile UI Troubleshooting

#### 1. Expo Cannot Connect to the App on Physical Device
* **Symptom**: Scanning the QR code results in a timeout or network error.
* **Fix**: 
  - Ensure your phone and your PC are on the **exact same Wi-Fi network**.
  - Check Windows Firewall rules. You may need to temporarily disable your firewall locally or allow ports `8081` and `19000-19002` through your firewall.
  - If running WSL/Docker on Windows, Expo sometimes broadcasts an internal Docker IP instead of your machine's local IP. You can force Expo to use your LAN IP by running:
    ```bash
    REACT_NATIVE_PACKAGER_HOSTNAME=<YOUR_IPV4_ADDRESS> npx expo start
    ```

#### 2. Node Modules / Dependencies Syncing Issues
* **Symptom**: `npm install` inside the container causes issues or missing modules on the host.
* **Fix**: Rebuild the UI container ignoring the current volumes, or thoroughly clean the cache:
  ```bash
  docker exec -it informer-app /bin/bash
  cd InformerAI
  rm -rf node_modules package-lock.json
  npm cache clean --force
  npm install
  ```

#### 3. Restarting Everything Fresh (Wiping all Cache and Containers)
If you encounter weird caching behaviour across docker networks, tear everything down completely:
```bash
# In the app_backend directory:
cd "c:\xampp\htdocs\stock Prediction latest model\app_backend"
docker-compose down -v --remove-orphans

# In the app_ui folder:
cd "c:\xampp\htdocs\stock Prediction latest model\app_ui"
docker-compose down -v --remove-orphans
```
*Warning: The `-v` flag removes named volumes, which means the Ollama model (`llama3`) will need to be downloaded again on the next boot.*
