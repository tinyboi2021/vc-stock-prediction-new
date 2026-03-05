#!/bin/bash
echo "InformerAI UI - Development Server Startup"
echo "--------------------------------------------"

echo "Detecting active IP address..."
if command -v ip > /dev/null; then
    export HOST_IP=$(ip route get 1 | sed -n 's/^.*src \([0-9.]*\) .*$/\1/p')
elif command -v ipconfig > /dev/null; then
    export HOST_IP=$(ipconfig getifaddr en0 || ipconfig getifaddr en1)
else
    export HOST_IP=$(hostname -I | awk '{print $1}')
fi

if [ -z "$HOST_IP" ]; then
    echo "Could not parse dynamic IP automatically. Please enter it:"
    read HOST_IP
fi

echo "+ Successfully grabbed Host IP: $HOST_IP"

echo -e "\n[1/3] Refreshing Docker Containers..."
docker-compose down

echo -e "\n[2/3] Building and starting the UI container..."
export REACT_NATIVE_PACKAGER_HOSTNAME=$HOST_IP
docker-compose up -d --build

echo -e "\n[3/3] Starting the Expo Metro Bundler on $HOST_IP ..."
sleep 3

docker exec -it informer-app sh -c "cd InformerAI && REACT_NATIVE_PACKAGER_HOSTNAME=$HOST_IP EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0 npx expo start --lan --clear"

echo -e "\nDone! Keep this window open while developing."
