Write-Host "InformerAI UI - Development Server Startup"
Write-Host "--------------------------------------------"

Write-Host "Detecting active Wi-Fi / Ethernet IP address..."
$ip = (Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -ne $null -and $_.NetAdapter.Status -eq "Up" } | Select-Object -ExpandProperty IPv4Address)[0].IPAddress

if (-not $ip) {
    Write-Warning "Could not automatically detect your IP address."
    $ip = Read-Host -Prompt "Please enter your local IPv4 address manually (e.g. 192.168.1.x)"
}

Write-Host "+ Successfully grabbed Host IP: $ip"

Write-Host "`n[1/3] Refreshing Docker Containers..."
docker-compose down

Write-Host "`n[2/3] Building and starting the UI container..."
$env:REACT_NATIVE_PACKAGER_HOSTNAME = $ip
docker-compose up -d --build

Write-Host "`n[3/3] Starting the Expo Metro Bundler on $ip ..."
Start-Sleep -Seconds 3

docker exec -it informer-app sh -c "cd InformerAI && REACT_NATIVE_PACKAGER_HOSTNAME=$ip EXPO_DEVTOOLS_LISTEN_ADDRESS=0.0.0.0 npx expo start --lan --clear"

Write-Host "`nDone! Keep this window open while developing."
