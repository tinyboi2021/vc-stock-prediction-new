FROM node:20-alpine

# Install git and bash as they are often required by Expo and other tools
RUN apk add --no-cache git bash

# Set working directory
WORKDIR /app

# Install global expo-cli (optional, but useful for debugging)
# npm install -g expo-cli is deprecated, we use local expo or npx
RUN npm install -g expo-cli

# Expose ports for Metro Bundler and Expo
EXPOSE 8081 19000 19001 19002

# Default command to keep container running if no other command is specified
CMD ["tail", "-f", "/dev/null"]
