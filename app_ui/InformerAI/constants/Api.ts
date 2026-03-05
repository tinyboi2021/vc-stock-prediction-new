import Constants from 'expo-constants';
import { Platform } from 'react-native';

export const getBackendUrl = () => {
    if (__DEV__) {
        // Expo sets the hostUri to the IP Address of the dev machine (e.g., 192.168.1.3:8081)
        const debuggerHost = Constants.expoConfig?.hostUri;
        if (debuggerHost) {
            const ipAddress = debuggerHost.split(':')[0];
            return `http://${ipAddress}:8000`;
        }

        // Fallbacks for emulators if hostUri is somehow missing
        if (Platform.OS === 'android') {
            return 'http://10.0.2.2:8000'; // Default Android emulator IP mapped to localhost
        }
        return 'http://localhost:8000'; // Default iOS simulator
    }

    // In production, you would replace this with your actual hosted backend URL
    return 'http://your-production-url.com';
};
