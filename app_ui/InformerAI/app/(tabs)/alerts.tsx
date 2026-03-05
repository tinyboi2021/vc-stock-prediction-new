import React from 'react';
import { StyleSheet, View, Text, FlatList } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Colors } from '../../constants/Colors';
import { MOCK_ALERTS } from '../../constants/MockData';
import { Ionicons } from '@expo/vector-icons';

export default function AlertsScreen() {
    return (
        <SafeAreaView style={styles.container}>
            <Text style={styles.title}>Alerts</Text>
            <FlatList
                data={MOCK_ALERTS}
                keyExtractor={(item) => item.id.toString()}
                renderItem={({ item }) => (
                    <View style={styles.alertItem}>
                        <Ionicons
                            name={item.type === 'target' ? 'disc-outline' : 'warning-outline'}
                            size={24}
                            color={item.type === 'target' ? Colors.success : Colors.warning}
                        />
                        <View style={styles.alertContent}>
                            <Text style={styles.alertMessage}>{item.message}</Text>
                            <Text style={styles.alertTime}>{item.time}</Text>
                        </View>
                        {!item.read && <View style={styles.unreadDot} />}
                    </View>
                )}
            />
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: Colors.background.light,
    },
    title: {
        fontSize: 24,
        fontWeight: 'bold',
        margin: 16,
        color: Colors.text.light,
    },
    alertItem: {
        flexDirection: 'row',
        padding: 16,
        borderBottomWidth: 1,
        borderBottomColor: Colors.border.light,
        alignItems: 'center',
    },
    alertContent: {
        marginLeft: 12,
        flex: 1,
    },
    alertMessage: {
        fontSize: 16,
        color: Colors.text.light,
        fontWeight: '500',
    },
    alertTime: {
        fontSize: 12,
        color: Colors.text.secondary,
        marginTop: 4,
    },
    unreadDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: Colors.primary,
    }
});
