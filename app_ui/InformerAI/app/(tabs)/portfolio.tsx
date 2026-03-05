import React from 'react';
import { StyleSheet, View, Text } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Colors } from '../../constants/Colors';

export default function PortfolioScreen() {
    return (
        <SafeAreaView style={styles.container}>
            <Text style={styles.title}>Portfolio</Text>
            <View style={styles.content}>
                <Text style={styles.text}>Portfolio tracking coming soon...</Text>
            </View>
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
    content: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    text: {
        color: Colors.text.secondary
    }
});
