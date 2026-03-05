import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { Colors } from '../constants/Colors';
import { Ionicons } from '@expo/vector-icons';
import { useRouter } from 'expo-router';

interface StockCardProps {
    stock: {
        id: string;
        name: string;
        currentPrice: number;
        predictedPrice1D: number;
        predictedPrice16D: number;
        change1D: number;
        change16D: number;
        confidence: number;
        sentiment: string;
    };
}

export const StockCard: React.FC<StockCardProps> = ({ stock }) => {
    const router = useRouter();

    const handlePress = () => {
        router.push({ pathname: '/details/[id]', params: { id: stock.id } });
    };

    const getSentimentColor = (sentiment: string) => {
        if (sentiment.includes('Positive')) return Colors.success;
        if (sentiment.includes('Negative')) return Colors.danger;
        return Colors.neutral;
    };

    return (
        <TouchableOpacity style={styles.card} onPress={handlePress}>
            <View style={styles.header}>
                <View>
                    <Text style={styles.symbol}>{stock.id}</Text>
                    <Text style={styles.name}>{stock.name}</Text>
                </View>
                <View style={styles.priceContainer}>
                    <Text style={styles.price}>${stock.currentPrice.toFixed(2)}</Text>
                    <Text style={[styles.change, { color: stock.change1D >= 0 ? Colors.success : Colors.danger }]}>
                        {stock.change1D >= 0 ? '+' : ''}{stock.change1D}%
                    </Text>
                </View>
            </View>

            <View style={styles.divider} />

            <View style={styles.predictionRow}>
                <View>
                    <Text style={styles.label}>1D Pred</Text>
                    <Text style={[styles.predictionValue, { color: stock.predictedPrice1D >= stock.currentPrice ? Colors.success : Colors.danger }]}>
                        ${stock.predictedPrice1D.toFixed(2)}
                    </Text>
                </View>
                <View>
                    <Text style={styles.label}>16D Pred</Text>
                    <Text style={[styles.predictionValue, { color: stock.predictedPrice16D >= stock.currentPrice ? Colors.success : Colors.danger }]}>
                        ${stock.predictedPrice16D.toFixed(2)}
                    </Text>
                </View>
                <View>
                    <Text style={styles.label}>Confidence</Text>
                    <Text style={styles.confidence}>{(stock.confidence * 100).toFixed(0)}%</Text>
                </View>
            </View>

            <View style={styles.progressBarContainer}>
                <View style={[styles.progressBar, { width: `${stock.confidence * 100}%`, backgroundColor: getSentimentColor(stock.sentiment) }]} />
            </View>
        </TouchableOpacity>
    );
};

const styles = StyleSheet.create({
    card: {
        backgroundColor: Colors.card.light,
        borderRadius: 12,
        padding: 16,
        marginBottom: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 3,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
    },
    symbol: {
        fontSize: 18,
        fontWeight: 'bold',
        color: Colors.text.light,
    },
    name: {
        fontSize: 14,
        color: Colors.text.secondary,
    },
    priceContainer: {
        alignItems: 'flex-end',
    },
    price: {
        fontSize: 18,
        fontWeight: 'bold',
        color: Colors.text.light,
    },
    change: {
        fontSize: 14,
        fontWeight: '500',
    },
    divider: {
        height: 1,
        backgroundColor: Colors.border.light,
        marginVertical: 8,
    },
    predictionRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 8,
    },
    label: {
        fontSize: 12,
        color: Colors.text.secondary,
    },
    predictionValue: {
        fontSize: 14,
        fontWeight: '600',
    },
    confidence: {
        fontSize: 14,
        fontWeight: 'bold',
        color: Colors.primary,
    },
    progressBarContainer: {
        height: 4,
        backgroundColor: Colors.border.light,
        borderRadius: 2,
        marginTop: 4,
        overflow: 'hidden',
    },
    progressBar: {
        height: '100%',
    }
});
