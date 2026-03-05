import React, { useState, useEffect } from 'react';
import { StyleSheet, View, Text, ScrollView, Dimensions, TouchableOpacity, ActivityIndicator } from 'react-native';
import { useLocalSearchParams, useRouter } from 'expo-router';
import { getBackendUrl } from '../../constants/Api';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Colors } from '../../constants/Colors';
import { MOCK_STOCKS } from '../../constants/MockData';
import { Ionicons } from '@expo/vector-icons';
import { LineChart } from 'react-native-chart-kit';

const screenWidth = Dimensions.get('window').width;

export default function StockDetectScreen() {
    const { id } = useLocalSearchParams();
    const router = useRouter();
    const initialStock = MOCK_STOCKS.find((s) => s.id === id);

    const [stock, setStock] = useState(initialStock);
    const [news, setNews] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [summary, setSummary] = useState('');

    useEffect(() => {
        if (id === 'AAPL') {
            const fetchDetails = async () => {
                setLoading(true);
                try {
                    console.log(`Fetching details for ${id}...`);

                    // Fetch prediction
                    const backendUrl = getBackendUrl();
                    const predRes = await fetch(`${backendUrl}/predict`);
                    if (predRes.ok) {
                        const data = await predRes.json();
                        console.log("Prediction Data for Details:", JSON.stringify(data, null, 2));

                        const predicted1D = data.forecast && data.forecast.length > 0 ? data.forecast[0].price : data.current_price;
                        const predicted16D = data.forecast && data.forecast.length > 15 ? data.forecast[15].price : data.current_price;
                        const change1D = ((predicted1D - data.current_price) / data.current_price) * 100;
                        const change16D = ((predicted16D - data.current_price) / data.current_price) * 100;

                        let sentimentText = 'Neutral';
                        if (data.sentiment_score > 0.2) sentimentText = 'Positive';
                        if (data.sentiment_score > 0.5) sentimentText = 'Very Positive';
                        if (data.sentiment_score < -0.2) sentimentText = 'Negative';
                        if (data.sentiment_score < -0.5) sentimentText = 'Very Negative';

                        setStock(prev => prev ? {
                            ...prev,
                            currentPrice: data.current_price,
                            predictedPrice1D: predicted1D,
                            predictedPrice16D: predicted16D,
                            change1D: Number(change1D.toFixed(2)),
                            change16D: Number(change16D.toFixed(2)),
                            confidence: data.sentiment_confidence,
                            sentiment: sentimentText,
                        } : prev);

                        setSummary(data.summary);
                    }

                    // Fetch news
                    const newsRes = await fetch(`${backendUrl}/news`);
                    if (newsRes.ok) {
                        const newsData = await newsRes.json();
                        console.log("News Data Received:", JSON.stringify(newsData, null, 2));
                        setNews(newsData);
                    }

                } catch (error) {
                    console.error("Error fetching details data:", error);
                } finally {
                    setLoading(false);
                }
            };
            fetchDetails();
        }
    }, [id]);

    if (!stock) {
        return (
            <SafeAreaView style={styles.container}>
                <Text>Stock not found</Text>
            </SafeAreaView>
        );
    }

    const chartData = {
        labels: ['Today', '7D', '14D', '16D'],
        datasets: [
            {
                data: [
                    stock.currentPrice,
                    stock.currentPrice * (1 + stock.change1D / 100 * 2),
                    stock.predictedPrice16D * 0.98,
                    stock.predictedPrice16D
                ],
                color: (opacity = 1) => `rgba(30, 136, 229, ${opacity})`,
                strokeWidth: 2
            },
            {
                data: [
                    stock.currentPrice,
                    stock.currentPrice * (1 - stock.change1D / 100 * 2),
                    stock.predictedPrice16D * 0.95,
                    stock.predictedPrice16D * 0.98
                ],
                color: (opacity = 1) => `rgba(200, 200, 200, ${opacity})`,
                strokeWidth: 1,
                withDots: false,
            }
        ],
        legend: ["Prediction"]
    };

    return (
        <SafeAreaView style={styles.container}>
            <View style={styles.header}>
                <TouchableOpacity onPress={() => router.back()}>
                    <Ionicons name="arrow-back" size={24} color={Colors.text.light} />
                </TouchableOpacity>
                <Text style={styles.headerTitle}>{stock.id} - {stock.name}</Text>
                <Ionicons name="ellipsis-vertical" size={24} color={Colors.text.light} />
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent}>
                <View style={styles.priceSection}>
                    <Text style={styles.currentPrice}>${stock.currentPrice.toFixed(2)}</Text>
                    <Text style={styles.predictionText}>
                        1D: ${stock.predictedPrice1D.toFixed(2)}
                        <Text style={{ color: stock.change1D >= 0 ? Colors.success : Colors.danger }}>
                            {stock.change1D >= 0 ? '↗' : '↘'} {stock.change1D}%
                        </Text>
                    </Text>
                    <Text style={styles.predictionText}>
                        16D: ${stock.predictedPrice16D.toFixed(2)}
                        <Text style={{ color: stock.change16D >= 0 ? Colors.success : Colors.danger }}>
                            {stock.change16D >= 0 ? '↗' : '↘'} {stock.change16D}%
                        </Text>
                    </Text>
                </View>

                <View style={styles.chartContainer}>
                    <Text style={styles.chartTitle}>Prediction Zone</Text>
                    <LineChart
                        data={chartData}
                        width={screenWidth - 32}
                        height={220}
                        chartConfig={{
                            backgroundColor: Colors.card.light,
                            backgroundGradientFrom: Colors.card.light,
                            backgroundGradientTo: Colors.card.light,
                            decimalPlaces: 1,
                            color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                            labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                        }}
                        bezier
                        style={{ borderRadius: 16 }}
                    />
                </View>

                <View style={styles.detailsCard}>
                    <Text style={styles.cardTitle}>Details</Text>
                    <View style={styles.detailRow}>
                        <Text>Confidence</Text>
                        <Text>{(stock.confidence * 100).toFixed(0)}%</Text>
                    </View>
                    <View style={styles.detailRow}>
                        <Text>Sentiment</Text>
                        <Text style={{ color: stock.sentiment.includes('Posit') ? Colors.success : (stock.sentiment.includes('Negat') ? Colors.danger : Colors.neutral) }}>{stock.sentiment}</Text>
                    </View>
                </View>

                {id === 'AAPL' && (
                    <View style={styles.newsSection}>
                        <Text style={styles.sectionTitle}>Latest News & AI Summary</Text>
                        {loading ? (
                            <ActivityIndicator size="small" color={Colors.primary} />
                        ) : (
                            <>
                                {summary ? (
                                    <View style={styles.summaryCard}>
                                        <Text style={styles.summaryTitle}>AI Summary</Text>
                                        <Text style={styles.summaryText}>{summary}</Text>
                                    </View>
                                ) : null}

                                {news && news.articles && news.articles.length > 0 ? (
                                    news.articles.slice(0, 5).map((article: string, idx: number) => (
                                        <View key={idx} style={styles.newsCard}>
                                            <Text style={styles.newsText}>{article}</Text>
                                        </View>
                                    ))
                                ) : (
                                    !loading && <Text style={styles.noNewsText}>No recent news available.</Text>
                                )}
                            </>
                        )}
                    </View>
                )}
            </ScrollView>
        </SafeAreaView>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: Colors.background.light },
    header: { flexDirection: 'row', justifyContent: 'space-between', padding: 16 },
    headerTitle: { fontSize: 18, fontWeight: 'bold' },
    scrollContent: { padding: 16 },
    priceSection: { marginBottom: 20 },
    currentPrice: { fontSize: 32, fontWeight: 'bold' },
    predictionText: { fontSize: 16, color: Colors.text.secondary },
    chartContainer: { alignItems: 'center', marginBottom: 20 },
    chartTitle: { marginBottom: 8 },
    detailsCard: { padding: 16, backgroundColor: Colors.card.light, borderRadius: 12 },
    cardTitle: { fontWeight: 'bold', marginBottom: 12 },
    detailRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
    newsSection: { marginTop: 24, marginBottom: 40 },
    sectionTitle: { fontSize: 20, fontWeight: 'bold', marginBottom: 16, color: Colors.text.light },
    summaryCard: { padding: 16, backgroundColor: Colors.primary + '1A', borderRadius: 12, marginBottom: 16, borderWidth: 1, borderColor: Colors.primary + '40' },
    summaryTitle: { fontWeight: 'bold', color: Colors.primary, marginBottom: 8, fontSize: 16 },
    summaryText: { color: Colors.text.light, lineHeight: 22, fontSize: 14 },
    newsCard: { padding: 16, backgroundColor: Colors.card.light, borderRadius: 12, marginBottom: 12, shadowColor: '#000', shadowOffset: { width: 0, height: 1 }, shadowOpacity: 0.05, shadowRadius: 2, elevation: 2 },
    newsText: { color: Colors.text.secondary, lineHeight: 20, fontSize: 14 },
    noNewsText: { color: Colors.text.secondary, fontStyle: 'italic', marginTop: 8 }
});
