import React, { useState, useEffect } from 'react';
import { StyleSheet, ScrollView, View, Text, ActivityIndicator } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { getBackendUrl } from '../../constants/Api';
import { Colors } from '../../constants/Colors';
import { MOCK_STOCKS } from '../../constants/MockData';
import { StockCard } from '../../components/StockCard';
import { Ionicons } from '@expo/vector-icons';

export default function HomeScreen() {
  const [stocks, setStocks] = useState(MOCK_STOCKS);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchAAPLData = async () => {
      setLoading(true);
      try {
        console.log("Fetching AAPL inference data from backend for HomeScreen...");
        const backendUrl = getBackendUrl();
        const response = await fetch(`${backendUrl}/predict`);
        if (!response.ok) throw new Error('Network response was not ok');
        const data = await response.json();
        console.log("Inference Data Received for AAPL (HomeScreen):", JSON.stringify(data, null, 2));

        const predicted1D = data.forecast && data.forecast.length > 0 ? data.forecast[0].price : data.current_price;
        const predicted16D = data.forecast && data.forecast.length > 15 ? data.forecast[15].price : data.current_price;
        const change1D = ((predicted1D - data.current_price) / data.current_price) * 100;
        const change16D = ((predicted16D - data.current_price) / data.current_price) * 100;

        let sentimentText = 'Neutral';
        if (data.sentiment_score > 0.2) sentimentText = 'Positive';
        if (data.sentiment_score > 0.5) sentimentText = 'Very Positive';
        if (data.sentiment_score < -0.2) sentimentText = 'Negative';
        if (data.sentiment_score < -0.5) sentimentText = 'Very Negative';

        setStocks(prevStocks => prevStocks.map(stock => {
          if (stock.id === 'AAPL') {
            return {
              ...stock,
              currentPrice: data.current_price,
              predictedPrice1D: predicted1D,
              predictedPrice16D: predicted16D,
              change1D: Number(change1D.toFixed(2)),
              change16D: Number(change16D.toFixed(2)),
              confidence: data.sentiment_confidence,
              sentiment: sentimentText,
            };
          }
          return stock;
        }));
      } catch (error) {
        console.error("Error fetching AAPL data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchAAPLData();
  }, []);

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>InformerAI</Text>
        <Ionicons name="settings-outline" size={24} color={Colors.text.light} />
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.sectionHeader}>
          <Text style={styles.sectionTitle}>Your Watchlist</Text>
          <Text style={styles.seeAll}>See All</Text>
        </View>

        {loading && <ActivityIndicator size="small" color={Colors.primary} style={{ marginBottom: 16 }} />}

        {stocks.map((stock) => (
          <StockCard key={stock.id} stock={stock} />
        ))}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background.light,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border.light,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: Colors.text.light,
  },
  scrollContent: {
    padding: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: Colors.text.light,
  },
  seeAll: {
    fontSize: 14,
    color: Colors.primary,
  }
});
