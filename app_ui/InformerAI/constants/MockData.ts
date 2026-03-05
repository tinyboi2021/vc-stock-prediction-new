export const MOCK_STOCKS = [
    {
        id: 'AAPL',
        name: 'Apple Inc.',
        currentPrice: 177.80,
        predictedPrice1D: 178.45,
        predictedPrice16D: 180.14,
        change1D: 0.37,
        change16D: 1.32,
        confidence: 0.85,
        sentiment: 'Positive',
        isTrending: false,
    },
    {
        id: 'NVDA',
        name: 'NVIDIA Corp',
        currentPrice: 850.20,
        predictedPrice1D: 862.10,
        predictedPrice16D: 895.40,
        change1D: 1.40,
        change16D: 5.32,
        confidence: 0.91,
        sentiment: 'Very Positive',
        isTrending: true,
    },
    {
        id: 'TSLA',
        name: 'Tesla Inc.',
        currentPrice: 175.30,
        predictedPrice1D: 174.10,
        predictedPrice16D: 165.20,
        change1D: -0.68,
        change16D: -5.76,
        confidence: 0.72,
        sentiment: 'Negative',
        isTrending: true,
    },
    {
        id: 'MSFT',
        name: 'Microsoft Corp',
        currentPrice: 415.50,
        predictedPrice1D: 416.20,
        predictedPrice16D: 422.80,
        change1D: 0.17,
        change16D: 1.76,
        confidence: 0.88,
        sentiment: 'Positive',
        isTrending: false,
    },
    {
        id: 'AMD',
        name: 'Advanced Micro Devices',
        currentPrice: 180.50,
        predictedPrice1D: 182.30,
        predictedPrice16D: 195.10,
        change1D: 1.00,
        change16D: 8.09,
        confidence: 0.79,
        sentiment: 'Neutral',
        isTrending: false,
    },
];

export const MOCK_NEWS = [
    {
        id: 1,
        title: 'NVIDIA Reports Q4 Earnings',
        source: 'TechCrunch',
        time: '2 mins ago',
        sentiment: 'Very Positive',
        stocks: ['NVDA'],
        impact: 'High',
    },
    {
        id: 2,
        title: 'New iPhone 16 Pre-orders',
        source: 'MacRumors',
        time: '15 mins ago',
        sentiment: 'Positive',
        stocks: ['AAPL'],
        impact: 'Medium',
    },
    {
        id: 3,
        title: 'Interest Rate Cut Hopes Fade',
        source: 'Bloomberg',
        time: '1 hour ago',
        sentiment: 'Negative',
        stocks: ['SPY', 'QQQ'],
        impact: 'High',
    },
];

export const MOCK_PORTFOLIO = [
    {
        id: 'AAPL',
        shares: 50,
        avgPrice: 170.00,
    },
    {
        id: 'NVDA',
        shares: 10,
        avgPrice: 650.00,
    }
];

export const MOCK_ALERTS = [
    {
        id: 1,
        type: 'target',
        stock: 'AAPL',
        message: 'Target Reached: Predicted $178.50',
        time: '2 hours ago',
        read: false,
    },
    {
        id: 2,
        type: 'volatility',
        stock: 'NVDA',
        message: 'Volatility Spike: ADX jumped to 0.45',
        time: '4 hours ago',
        read: true,
    },
];
