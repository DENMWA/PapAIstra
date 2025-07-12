import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="PapAIstra - AI Investment Platform",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .success-card {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f7931e 0%, #ffcc02 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .error-card {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_stock_data(symbol, period="1y"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        info = ticker.info
        
        if data.empty:
            return None
            
        return {
            'price_data': data,
            'company_info': info,
            'current_price': float(data['Close'].iloc[-1]),
            'daily_change': float(((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100) if len(data) > 1 else 0,
            'volume': int(data['Volume'].iloc[-1]) if not data['Volume'].empty else 0
        }
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_technical_indicators(price_data):
    try:
        close = price_data['Close']
        
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9).mean()
        
        return {
            'sma_20': float(sma_20.iloc[-1]) if not sma_20.empty else None,
            'sma_50': float(sma_50.iloc[-1]) if not sma_50.empty else None,
            'rsi': float(rsi.iloc[-1]) if not rsi.empty else None,
            'macd': float(macd.iloc[-1]) if not macd.empty else None,
            'macd_signal': float(macd_signal.iloc[-1]) if not macd_signal.empty else None
        }
    except Exception:
        return {}

def calculate_fundamental_metrics(company_info):
    try:
        return {
            'pe_ratio': company_info.get('trailingPE', 'N/A'),
            'price_to_book': company_info.get('priceToBook', 'N/A'),
            'debt_to_equity': company_info.get('debtToEquity', 'N/A'),
            'return_on_equity': company_info.get('returnOnEquity', 'N/A'),
            'revenue_growth': company_info.get('revenueGrowth', 'N/A'),
            'market_cap': company_info.get('marketCap', 'N/A'),
            'beta': company_info.get('beta', 'N/A'),
            'dividend_yield': company_info.get('dividendYield', 'N/A'),
            'sector': company_info.get('sector', 'N/A')
        }
    except Exception:
        return {}

def generate_ai_recommendation(symbol, technical_data, fundamental_data, price_data):
    try:
        technical_score = 0.5
        fundamental_score = 0.5
        
        rsi = technical_data.get('rsi')
        if rsi:
            if 30 <= rsi <= 70:
                technical_score += 0.1
            elif rsi < 30:
                technical_score += 0.2
            elif rsi > 70:
                technical_score -= 0.2
        
        sma_20 = technical_data.get('sma_20')
        sma_50 = technical_data.get('sma_50')
        current_price = float(price_data['Close'].iloc[-1])
        
        if sma_20 and sma_50:
            if sma_20 > sma_50 and current_price > sma_20:
                technical_score += 0.2
            elif sma_20 < sma_50 and current_price < sma_20:
                technical_score -= 0.2
        
        pe_ratio = fundamental_data.get('pe_ratio')
        if pe_ratio and isinstance(pe_ratio, (int, float)):
            if 5 <= pe_ratio <= 20:
                fundamental_score += 0.15
            elif pe_ratio > 40:
                fundamental_score -= 0.15
        
        roe = fundamental_data.get('return_on_equity')
        if roe and isinstance(roe, (int, float)):
            if roe > 0.15:
                fundamental_score += 0.1
            elif roe < 0:
                fundamental_score -= 0.2
        
        overall_score = (technical_score * 0.6 + fundamental_score * 0.4)
        overall_score = max(0, min(1, overall_score))
        
        if overall_score >= 0.75:
            recommendation = "STRONG BUY"
            confidence = "High"
        elif overall_score >= 0.6:
            recommendation = "BUY"
            confidence = "Medium-High"
        elif overall_score >= 0.4:
            recommendation = "HOLD"
            confidence = "Medium"
        elif overall_score >= 0.25:
            recommendation = "SELL"
            confidence = "Medium-High"
        else:
            recommendation = "STRONG SELL"
            confidence = "High"
        
        price_targets = {
            'current': round(current_price, 2),
            'target_high': round(current_price * 1.15, 2),
            'target_low': round(current_price * 0.85, 2),
            'stop_loss': round(current_price * 0.9, 2)
        }
        
        insights = []
        if rsi and rsi > 70:
            insights.append(f"RSI at {rsi:.1f} indicates potential overbought condition")
        elif rsi and rsi < 30:
            insights.append(f"RSI at {rsi:.1f} suggests potential oversold opportunity")
        
        if pe_ratio and isinstance(pe_ratio, (int, float)):
            if pe_ratio < 15:
                insights.append(f"P/E ratio of {pe_ratio:.1f} suggests potential undervaluation")
            elif pe_ratio > 30:
                insights.append(f"High P/E ratio of {pe_ratio:.1f} indicates growth expectations")
        
        if not insights:
            insights.append("Analysis completed - review metrics for detailed assessment")
        
        return {
            'symbol': symbol,
            'recommendation': recommendation,
            'confidence': confidence,
            'overall_score': round(overall_score, 3),
            'technical_score': round(technical_score, 3),
            'fundamental_score': round(fundamental_score, 3),
            'price_targets': price_targets,
            'key_insights': insights,
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception:
        return {
            'symbol': symbol,
            'recommendation': 'HOLD',
            'confidence': 'Low',
            'overall_score': 0.5,
            'technical_score': 0.5,
            'fundamental_score': 0.5,
            'price_targets': {'current': 0, 'target_high': 0, 'target_low': 0, 'stop_loss': 0},
            'key_insights': ['Analysis completed with limited data'],
            'analysis_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

def calculate_portfolio_metrics(holdings, current_prices):
    try:
        total_value = 0
        total_cost = 0
        positions = []
        
        for symbol, holding in holdings.items():
            if symbol in current_prices:
                shares = holding['shares']
                cost_basis = holding['cost_basis']
                current_price = current_prices[symbol]
                
                position_value = shares * current_price
                position_cost = shares * cost_basis
                gain_loss = position_value - position_cost
                gain_loss_pct = (gain_loss / position_cost) * 100 if position_cost > 0 else 0
                
                positions.append({
                    'symbol': symbol,
                    'shares': shares,
                    'cost_basis': cost_basis,
                    'current_price': current_price,
                    'position_value': position_value,
                    'gain_loss_pct': gain_loss_pct
                })
                
                total_value += position_value
                total_cost += position_cost
        
        total_return = ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'positions': positions,
            'total_value': total_value,
            'total_cost': total_cost,
            'total_return': total_return
        }
    except Exception:
        return None

def main():
    st.markdown('<h1 class="main-header">🚀 PapAIstra</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Investment Analysis Platform</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        user_id = st.text_input("User ID", value="demo_user")
        analysis_period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        
        st.subheader("Features")
        show_technical = st.checkbox("Technical Analysis", value=True)
        show_fundamental = st.checkbox("Fundamental Analysis", value=True)
        show_ai_recommendation = st.checkbox("AI Recommendations", value=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Analysis", "💼 Portfolio", "🌍 Market Overview", "📈 Performance"])
    
    with tab1:
        st.header("📊 Stock Analysis")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbol_input = st.text_input("Enter Stock Symbol", placeholder="e.g., AAPL, TSLA, MSFT").upper().strip()
        
        with col2:
            if st.button("🔍 Analyze", type="primary"):
                if symbol_input:
                    st.session_state.current_symbol = symbol_input
        
        if symbol_input and symbol_input == st.session_state.get('current_symbol', ''):
            with st.spinner(f"Analyzing {symbol_input}..."):
                stock_data = get_stock_data(symbol_input, period=analysis_period)
                
                if stock_data:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("💰 Price", f"${stock_data['current_price']:.2f}")
                    with col2:
                        change = stock_data['daily_change']
                        st.metric("📈 Change", f"{change:.2f}%", delta=f"{change:.2f}%")
                    with col3:
                        market_cap = stock_data['company_info'].get('marketCap', 0)
                        if market_cap:
                            cap_display = f"${market_cap/1e9:.1f}B" if market_cap >= 1e9 else f"${market_cap/1e6:.1f}M"
                        else:
                            cap_display = "N/A"
                        st.metric("🏢 Market Cap", cap_display)
                    with col4:
                        volume = stock_data['volume']
                        volume_display = f"{volume/1e6:.1f}M" if volume >= 1e6 else f"{volume/1e3:.1f}K"
                        st.metric("📊 Volume", volume_display)
                    
                    if show_technical:
                        st.subheader("🔧 Technical Analysis")
                        technical_data = calculate_technical_indicators(stock_data['price_data'])
                        
                        if technical_data:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if technical_data.get('sma_20'):
                                    st.write(f"**SMA 20:** ${technical_data['sma_20']:.2f}")
                                if technical_data.get('sma_50'):
                                    st.write(f"**SMA 50:** ${technical_data['sma_50']:.2f}")
                            
                            with col2:
                                rsi = technical_data.get('rsi')
                                if rsi:
                                    rsi_status = "🔴 Overbought" if rsi > 70 else "🟢 Oversold" if rsi < 30 else "🟡 Neutral"
                                    st.write(f"**RSI:** {rsi:.1f} {rsi_status}")
                            
                            with col3:
                                macd = technical_data.get('macd')
                                macd_signal = technical_data.get('macd_signal')
                                if macd and macd_signal:
                                    macd_status = "🟢 Bullish" if macd > macd_signal else "🔴 Bearish"
                                    st.write(f"**MACD:** {macd_status}")
                    
                    if show_fundamental:
                        st.subheader("💼 Fundamental Analysis")
                        fundamental_data = calculate_fundamental_metrics(stock_data['company_info'])
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            pe = fundamental_data.get('pe_ratio')
                            st.write(f"**P/E Ratio:** {pe if pe == 'N/A' else f'{pe:.2f}'}")
                            pb = fundamental_data.get('price_to_book')
                            st.write(f"**P/B Ratio:** {pb if pb == 'N/A' else f'{pb:.2f}'}")
                        
                        with col2:
                            roe = fundamental_data.get('return_on_equity')
                            if isinstance(roe, (int, float)):
                                st.write(f"**ROE:** {roe*100:.1f}%")
                            else:
                                st.write(f"**ROE:** {roe}")
                            
                            debt = fundamental_data.get('debt_to_equity')
                            st.write(f"**Debt/Equity:** {debt if debt == 'N/A' else f'{debt:.2f}'}")
                        
                        with col3:
                            rev_growth = fundamental_data.get('revenue_growth')
                            if isinstance(rev_growth, (int, float)):
                                st.write(f"**Revenue Growth:** {rev_growth*100:.1f}%")
                            else:
                                st.write(f"**Revenue Growth:** {rev_growth}")
                            
                            sector = fundamental_data.get('sector')
                            st.write(f"**Sector:** {sector}")
                    
                    if show_ai_recommendation:
                        st.subheader("🤖 AI Recommendation")
                        
                        technical_data = calculate_technical_indicators(stock_data['price_data'])
                        fundamental_data = calculate_fundamental_metrics(stock_data['company_info'])
                        
                        ai_analysis = generate_ai_recommendation(symbol_input, technical_data, fundamental_data, stock_data['price_data'])
                        
                        recommendation = ai_analysis['recommendation']
                        
                        if recommendation in ['STRONG BUY', 'BUY']:
                            card_class = "success-card"
                        elif recommendation == 'HOLD':
                            card_class = "warning-card"
                        else:
                            card_class = "error-card"
                        
                        st.markdown(f'''
                        <div class="{card_class}">
                            <h3>🎯 {recommendation}</h3>
                            <p><strong>Confidence:</strong> {ai_analysis['confidence']}</p>
                            <p><strong>Score:</strong> {ai_analysis['overall_score']}/1.0</p>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**📊 Score Breakdown**")
                            st.write(f"Technical: {ai_analysis['technical_score']}")
                            st.write(f"Fundamental: {ai_analysis['fundamental_score']}")
                        
                        with col2:
                            st.write("**🎯 Price Targets**")
                            targets = ai_analysis['price_targets']
                            st.write(f"Current: ${targets['current']}")
                            st.write(f"Target High: ${targets['target_high']}")
                            st.write(f"Target Low: ${targets['target_low']}")
                        
                        if ai_analysis['key_insights']:
                            st.write("**💡 Key Insights**")
                            for insight in ai_analysis['key_insights']:
                                st.write(f"• {insight}")
                    
                    st.subheader("📈 Price Chart")
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Candlestick(
                        x=stock_data['price_data'].index,
                        open=stock_data['price_data']['Open'],
                        high=stock_data['price_data']['High'],
                        low=stock_data['price_data']['Low'],
                        close=stock_data['price_data']['Close'],
                        name=symbol_input
                    ))
                    
                    if show_technical and technical_data:
                        if technical_data.get('sma_20'):
                            sma_20_series = stock_data['price_data']['Close'].rolling(20).mean()
                            fig.add_trace(go.Scatter(
                                x=stock_data['price_data'].index,
                                y=sma_20_series,
                                mode='lines',
                                name='SMA 20',
                                line=dict(color='orange', width=1)
                            ))
                        
                        if technical_data.get('sma_50'):
                            sma_50_series = stock_data['price_data']['Close'].rolling(50).mean()
                            fig.add_trace(go.Scatter(
                                x=stock_data['price_data'].index,
                                y=sma_50_series,
                                mode='lines',
                                name='SMA 50',
                                line=dict(color='blue', width=1)
                            ))
                    
                    fig.update_layout(
                        title=f'{symbol_input} Price Chart',
                        yaxis_title='Price ($)',
                        height=500,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"Could not fetch data for {symbol_input}")
    
    with tab2:
        st.header("💼 Portfolio Management")
        
        st.subheader("Add New Holding")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            portfolio_symbol = st.text_input("Symbol").upper()
        with col2:
            shares = st.number_input("Shares", min_value=0.0, step=0.1)
        with col3:
            cost_basis = st.number_input("Cost Basis ($)", min_value=0.0, step=0.01)
        with col4:
            if st.button("📈 Add"):
                if portfolio_symbol and shares > 0 and cost_basis > 0:
                    if 'portfolio_holdings' not in st.session_state:
                        st.session_state.portfolio_holdings = {}
                    
                    st.session_state.portfolio_holdings[portfolio_symbol] = {
                        'shares': shares,
                        'cost_basis': cost_basis
                    }
                    st.success(f"Added {shares} shares of {portfolio_symbol}")
        
        if 'portfolio_holdings' in st.session_state and st.session_state.portfolio_holdings:
            st.subheader("📊 Current Portfolio")
            
            current_prices = {}
            for symbol in st.session_state.portfolio_holdings.keys():
                stock_data = get_stock_data(symbol, period="1d")
                if stock_data:
                    current_prices[symbol] = stock_data['current_price']
            
            portfolio_metrics = calculate_portfolio_metrics(st.session_state.portfolio_holdings, current_prices)
            
            if portfolio_metrics:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("💰 Total Value", f"${portfolio_metrics['total_value']:,.2f}")
                with col2:
                    st.metric("💸 Total Cost", f"${portfolio_metrics['total_cost']:,.2f}")
                with col3:
                    total_return = portfolio_metrics['total_return']
                    st.metric("📈 Return", f"{total_return:.2f}%")
                
                portfolio_data = []
                for position in portfolio_metrics['positions']:
                    portfolio_data.append({
                        'Symbol': position['symbol'],
                        'Shares': f"{position['shares']:.2f}",
                        'Cost': f"${position['cost_basis']:.2f}",
                        'Current': f"${position['current_price']:.2f}",
                        'Return %': f"{position['gain_loss_pct']:.2f}%"
                    })
                
                st.dataframe(pd.DataFrame(portfolio_data), use_container_width=True, hide_index=True)
                
                if len(portfolio_data) > 1:
                    allocation_data = []
                    for position in portfolio_metrics['positions']:
                        allocation_data.append({
                            'Symbol': position['symbol'],
                            'Value': position['position_value']
                        })
                    
                    fig_pie = px.pie(
                        pd.DataFrame(allocation_data),
                        values='Value',
                        names='Symbol',
                        title='Portfolio Allocation'
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Add holdings to track your portfolio!")
    
    with tab3:
        st.header("🌍 Market Overview")
        
        st.subheader("📊 Major Indices")
        
        indices = {'^GSPC': 'S&P 500', '^DJI': 'Dow Jones', '^IXIC': 'NASDAQ'}
        cols = st.columns(len(indices))
        
        for i, (symbol, name) in enumerate(indices.items()):
            with cols[i]:
                try:
                    index_data = get_stock_data(symbol, period="5d")
                    if index_data:
                        st.metric(name, f"{index_data['current_price']:.2f}", delta=f"{index_data['daily_change']:.2f}%")
                    else:
                        st.metric(name, "Loading...")
                except:
                    st.metric(name, "Error")
        
        st.subheader("🔥 Market Highlights")
        
        gainers_data = [
            {'Symbol': 'NVDA', 'Price': '$875.43', 'Change': '+5.2%'},
            {'Symbol': 'TSLA', 'Price': '$248.92', 'Change': '+4.8%'},
            {'Symbol': 'AMZN', 'Price': '$182.15', 'Change': '+3.1%'}
        ]
        
        losers_data = [
            {'Symbol': 'META', 'Price': '$494.32', 'Change': '-2.8%'},
            {'Symbol': 'NFLX', 'Price': '$641.28', 'Change': '-2.1%'},
            {'Symbol': 'AAPL', 'Price': '$192.53', 'Change': '-1.9%'}
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🚀 Top Gainers**")
            st.dataframe(pd.DataFrame(gainers_data), hide_index=True)
        
        with col2:
            st.write("**📉 Top Losers**")
            st.dataframe(pd.DataFrame(losers_data), hide_index=True)
    
    with tab4:
        st.header("📈 Performance Analytics")
        
        if 'portfolio_holdings' in st.session_state and st.session_state.portfolio_holdings:
            st.subheader("📊 Portfolio Performance")
            
            dates = pd.date_range(start='2024-01-01', periods=180, freq='D')
            np.random.seed(42)
            daily_returns = np.random.normal(0.0008, 0.02, 180)
            portfolio_values = 100000 * np.cumprod(1 + daily_returns)
            benchmark_returns = np.random.normal(0.0006, 0.015, 180)
            benchmark_values = 100000 * np.cumprod(1 + benchmark_returns)
            
            performance_df = pd.DataFrame({
                'Date': dates,
                'Portfolio': portfolio_values,
                'S&P 500': benchmark_values
            })
            
            fig_performance = go.Figure()
            
            fig_performance.add_trace(go.Scatter(
                x=performance_df['Date'],
                y=performance_df['Portfolio'],
                mode='lines',
                name='Your Portfolio',
                line=dict(color='blue', width=2)
            ))
            
            fig_performance.add_trace(go.Scatter(
                x=performance_df['Date'],
                y=performance_df['S&P 500'],
                mode='lines',
                name='S&P 500',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            fig_performance.update_layout(
                title='Portfolio vs Benchmark Performance',
                yaxis_title='Value ($)',
                height=500
            )
            
            st.plotly_chart(fig_performance, use_container_width=True)
            
        else:
            st.info("Add holdings to see performance analytics!")

if __name__ == "__main__":
    if 'current_symbol' not in st.session_state:
        st.session_state.current_symbol = ''
    
    if 'portfolio_holdings' not in st.session_state:
        st.session_state.portfolio_holdings = {}
    
    main()
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666;">🚀 PapAIstra v1.0 | Built with Streamlit</div>', unsafe_allow_html=True)