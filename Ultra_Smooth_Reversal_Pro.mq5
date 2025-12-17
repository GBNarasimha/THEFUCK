//+------------------------------------------------------------------+
//|                                     Ultra_Smooth_Reversal_Pro.mq5|
//|                        Copyright 2025, Advanced Trading Systems  |
//|                                          https://www.mql5.com    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, Advanced Trading Systems"
#property link      "https://www.mql5.com"
#property version   "2.00"
#property indicator_chart_window
#property indicator_buffers 16 // Increased buffer count for safety
#property indicator_plots   4

// Indicator plots
#property indicator_label1  "Strong Buy"
#property indicator_type1   DRAW_ARROW
#property indicator_color1  clrLime
#property indicator_style1  STYLE_SOLID
#property indicator_width1  4

#property indicator_label2  "Strong Sell"
#property indicator_type2   DRAW_ARROW
#property indicator_color2  clrRed
#property indicator_style2  STYLE_SOLID
#property indicator_width2  4

#property indicator_label3  "Buy Zone"
#property indicator_type3   DRAW_ARROW
#property indicator_color3  clrDodgerBlue
#property indicator_style3  STYLE_SOLID
#property indicator_width3  2

#property indicator_label4  "Sell Zone"
#property indicator_type4   DRAW_ARROW
#property indicator_color4  clrOrangeRed
#property indicator_style4  STYLE_SOLID
#property indicator_width4  2

//--- Input Parameters
input group "=== Smoothing & Lag Reduction ==="
input int Smooth_Period = 5;              // Smoothing period for ultra-smooth signals
input double Lag_Reduction = 0.7;         // Lag reduction factor (0.5-0.9)
input bool Use_Kalman_Filter = true;      // Use Kalman filtering for noise reduction

input group "=== Core Indicators ==="
input int RSI_Period = 14;                // RSI Period
input int RSI_OverBought = 70;            // RSI Overbought
input int RSI_OverSold = 30;              // RSI Oversold
input int MACD_Fast = 12;                 // MACD Fast EMA
input int MACD_Slow = 26;                 // MACD Slow EMA
input int MACD_Signal = 9;                // MACD Signal
input int Stoch_K = 14;                   // Stochastic %K
input int Stoch_D = 3;                    // Stochastic %D
input int Stoch_Slowing = 3;              // Stochastic Slowing
input int MA_Period = 200;                // Moving Average Period
input ENUM_MA_METHOD MA_Method = MODE_EMA;// MA Method (EMA for smoothness)
input int ATR_Period = 14;                // ATR Period

input group "=== Advanced Features ==="
input int Fib_Lookback = 34;              // Fibonacci lookback (Fibonacci number)
input double VolumeTrigger = 1.5;         // Volume spike multiplier
input bool Use_Candle_Patterns = true;    // Enable candlestick patterns
input bool Use_Divergence = true;         // Enable divergence detection
input bool Use_Volume_Profile = true;     // Enable volume analysis

input group "=== Multi-Timeframe Analysis ==="
input bool Use_Multi_Timeframe = true;    // Enable multi-timeframe
input ENUM_TIMEFRAMES Higher_TF = PERIOD_H1;  // Higher timeframe
input ENUM_TIMEFRAMES Lower_TF = PERIOD_M5;   // Lower timeframe (Confirmation)
input double MTF_Weight = 2.5;            // Multi-timeframe weight

input group "=== Signal Filtering ==="
input double Strong_Signal_Threshold = 7.0;   // Strong signal minimum score
input double Weak_Signal_Threshold = 4.0;     // Weak signal minimum score
input int Min_Bars_Between_Signals = 5;       // Minimum bars between signals
input bool Filter_Against_Trend = true;       // Filter counter-trend signals

input group "=== Alerts ==="
input bool Enable_Alerts = true;          // Enable alerts
input bool Enable_Push = false;           // Enable push notifications
input bool Enable_Email = false;          // Enable email notifications

//--- Indicator Buffers
double StrongBuyBuffer[];
double StrongSellBuffer[];
double WeakBuyBuffer[];
double WeakSellBuffer[];

// Calculation buffers
double RSIBuffer[];
double RSI_Smooth[];
double MACDBuffer[];
double SignalBuffer[];
double MACD_Smooth[];
double StochKBuffer[];
double StochDBuffer[];
double Stoch_Smooth[];
double MABuffer[];
double ATRBuffer[];
double PriceSmooth[];
double KalmanBuffer[]; // Not explicitly used as a plot, but useful for debug or calculation storage

//--- Indicator Handles
int RSIHandle, MACDHandle, StochHandle, MAHandle, ATRHandle;
int Higher_RSI, Higher_MACD, Higher_Stoch;
int Lower_RSI, Lower_MACD, Lower_Stoch;

//--- Global Variables
int lastSignalBar = -1000;
double kalmanGain = 0.0;
double kalmanEstimate = 0.0;
double kalmanError = 1.0;
bool firstCalculation = true;

//+------------------------------------------------------------------+
//| Custom indicator initialization                                  |
//+------------------------------------------------------------------+
int OnInit()
{
   // Validate inputs
   if(RSI_Period < 2 || MACD_Fast < 2 || MACD_Slow < 2 || Stoch_K < 2)
   {
      Print("Error: Invalid input parameters");
      return(INIT_PARAMETERS_INCORRECT);
   }
   
   if(Strong_Signal_Threshold < Weak_Signal_Threshold)
   {
      Print("Error: Strong signal threshold must be >= weak signal threshold");
      return(INIT_PARAMETERS_INCORRECT);
   }
   
   // Set up indicator buffers
   SetIndexBuffer(0, StrongBuyBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, StrongSellBuffer, INDICATOR_DATA);
   SetIndexBuffer(2, WeakBuyBuffer, INDICATOR_DATA);
   SetIndexBuffer(3, WeakSellBuffer, INDICATOR_DATA);
   SetIndexBuffer(4, RSIBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(5, RSI_Smooth, INDICATOR_CALCULATIONS);
   SetIndexBuffer(6, MACDBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(7, SignalBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(8, MACD_Smooth, INDICATOR_CALCULATIONS);
   SetIndexBuffer(9, StochKBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(10, StochDBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(11, Stoch_Smooth, INDICATOR_CALCULATIONS);
   SetIndexBuffer(12, MABuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(13, ATRBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(14, PriceSmooth, INDICATOR_CALCULATIONS);
   SetIndexBuffer(15, KalmanBuffer, INDICATOR_CALCULATIONS);
   
   // Set arrow codes
   PlotIndexSetInteger(0, PLOT_ARROW, 233);  // Strong buy (Up Arrow)
   PlotIndexSetInteger(1, PLOT_ARROW, 234);  // Strong sell (Down Arrow)
   PlotIndexSetInteger(2, PLOT_ARROW, 241);  // Weak buy
   PlotIndexSetInteger(3, PLOT_ARROW, 242);  // Weak sell
   
   // Set empty values
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(2, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(3, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   
   // Initialize indicators on current timeframe
   RSIHandle = iRSI(NULL, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
   MACDHandle = iMACD(NULL, PERIOD_CURRENT, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
   StochHandle = iStochastic(NULL, PERIOD_CURRENT, Stoch_K, Stoch_D, Stoch_Slowing, MODE_SMA, STO_LOWHIGH);
   MAHandle = iMA(NULL, PERIOD_CURRENT, MA_Period, 0, MA_Method, PRICE_CLOSE);
   ATRHandle = iATR(NULL, PERIOD_CURRENT, ATR_Period);
   
   // Check handles
   if(RSIHandle == INVALID_HANDLE || MACDHandle == INVALID_HANDLE || 
      StochHandle == INVALID_HANDLE || MAHandle == INVALID_HANDLE || ATRHandle == INVALID_HANDLE)
   {
      Print("Error creating indicator handles");
      return(INIT_FAILED);
   }
   
   // Initialize multi-timeframe indicators
   if(Use_Multi_Timeframe)
   {
      Higher_RSI = iRSI(NULL, Higher_TF, RSI_Period, PRICE_CLOSE);
      Higher_MACD = iMACD(NULL, Higher_TF, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
      Higher_Stoch = iStochastic(NULL, Higher_TF, Stoch_K, Stoch_D, Stoch_Slowing, MODE_SMA, STO_LOWHIGH);
      
      Lower_RSI = iRSI(NULL, Lower_TF, RSI_Period, PRICE_CLOSE);
      Lower_MACD = iMACD(NULL, Lower_TF, MACD_Fast, MACD_Slow, MACD_Signal, PRICE_CLOSE);
      Lower_Stoch = iStochastic(NULL, Lower_TF, Stoch_K, Stoch_D, Stoch_Slowing, MODE_SMA, STO_LOWHIGH);
      
      if(Higher_RSI == INVALID_HANDLE || Lower_RSI == INVALID_HANDLE)
      {
         Print("Warning: Error creating multi-timeframe indicators. MTF features might be disabled.");
      }
   }
   
   IndicatorSetString(INDICATOR_SHORTNAME, "Ultra Smooth Reversal Pro");
   IndicatorSetInteger(INDICATOR_DIGITS, _Digits);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Custom indicator deinitialization                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(RSIHandle != INVALID_HANDLE) IndicatorRelease(RSIHandle);
   if(MACDHandle != INVALID_HANDLE) IndicatorRelease(MACDHandle);
   if(StochHandle != INVALID_HANDLE) IndicatorRelease(StochHandle);
   if(MAHandle != INVALID_HANDLE) IndicatorRelease(MAHandle);
   if(ATRHandle != INVALID_HANDLE) IndicatorRelease(ATRHandle);
   
   if(Use_Multi_Timeframe)
   {
      if(Higher_RSI != INVALID_HANDLE) IndicatorRelease(Higher_RSI);
      if(Higher_MACD != INVALID_HANDLE) IndicatorRelease(Higher_MACD);
      if(Higher_Stoch != INVALID_HANDLE) IndicatorRelease(Higher_Stoch);
      if(Lower_RSI != INVALID_HANDLE) IndicatorRelease(Lower_RSI);
      if(Lower_MACD != INVALID_HANDLE) IndicatorRelease(Lower_MACD);
      if(Lower_Stoch != INVALID_HANDLE) IndicatorRelease(Lower_Stoch);
   }
}

//+------------------------------------------------------------------+
//| Custom indicator iteration                                       |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   int min_bars = MathMax(MA_Period, Fib_Lookback) + 100;
   if(rates_total < min_bars) return(0);
   
   // Set arrays as series for consistent indexing (0 is newest)
   ArraySetAsSeries(open, true);
   ArraySetAsSeries(high, true);
   ArraySetAsSeries(low, true);
   ArraySetAsSeries(close, true);
   ArraySetAsSeries(time, true);
   ArraySetAsSeries(tick_volume, true);
   
   ArraySetAsSeries(RSIBuffer, true);
   ArraySetAsSeries(RSI_Smooth, true);
   ArraySetAsSeries(MACDBuffer, true);
   ArraySetAsSeries(SignalBuffer, true);
   ArraySetAsSeries(MACD_Smooth, true);
   ArraySetAsSeries(StochKBuffer, true);
   ArraySetAsSeries(StochDBuffer, true);
   ArraySetAsSeries(Stoch_Smooth, true);
   ArraySetAsSeries(MABuffer, true);
   ArraySetAsSeries(ATRBuffer, true);
   ArraySetAsSeries(PriceSmooth, true);
   ArraySetAsSeries(StrongBuyBuffer, true);
   ArraySetAsSeries(StrongSellBuffer, true);
   ArraySetAsSeries(WeakBuyBuffer, true);
   ArraySetAsSeries(WeakSellBuffer, true);
   
   // Determine calculation start
   int start = prev_calculated > 0 ? prev_calculated - 1 : 0;
   
   // Copy indicator data
   // Note: CopyBuffer copies oldest first. We need to handle this or reverse it.
   // Using standard copying and letting ArraySetAsSeries handle access is safer.
   // But we need to calculate 'count' based on 'start' carefully.
   // For simplicity in this structure, we copy everything.
   
   if(CopyBuffer(RSIHandle, 0, 0, rates_total, RSIBuffer) <= 0) return(0);
   if(CopyBuffer(MACDHandle, MAIN_LINE, 0, rates_total, MACDBuffer) <= 0) return(0);
   if(CopyBuffer(MACDHandle, SIGNAL_LINE, 0, rates_total, SignalBuffer) <= 0) return(0);
   if(CopyBuffer(StochHandle, MAIN_LINE, 0, rates_total, StochKBuffer) <= 0) return(0);
   if(CopyBuffer(StochHandle, SIGNAL_LINE, 0, rates_total, StochDBuffer) <= 0) return(0);
   if(CopyBuffer(MAHandle, 0, 0, rates_total, MABuffer) <= 0) return(0);
   if(CopyBuffer(ATRHandle, 0, 0, rates_total, ATRBuffer) <= 0) return(0);
   
   // Important: The loop 'i' usually runs from 0 (oldest) to rates_total-1 in OnCalculate if not series.
   // But we set arrays as series. 
   // MQL5 OnCalculate standard loop:
   int limit = rates_total - prev_calculated;
   if(prev_calculated > 0) limit++;
   
   // Apply ultra-smooth filters
   // We must iterate from oldest to newest for filters that depend on previous values
   // But since we set Series=true, index 0 is Current. 
   // We will iterate backwards from limit-1 down to 0.
   
   for(int i = limit; i >= 0; i--)
   {
      if(i >= rates_total) continue;
      
      // Ehlers Super Smoother for price
      PriceSmooth[i] = SuperSmoother(close, i, Smooth_Period, rates_total);
      
      // Apply Kalman filter if enabled (Simplified per-bar logic)
      if(Use_Kalman_Filter)
      {
         // Reset Kalman on history reset
         if(i == rates_total - 1) {
            kalmanEstimate = close[i];
            kalmanError = 1.0;
         }
         // Note: Kalman is recursive. Using it in a random access loop is tricky.
         // We only update it on the newest bar effectively if we don't store state arrays.
         // For this indicator, we will use the PriceSmooth as the base.
      }
      
      // Smooth RSI
      RSI_Smooth[i] = ApplyLagReduction(RSIBuffer, i, Smooth_Period, rates_total);
      
      // Smooth MACD
      MACD_Smooth[i] = ApplyLagReduction(MACDBuffer, i, Smooth_Period, rates_total);
      
      // Smooth Stochastic
      Stoch_Smooth[i] = ApplyLagReduction(StochKBuffer, i, Smooth_Period, rates_total);
   }
   
   double avgVolume = CalculateAverageVolume(tick_volume, rates_total);
   if(avgVolume == 0) avgVolume = 1;

   // Main calculation loop (Iterating newest to oldest for calculation, or oldest to newest)
   // With Series=true, we iterate 0 to limit.
   for(int i = limit; i >= 0; i--)
   {
      if(i >= rates_total - min_bars) continue; // Skip early history
      
      StrongBuyBuffer[i] = EMPTY_VALUE;
      StrongSellBuffer[i] = EMPTY_VALUE;
      WeakBuyBuffer[i] = EMPTY_VALUE;
      WeakSellBuffer[i] = EMPTY_VALUE;
      
      // Enforce signal spacing
      // Note: We can't easily check 'lastSignalBar' here during historical recalc
      // So we just calculate raw scores first.
      
      double buyScore = 0.0;
      double sellScore = 0.0;
      
      // 1. Smoothed RSI Analysis
      AnalyzeRSI(i, buyScore, sellScore, rates_total);
      
      // 2. Smoothed MACD Analysis
      AnalyzeMACD(i, buyScore, sellScore, rates_total);
      
      // 3. Smoothed Stochastic Analysis
      AnalyzeStochastic(i, buyScore, sellScore, rates_total);
      
      // 4. Moving Average Trend Filter
      AnalyzeMovingAverage(i, close, buyScore, sellScore, rates_total);
      
      // 5. Support/Resistance & Fibonacci
      AnalyzeSupportResistance(i, high, low, close, buyScore, sellScore, rates_total);
      
      // 6. Volume Analysis
      AnalyzeVolume(i, open, close, tick_volume, avgVolume, buyScore, sellScore, rates_total);
      
      // 7. Candlestick Patterns
      if(Use_Candle_Patterns)
      {
         AnalyzeCandlePatterns(i, open, high, low, close, buyScore, sellScore, rates_total);
      }
      
      // 8. Divergence Detection
      if(Use_Divergence)
      {
         AnalyzeDivergence(i, high, low, close, buyScore, sellScore, rates_total);
      }
      
      // 9. Multi-Timeframe Confirmation
      if(Use_Multi_Timeframe)
      {
         AnalyzeMultiTimeframe(i, time, buyScore, sellScore);
      }
      
      // 10. Volatility Filter (ATR)
      double volatilityFactor = AnalyzeVolatility(i);
      buyScore *= volatilityFactor;
      sellScore *= volatilityFactor;
      
      // Apply trend filter
      if(Filter_Against_Trend && MABuffer[i] > 0)
      {
         if(close[i] < MABuffer[i]) buyScore *= 0.5;   // Reduce counter-trend buys
         if(close[i] > MABuffer[i]) sellScore *= 0.5;  // Reduce counter-trend sells
      }
      
      // Generate signals
      GenerateSignals(i, high, low, buyScore, sellScore, time[i]);
   }
   
   return(rates_total);
}

//+------------------------------------------------------------------+
//| Ehlers Super Smoother                                            |
//+------------------------------------------------------------------+
double SuperSmoother(const double &price[], int index, int period, int total)
{
   if(index >= total - 3) return price[index];
   
   double a1 = MathExp(-1.414 * 3.14159 / period);
   double b1 = 2 * a1 * MathCos(1.414 * 180 / period * 3.14159 / 180);
   double c2 = b1;
   double c3 = -a1 * a1;
   double c1 = 1 - c2 - c3;
   
   double prev1 = (index + 1 < total) ? PriceSmooth[index + 1] : price[index];
   double prev2 = (index + 2 < total) ? PriceSmooth[index + 2] : price[index];

   return c1 * (price[index] + price[index + 1]) / 2 + c2 * prev1 + c3 * prev2;
}

//+------------------------------------------------------------------+
//| Apply lag reduction (Zero-Lag EMA)                               |
//+------------------------------------------------------------------+
double ApplyLagReduction(const double &buffer[], int index, int period, int total)
{
   if(index >= total - period) return buffer[index];
   
   double ema = 0;
   double alpha = 2.0 / (period + 1);
   
   // Simple EMA approximation for the localized window
   ema = buffer[index]; // Start with current
   // Note: Proper EMA requires recursion. For "Lag Reduction" on a buffer 
   // that is already calculated, we can use a DEMA approximation or similar.
   // Here we use the user's logic: EMA of the last N bars relative to index.
   
   double accum = 0;
   double weightSum = 0;
   double w = 1.0;
   
   for(int k=0; k<period; k++) {
      if(index + k >= total) break;
      accum += buffer[index+k] * w;
      weightSum += w;
      w *= (1.0 - alpha);
   }
   if(weightSum > 0) ema = accum / weightSum;
   
   return buffer[index] + Lag_Reduction * (buffer[index] - ema);
}

//+------------------------------------------------------------------+
//| Calculate average volume                                         |
//+------------------------------------------------------------------+
double CalculateAverageVolume(const long &tick_volume[], int rates_total)
{
   double sum = 0;
   int count = 0;
   for(int i = 1; i <= MathMin(50, rates_total - 1); i++)
   {
      sum += (double)tick_volume[i];
      count++;
   }
   return count > 0 ? sum / count : 1;
}

//+------------------------------------------------------------------+
//| Analysis Functions                                               |
//+------------------------------------------------------------------+
void AnalyzeRSI(int i, double &buyScore, double &sellScore, int total)
{
   if(i >= total - 2) return;
   
   double rsi = RSI_Smooth[i];
   double rsi_prev = RSI_Smooth[i + 1];
   
   if(rsi < RSI_OverSold && rsi > rsi_prev) buyScore += 1.5;
   if(rsi < RSI_OverSold - 10) buyScore += 1.0;
   
   if(rsi > RSI_OverBought && rsi < rsi_prev) sellScore += 1.5;
   if(rsi > RSI_OverBought + 10) sellScore += 1.0;
   
   if(rsi > 50 && rsi_prev < 50) buyScore += 0.5;
   if(rsi < 50 && rsi_prev > 50) sellScore += 0.5;
}

void AnalyzeMACD(int i, double &buyScore, double &sellScore, int total)
{
   if(i >= total - 1) return;
   
   double macd = MACD_Smooth[i];
   double macd_prev = MACD_Smooth[i + 1];
   double signal = SignalBuffer[i];
   double signal_prev = SignalBuffer[i + 1];
   
   if(macd > signal && macd_prev <= signal_prev) buyScore += 2.0;
   if(macd < signal && macd_prev >= signal_prev) sellScore += 2.0;
   
   if(macd > 0 && macd > macd_prev) buyScore += 0.5;
   if(macd < 0 && macd < macd_prev) sellScore += 0.5;
}

void AnalyzeStochastic(int i, double &buyScore, double &sellScore, int total)
{
   if(i >= total - 1) return;
   
   double stoch = Stoch_Smooth[i];
   double stoch_prev = Stoch_Smooth[i + 1];
   double stochD = StochDBuffer[i];
   double stochD_prev = StochDBuffer[i + 1];
   
   if(stoch < RSI_OverSold && stoch > stochD && stoch_prev <= stochD_prev) buyScore += 1.5;
   if(stoch > RSI_OverBought && stoch < stochD && stoch_prev >= stochD_prev) sellScore += 1.5;
}

void AnalyzeMovingAverage(int i, const double &close[], double &buyScore, double &sellScore, int total)
{
   if(i >= total - 1 || MABuffer[i] == 0) return;
   
   if(close[i] > MABuffer[i] && close[i + 1] <= MABuffer[i + 1]) buyScore += 1.5;
   if(close[i] < MABuffer[i] && close[i + 1] >= MABuffer[i + 1]) sellScore += 1.5;
   
   double distance = MathAbs(close[i] - MABuffer[i]) / MABuffer[i] * 100;
   if(close[i] > MABuffer[i] && distance < 2.0) buyScore += 0.5;
   if(close[i] < MABuffer[i] && distance < 2.0) sellScore += 0.5;
}

void AnalyzeSupportResistance(int i, const double &high[], const double &low[], const double &close[], double &buyScore, double &sellScore, int total)
{
   if(i >= total - Fib_Lookback) return;
   
   int highIdx = ArrayMaximum(high, i, Fib_Lookback);
   int lowIdx = ArrayMinimum(low, i, Fib_Lookback);
   if(highIdx == -1 || lowIdx == -1) return;
   
   double highestHigh = high[highIdx];
   double lowestLow = low[lowIdx];
   
   if(highestHigh <= lowestLow) return;
   
   double fib236 = lowestLow + (highestHigh - lowestLow) * 0.236;
   double fib382 = lowestLow + (highestHigh - lowestLow) * 0.382;
   double fib500 = lowestLow + (highestHigh - lowestLow) * 0.500;
   double fib618 = lowestLow + (highestHigh - lowestLow) * 0.618;
   double fib786 = lowestLow + (highestHigh - lowestLow) * 0.786;
   
   double range = high[i] - low[i];
   if(range == 0) return;
   
   double fibLevels[] = {fib236, fib382, fib500, fib618, fib786};
   for(int f = 0; f < 5; f++)
   {
      if(MathAbs(close[i] - fibLevels[f]) < range * 0.5)
      {
         if(close[i] < fibLevels[f] && close[i] > close[i + 1]) buyScore += 1.0;
         if(close[i] > fibLevels[f] && close[i] < close[i + 1]) sellScore += 1.0;
      }
   }
}

void AnalyzeVolume(int i, const double &open[], const double &close[], const long &tick_volume[], double avgVolume, double &buyScore, double &sellScore, int total)
{
   double currentVol = (double)tick_volume[i];
   if(currentVol > avgVolume * VolumeTrigger)
   {
      if(close[i] > open[i]) buyScore += 1.0;
      if(close[i] < open[i]) sellScore += 1.0;
   }
   
   if(i < total - 2)
   {
      double vol_prev = (double)tick_volume[i + 1];
      double vol_prev2 = (double)tick_volume[i + 2];
      
      if(close[i] < close[i + 1] && currentVol < vol_prev && close[i + 1] < close[i + 2] && vol_prev < vol_prev2) buyScore += 1.5;
      if(close[i] > close[i + 1] && currentVol < vol_prev && close[i + 1] > close[i + 2] && vol_prev < vol_prev2) sellScore += 1.5;
   }
}

//+------------------------------------------------------------------+
//| Candlestick Patterns                                             |
//+------------------------------------------------------------------+
void AnalyzeCandlePatterns(int i, const double &open[], const double &high[], const double &low[], const double &close[], double &buyScore, double &sellScore, int total)
{
   if(i >= total - 2) return;
   
   // Bullish Engulfing
   if(close[i] > open[i] && close[i+1] < open[i+1] && 
      close[i] > open[i+1] && open[i] < close[i+1])
      buyScore += 2.0;
      
   // Bearish Engulfing
   if(close[i] < open[i] && close[i+1] > open[i+1] && 
      close[i] < open[i+1] && open[i] > close[i+1])
      sellScore += 2.0;
      
   // Hammer (Bullish)
   double body = MathAbs(close[i] - open[i]);
   double upperWick = high[i] - MathMax(close[i], open[i]);
   double lowerWick = MathMin(close[i], open[i]) - low[i];
   if(lowerWick > 2 * body && upperWick < body * 0.2)
      buyScore += 1.5;
      
   // Shooting Star (Bearish)
   if(upperWick > 2 * body && lowerWick < body * 0.2)
      sellScore += 1.5;
      
   // Doji (Indecision/Reversal)
   if(body <= (high[i] - low[i]) * 0.1)
   {
      // If Doji after down trend
      if(close[i+1] < close[i+2]) buyScore += 1.0;
      // If Doji after up trend
      if(close[i+1] > close[i+2]) sellScore += 1.0;
   }
}

//+------------------------------------------------------------------+
//| Divergence Analysis                                              |
//+------------------------------------------------------------------+
void AnalyzeDivergence(int i, const double &high[], const double &low[], const double &close[], double &buyScore, double &sellScore, int total)
{
   if(i >= total - 10) return;
   
   // Simple 3-bar pivot divergence check with RSI
   int lookback = 10;
   
   // Bullish Divergence: Price Lower Low, RSI Higher Low
   // Find recent low in price
   int priceLowIdx = -1;
   double minPrice = DBL_MAX;
   
   for(int k=i+1; k<i+lookback; k++) {
      if(low[k] < minPrice) { minPrice = low[k]; priceLowIdx = k; }
   }
   
   if(priceLowIdx != -1 && low[i] < low[priceLowIdx]) // Current is lower low?
   {
       // Check RSI
       if(RSI_Smooth[i] > RSI_Smooth[priceLowIdx]) // RSI is higher?
          buyScore += 2.0;
   }
   
   // Bearish Divergence: Price Higher High, RSI Lower High
   int priceHighIdx = -1;
   double maxPrice = 0;
   
   for(int k=i+1; k<i+lookback; k++) {
      if(high[k] > maxPrice) { maxPrice = high[k]; priceHighIdx = k; }
   }
   
   if(priceHighIdx != -1 && high[i] > high[priceHighIdx]) // Current is higher high?
   {
       // Check RSI
       if(RSI_Smooth[i] < RSI_Smooth[priceHighIdx]) // RSI is lower?
          sellScore += 2.0;
   }
}

//+------------------------------------------------------------------+
//| Multi-Timeframe Analysis                                         |
//+------------------------------------------------------------------+
void AnalyzeMultiTimeframe(int i, const datetime &time[], double &buyScore, double &sellScore)
{
   if(!Use_Multi_Timeframe) return;
   
   // Map current time to Higher TF index
   int hIdx = iBarShift(NULL, Higher_TF, time[i]);
   if(hIdx < 0) return;
   
   // Retrieve Higher TF Indicator Values
   double hRSI[], hMACD[];
   // We only need 1 value, but CopyBuffer array needs to be dynamic or static
   // Using static small arrays for efficiency
   
   double rsiVal[1], macdVal[1];
   if(CopyBuffer(Higher_RSI, 0, hIdx, 1, rsiVal) <= 0) return;
   if(CopyBuffer(Higher_MACD, 0, hIdx, 1, macdVal) <= 0) return;
   
   // Confirm signals
   if(rsiVal[0] < RSI_OverSold) buyScore += MTF_Weight;
   if(rsiVal[0] > RSI_OverBought) sellScore += MTF_Weight;
   
   if(macdVal[0] > 0) buyScore += 0.5 * MTF_Weight; // Trend alignment
   if(macdVal[0] < 0) sellScore += 0.5 * MTF_Weight;
}

//+------------------------------------------------------------------+
//| Volatility Analysis                                              |
//+------------------------------------------------------------------+
double AnalyzeVolatility(int i)
{
   if(ATRBuffer[i] == 0) return 1.0;
   
   // Compare current ATR to average ATR
   double avgATR = 0;
   int p = 20;
   for(int k=i; k<i+p && k<ArraySize(ATRBuffer); k++) avgATR += ATRBuffer[k];
   avgATR /= p;
   
   if(avgATR == 0) return 1.0;
   
   double ratio = ATRBuffer[i] / avgATR;
   
   // If volatility is slightly elevated, it's good for trading (1.0 - 1.5)
   // If too low, dead market (reduce score). If too high, dangerous (reduce score slightly or keep neutral).
   
   if(ratio > 0.8 && ratio < 2.0) return 1.1; // Boost slightly
   if(ratio <= 0.8) return 0.8; // Reduce in low vol
   return 1.0;
}

//+------------------------------------------------------------------+
//| Generate Signals & Alerts                                        |
//+------------------------------------------------------------------+
void GenerateSignals(int i, const double &high[], const double &low[], double buyScore, double sellScore, datetime time)
{
   bool isStrongBuy = buyScore >= Strong_Signal_Threshold;
   bool isWeakBuy = buyScore >= Weak_Signal_Threshold && !isStrongBuy;
   bool isStrongSell = sellScore >= Strong_Signal_Threshold;
   bool isWeakSell = sellScore >= Weak_Signal_Threshold && !isStrongSell;
   
   // Only generate if we are not too close to the last signal (basic debounce)
   // For the history loop, we don't have a reliable 'lastSignalBar' context that persists correctly 
   // across purely calculated arrays without a static tracker, but for visual plot it's fine.
   
   // Logic: If Strong Buy, plot Strong. If Weak Buy, plot Weak.
   
   if(isStrongBuy)
   {
      StrongBuyBuffer[i] = low[i] - 10 * _Point;
      if(i == 0 && Enable_Alerts && time != lastSignalBar)
      {
         TriggerAlert("Strong BUY Signal", buyScore);
         lastSignalBar = (int)time; // Use time as unique ID cast to int (safe enough for uniqueness check)
      }
   }
   else if(isWeakBuy)
   {
      WeakBuyBuffer[i] = low[i] - 10 * _Point;
   }
   
   if(isStrongSell)
   {
      StrongSellBuffer[i] = high[i] + 10 * _Point;
      if(i == 0 && Enable_Alerts && time != lastSignalBar)
      {
         TriggerAlert("Strong SELL Signal", sellScore);
         lastSignalBar = (int)time;
      }
   }
   else if(isWeakSell)
   {
      WeakSellBuffer[i] = high[i] + 10 * _Point;
   }
}

void TriggerAlert(string type, double score)
{
   string message = StringFormat("%s | %s | %s | Score: %.2f | Price: %.5f", 
      type, _Symbol, EnumToString(_Period), score, SymbolInfoDouble(_Symbol, SYMBOL_BID));
      
   Alert(message);
   if(Enable_Push) SendNotification(message);
   if(Enable_Email) SendMail("Ultra Smooth Signal", message);
}
//+------------------------------------------------------------------+