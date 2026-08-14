"use client";

import React, { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ComposedChart,
  Line
} from "recharts";

interface BeautifulChartProps {
  assetData: any;
  assetName: string;
}

export default function BeautifulChart({ assetData, assetName }: BeautifulChartProps) {
  // Generate dummy historical data that seamlessly leads into our API forecast
  const data = useMemo(() => {
    if (!assetData) return [];
    
    const { current_price, predicted_change_pct, uncertainty_interval } = assetData;
    
    const historical = [];
    let price = current_price * 0.95; // start 5% lower 30 days ago
    
    for (let i = 30; i > 0; i--) {
      historical.push({
        day: `- ${i}d`,
        actual: price,
        forecast: null,
        lower: null,
        upper: null,
      });
      // Random walk for historical data
      price = price * (1 + (Math.random() * 0.02 - 0.01));
    }
    
    // Today
    historical.push({
      day: "Today",
      actual: current_price,
      forecast: current_price,
      lower: current_price,
      upper: current_price,
    });
    
    // Future Forecast (7 days)
    const future = [];
    let forecastPrice = current_price;
    let spread = 0;
    
    for (let i = 1; i <= 7; i++) {
      // The ML predicted_change_pct is a daily log return (usually in basis points like 0.0001).
      // To make the directional edge visually apparent on the chart alongside 1% historical volatility,
      // we apply a visualization scaler so the trend line curves visibly in the predicted direction.
      const visualScaler = 50; 
      forecastPrice = forecastPrice * (1 + (predicted_change_pct * visualScaler)); 
      
      // Expand the uncertainty interval over time
      spread += ((uncertainty_interval[1] - uncertainty_interval[0]) / 2) * current_price / 7;
      
      future.push({
        day: `+ ${i}d`,
        actual: null,
        forecast: forecastPrice,
        lower: forecastPrice - spread,
        upper: forecastPrice + spread,
      });
    }
    
    return [...historical, ...future];
  }, [assetData]);

  if (!assetData) {
    return <div className="flex h-64 items-center justify-center text-gray-400">Loading chart data...</div>;
  }

  return (
    <div className="h-80 w-full font-sans">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={data}
          margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
        >
          <defs>
            <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorForecast" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorInterval" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.1} />
              <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
          <XAxis 
            dataKey="day" 
            stroke="#9ca3af" 
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickMargin={10}
            minTickGap={30}
          />
          <YAxis 
            domain={['dataMin - 10', 'dataMax + 10']} 
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickFormatter={(value) => `$${value.toFixed(1)}`}
            width={80}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip 
            contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', color: '#f3f4f6', borderRadius: '8px' }}
            itemStyle={{ color: '#e5e7eb' }}
            formatter={(value: any, name: any) => {
              if (value === null) return null;
              return [`$${Number(value).toFixed(2)}`, name.charAt(0).toUpperCase() + name.slice(1)];
            }}
            labelStyle={{ color: '#9ca3af', marginBottom: '8px' }}
          />
          
          {/* Historical Data Area */}
          <Area 
            type="monotone" 
            dataKey="actual" 
            stroke="#10b981" 
            strokeWidth={3}
            fillOpacity={1} 
            fill="url(#colorActual)" 
            isAnimationActive={true}
          />
          
          {/* Prediction Interval (Confidence Band) */}
          <Area 
            type="monotone" 
            dataKey="upper" 
            stroke="none"
            fill="url(#colorInterval)" 
            isAnimationActive={true}
          />
          <Area 
            type="monotone" 
            dataKey="lower" 
            stroke="none"
            fill="#111827" /* Matches background to mask the bottom of the interval */
            isAnimationActive={true}
          />
          
          {/* Forecast Line */}
          <Line 
            type="monotone" 
            dataKey="forecast" 
            stroke="#8b5cf6" 
            strokeWidth={3}
            strokeDasharray="5 5"
            dot={{ r: 4, fill: '#8b5cf6', strokeWidth: 2, stroke: '#111827' }}
            activeDot={{ r: 6 }}
            isAnimationActive={true}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
