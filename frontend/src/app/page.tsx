"use client";

import React, { useState, useEffect } from "react";
import BeautifulChart from "../components/BeautifulChart";
import { Activity, TrendingUp, TrendingDown, DollarSign, Droplet, Coins, Bitcoin, AlertCircle, Info, Calendar } from "lucide-react";

// The assets we are tracking
const ASSETS = [
  { id: "USD", name: "US Dollar Index", icon: DollarSign, color: "text-green-400", bg: "bg-green-400/10" },
  { id: "OIL", name: "Crude Oil (WTI)", icon: Droplet, color: "text-blue-400", bg: "bg-blue-400/10" },
  { id: "GOLD", name: "Gold", icon: Coins, color: "text-yellow-400", bg: "bg-yellow-400/10" },
  { id: "BTC", name: "Bitcoin", icon: Bitcoin, color: "text-orange-400", bg: "bg-orange-400/10" }
];

export default function Dashboard() {
  const [activeAsset, setActiveAsset] = useState("USD");
  const [forecastData, setForecastData] = useState<any>(null);
  const [calendarEvents, setCalendarEvents] = useState<any>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [backendUrl, setBackendUrl] = useState("");
  const [isEditingUrl, setIsEditingUrl] = useState(false);
  const [isInitialized, setIsInitialized] = useState(false);

  useEffect(() => {
    // Rely on Next.js server-side API proxy to hide IP
    localStorage.setItem("macro_backend_url", "");
    setBackendUrl("");
    setIsEditingUrl(false);
    setIsInitialized(true);
  }, []);

  const fetchData = async () => {
    // If backendUrl is set via UI, use it. Otherwise, use local Vercel /api proxy.
    const urlPrefix = backendUrl ? backendUrl : "";
    try {
      setLoading(true);
      setError(null);
      
      const [resForecast, resCalendar] = await Promise.all([
        fetch(`${urlPrefix}/api/forecast`),
        fetch(`${urlPrefix}/api/calendar`)
      ]);
      
      if (!resForecast.ok) {
        throw new Error(`Forecast Error: ${resForecast.status}`);
      }
      if (!resCalendar.ok) {
        throw new Error(`Calendar Error: ${resCalendar.status}`);
      }
      
      const jsonForecast = await resForecast.json();
      const jsonCalendar = await resCalendar.json();
      
      setForecastData(jsonForecast.data);
      setLastUpdated(jsonForecast.last_updated);
      setCalendarEvents(jsonCalendar.events);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isInitialized && (backendUrl !== null) && !isEditingUrl) {
      fetchData();
    }
  }, [backendUrl, isEditingUrl, isInitialized]);

  const saveUrl = () => {
    let finalUrl = backendUrl.trim();
    if (finalUrl && !finalUrl.startsWith("http")) {
      finalUrl = "https://" + finalUrl;
      setBackendUrl(finalUrl);
    }
    localStorage.setItem("macro_backend_url", finalUrl);
    setIsEditingUrl(false);
  };

  const currentData = forecastData ? forecastData[activeAsset] : null;

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-gray-100 font-sans p-6 md:p-12">
      {/* Backend Connection Banner */}
      {isEditingUrl ? (
        <div className="bg-indigo-900/40 border border-indigo-500/30 p-4 rounded-xl mb-8 flex flex-col md:flex-row items-center gap-4">
          <Info className="text-indigo-400" size={24} />
          <div className="flex-1 text-sm text-indigo-200">
            <strong>Colab Backend URL:</strong> Paste the new loca.lt URL from your Google Colab notebook here:
          </div>
          <div className="flex w-full md:w-auto gap-2">
            <input 
              type="text" 
              value={backendUrl} 
              onChange={(e) => setBackendUrl(e.target.value)}
              className="bg-gray-900 border border-gray-700 text-white rounded-lg px-4 py-2 w-full md:w-80 text-sm focus:outline-none focus:border-indigo-500"
              placeholder="https://your-url.loca.lt"
            />
            <button onClick={saveUrl} className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors">
              Connect
            </button>
          </div>
        </div>
      ) : (
        <div className="flex justify-end mb-4">
          <button onClick={() => setIsEditingUrl(true)} className="text-xs text-gray-500 hover:text-indigo-400 flex items-center gap-1 transition-colors">
            <Activity size={12} /> Connected to: {backendUrl ? backendUrl.replace("https://", "") : "Institutional Server"} (Click to change)
          </button>
        </div>
      )}

      {/* Header */}
      <header className="mb-12 flex flex-col md:flex-row md:items-center justify-between gap-6 border-b border-gray-800 pb-6">
        <div>
          <h1 className="text-3xl md:text-4xl font-light tracking-tight flex items-center gap-3">
            <Activity className="text-indigo-500" size={32} />
            Institutional <span className="font-semibold text-white">Macro Forecaster</span>
          </h1>
          <p className="text-gray-400 mt-3 text-sm md:text-base max-w-4xl leading-relaxed">
            An advanced multi-modal machine learning architecture combining gradient-boosted decision trees (XGBoost) and transformer-based Natural Language Processing (FinBERT) to dynamically predict cross-asset directional probabilities. The system ingests real-time macroeconomic indicators, sovereign yield curves, and FOMC sentiment analysis via a high-performance FastAPI backend, delivering sub-second probabilistic forecasting to a Next.js edge-rendered dashboard.
          </p>
        </div>
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-sm text-gray-400 bg-gray-900/50 px-4 py-2 rounded-full border border-gray-800 transition-colors">
            {loading ? (
              <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse"></div>
            ) : error ? (
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
            ) : (
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            )}
            {loading ? "Warming up Models..." : error ? "Ensemble Offline" : "Ensemble Models Online"}
          </div>
          <div className="flex items-center gap-2 text-sm text-gray-400 bg-gray-900/50 px-4 py-2 rounded-full border border-gray-800 transition-colors">
            {loading ? (
              <div className="w-2 h-2 rounded-full bg-yellow-500 animate-pulse"></div>
            ) : error ? (
              <div className="w-2 h-2 rounded-full bg-red-500"></div>
            ) : (
              <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
            )}
            {loading ? "Scanning Sentiment..." : error ? "Radar Disconnected" : "Sentiment Radar Online"}
          </div>
          {lastUpdated && !loading && !error && (
            <div className="flex items-center gap-2 text-xs text-gray-400 bg-gray-900/50 px-4 py-2 rounded-full border border-gray-800 transition-colors">
              <Activity size={12} className="text-gray-500" />
              Last Updated: <span className="text-gray-300 font-mono">{lastUpdated}</span>
            </div>
          )}
        </div>
      </header>

      {error && (
        <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl mb-8 flex items-center gap-3">
          <AlertCircle size={20} />
          {error}. Ensure the Python backend is running on port 8000.
        </div>
      )}

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        
        {/* Left Sidebar: Asset Selection & Radar */}
        <div className="lg:col-span-1 space-y-8">
          
          <div className="space-y-4">
            <h2 className="text-sm font-semibold tracking-widest text-gray-500 uppercase mb-4">Select Asset</h2>
            {ASSETS.map((asset) => {
              const Icon = asset.icon;
              const isActive = activeAsset === asset.id;
              return (
                <button
                  key={asset.id}
                  onClick={() => setActiveAsset(asset.id)}
                  className={`w-full flex items-center gap-4 p-4 rounded-2xl transition-all duration-300 border ${
                    isActive 
                      ? "bg-gray-800/80 border-indigo-500/50 shadow-lg shadow-indigo-500/10" 
                      : "bg-gray-900/30 border-gray-800 hover:bg-gray-800/50"
                  }`}
                >
                  <div className={`p-3 rounded-xl ${asset.bg} ${asset.color}`}>
                    <Icon size={24} />
                  </div>
                  <div className="text-left">
                    <div className="font-medium text-gray-200">{asset.name}</div>
                    <div className="text-sm text-gray-500 font-mono">{asset.id}</div>
                  </div>
                </button>
              )
            })}
          </div>

          {/* New Macro Radar Section */}
          <div className="bg-gray-900/40 border border-gray-800 rounded-3xl p-5 relative overflow-hidden">
             <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-3xl -mr-10 -mt-10 pointer-events-none"></div>
             <h2 className="text-sm font-semibold tracking-widest text-blue-400 flex items-center gap-2 uppercase mb-6">
                <Calendar size={16} />
                Macro Event Radar
             </h2>
             
             {calendarEvents ? (
               <div className="space-y-4">
                 {calendarEvents.map((ev: any) => (
                   <div key={ev.id} className="border-l-2 border-blue-500/50 pl-3 py-1">
                     <div className="text-sm font-medium text-gray-200">{ev.event}</div>
                     <div className="text-xs text-gray-500 mt-1 flex justify-between">
                       <span>{ev.date}</span>
                       <span className="text-orange-400 font-medium">Est: {ev.consensus}</span>
                     </div>
                   </div>
                 ))}
               </div>
             ) : (
               <div className="text-sm text-gray-500">Scanning horizon...</div>
             )}
          </div>

        </div>

        {/* Right Content: Dashboard */}
        <div className="lg:col-span-3 space-y-8">
          
          {loading && !currentData ? (
            <div className="h-96 w-full flex items-center justify-center border border-gray-800 rounded-3xl bg-gray-900/20 backdrop-blur-sm">
              <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin"></div>
            </div>
          ) : currentData && (
            <>
              {/* Top Metrics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                
                {/* Current Price */}
                <div className="bg-gradient-to-br from-gray-900 to-[#11111a] border border-gray-800 p-6 rounded-3xl shadow-xl">
                  <div className="text-gray-400 text-sm font-medium mb-2">Current Spot Price</div>
                  <div className="text-3xl lg:text-4xl font-light text-white font-mono">
                    ${currentData.current_price.toFixed(2)}
                  </div>
                </div>

                {/* Forecast Direction */}
                <div className="bg-gradient-to-br from-gray-900 to-[#11111a] border border-gray-800 p-6 rounded-3xl shadow-xl relative overflow-hidden">
                  <div className="text-gray-400 text-sm font-medium mb-2">7-Day Ensemble Edge</div>
                  <div className="flex items-end gap-3">
                    <div className={`text-3xl lg:text-4xl font-light capitalize ${
                      currentData.forecast_direction === "bullish" ? "text-green-400" :
                      currentData.forecast_direction === "bearish" ? "text-red-400" : "text-gray-400"
                    }`}>
                      {currentData.forecast_direction}
                    </div>
                    {currentData.forecast_direction === "bullish" && <TrendingUp className="text-green-400 mb-1" size={24} />}
                    {currentData.forecast_direction === "bearish" && <TrendingDown className="text-red-400 mb-1" size={24} />}
                  </div>
                  <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 rounded-full blur-3xl -mr-10 -mt-10 pointer-events-none"></div>
                </div>

                {/* Probability */}
                <div className="bg-gradient-to-br from-gray-900 to-[#11111a] border border-gray-800 p-6 rounded-3xl shadow-xl">
                  <div className="text-gray-400 text-sm font-medium mb-2">Directional Probability</div>
                  <div className="flex items-center gap-4">
                    <div className="text-3xl lg:text-4xl font-light text-white font-mono">
                      {(currentData.probability * 100).toFixed(0)}%
                    </div>
                    <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${
                          currentData.probability > 0.6 ? "bg-indigo-500" : 
                          currentData.probability > 0.5 ? "bg-yellow-500" : "bg-red-500"
                        }`}
                        style={{ width: `${currentData.probability * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

                {/* NEW: Institutional Alignment */}
                <div className="bg-gradient-to-br from-gray-900 to-[#11111a] border border-gray-800 p-6 rounded-3xl shadow-xl">
                  <div className="text-gray-400 text-sm font-medium mb-2">Institutional Alignment</div>
                  <div className="flex items-center gap-4">
                    <div className="text-3xl lg:text-4xl font-light text-white font-mono">
                      {currentData.alignment_score}%
                    </div>
                    <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${
                          currentData.alignment_score > 90 ? "bg-blue-500" : 
                          currentData.alignment_score > 75 ? "bg-yellow-500" : "bg-red-500"
                        }`}
                        style={{ width: `${currentData.alignment_score}%` }}
                      ></div>
                    </div>
                  </div>
                </div>

              </div>

              {/* Chart Section */}
              <div className="bg-gradient-to-b from-[#15151e] to-gray-900 border border-gray-800 p-6 md:p-8 rounded-3xl shadow-2xl relative">
                <div className="flex items-center justify-between mb-8">
                  <div>
                    <h3 className="text-xl font-medium text-white">Probabilistic Forecast</h3>
                    <p className="text-gray-400 text-sm mt-1 flex items-center gap-2">
                      <Info size={14} /> Shaded area indicates the 80% confidence interval based on historical model residuals.
                    </p>
                  </div>
                  <div className="px-4 py-1.5 bg-indigo-500/10 text-indigo-400 text-sm rounded-full font-medium border border-indigo-500/20">
                    {activeAsset}
                  </div>
                </div>
                
                {/* The Recharts Component */}
                <BeautifulChart assetData={currentData} assetName={activeAsset} />
              </div>

              {/* Explanation Engine Section */}
              <div className="bg-[#11111a] border border-gray-800 p-6 md:p-8 rounded-3xl">
                <h3 className="text-lg font-medium text-white mb-6">Explanation Engine (Why?)</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {currentData.reasoning.map((reason: any, idx: number) => (
                    <div key={idx} className="bg-gray-900/50 p-4 rounded-2xl border border-gray-800/50 flex items-start gap-4">
                      <div className="p-2 bg-gray-800 rounded-lg text-indigo-400">
                        <Activity size={18} />
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-200">{reason.feature}</div>
                        <div className="text-sm text-gray-500 mt-1">{reason.impact}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 pt-8 border-t border-gray-800 flex flex-col md:flex-row items-center justify-between gap-4 text-xs text-gray-600">
        <div className="flex items-center gap-2">
          <Activity size={14} className="text-gray-700" />
          <span>Macro Forecaster Engine</span>
        </div>
        <div>
          &copy; {new Date().getFullYear()} Wathsala Nitthawela. All rights reserved.
        </div>
      </footer>
    </div>
  );
}
