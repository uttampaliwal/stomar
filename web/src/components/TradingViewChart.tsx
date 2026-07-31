import { useEffect, useRef, useCallback } from 'react'
import {
  createChart,
  ColorType,
  CrosshairMode,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type Time,
  type LineWidth,
  createSeriesMarkers,
} from 'lightweight-charts'
import type { Candle } from '@/lib/api-types'

export interface TradeMarker {
  time: string
  position: 'aboveBar' | 'belowBar'
  color: string
  shape: 'arrowUp' | 'arrowDown' | 'circle'
  text: string
  size?: number
}

export interface HorizontalLine {
  price: number
  color: string
  lineWidth?: number
  lineStyle?: number
  title?: string
  axisLabelVisible?: boolean
}

interface TradingViewChartProps {
  data: Candle[]
  height?: number
  sma20?: boolean
  sma50?: boolean
  sma200?: boolean
  bollingerBands?: boolean
  volume?: boolean
  markers?: TradeMarker[]
  horizontalLines?: HorizontalLine[]
  onCrosshairMove?: (time: string, price: number) => void
  className?: string
}

function computeSMA(data: number[], period: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < period - 1) return null
    const slice = data.slice(i - period + 1, i + 1)
    return slice.reduce((a, b) => a + b, 0) / period
  })
}

function computeBollingerBands(
  data: number[],
  period: number = 20,
  stdDev: number = 2,
): { upper: (number | null)[]; middle: (number | null)[]; lower: (number | null)[] } {
  const middle = computeSMA(data, period)
  const upper: (number | null)[] = []
  const lower: (number | null)[] = []

  for (let i = 0; i < data.length; i++) {
    if (middle[i] === null) {
      upper.push(null)
      lower.push(null)
      continue
    }
    const slice = data.slice(i - period + 1, i + 1)
    const mean = middle[i]!
    const variance = slice.reduce((sum, val) => sum + (val - mean) ** 2, 0) / period
    const std = Math.sqrt(variance)
    upper.push(mean + stdDev * std)
    lower.push(mean - stdDev * std)
  }

  return { upper, middle, lower }
}

const CHART_THEME = {
  layout: {
    background: { type: ColorType.Solid as const, color: 'transparent' },
    textColor: '#94a3b8',
    fontSize: 11,
    fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
  },
  grid: {
    vertLines: { color: 'rgba(148, 163, 184, 0.06)' },
    horzLines: { color: 'rgba(148, 163, 184, 0.06)' },
  },
  crosshair: {
    mode: CrosshairMode.Normal,
    vertLine: { color: 'rgba(34, 211, 238, 0.3)', width: 1 as LineWidth, style: LineStyle.Dashed, labelBackgroundColor: '#0e7490' },
    horzLine: { color: 'rgba(34, 211, 238, 0.3)', width: 1 as LineWidth, style: LineStyle.Dashed, labelBackgroundColor: '#0e7490' },
  },
  rightPriceScale: {
    borderColor: 'rgba(148, 163, 184, 0.1)',
    scaleMargins: { top: 0.1, bottom: 0.25 },
  },
  timeScale: {
    borderColor: 'rgba(148, 163, 184, 0.1)',
    timeVisible: false,
    secondsVisible: false,
  },
}

export default function TradingViewChart({
  data,
  height = 400,
  sma20 = true,
  sma50 = true,
  sma200 = false,
  bollingerBands = false,
  volume = true,
  markers = [],
  horizontalLines = [],
  onCrosshairMove,
  className = '',
}: TradingViewChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candleSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)

  const handleResize = useCallback(() => {
    if (chartRef.current && containerRef.current) {
      chartRef.current.applyOptions({
        width: containerRef.current.clientWidth,
      })
    }
  }, [])

  // Create chart
  useEffect(() => {
    if (!containerRef.current) return

    const chart = createChart(containerRef.current, {
      ...CHART_THEME,
      width: containerRef.current.clientWidth,
      height,
      autoSize: true,
    })

    chartRef.current = chart

    // Crosshair callback
    if (onCrosshairMove) {
      chart.subscribeCrosshairMove((param) => {
        if (param.time && param.point) {
          const pane = chart.panes()[0]
          if (pane) {
            const series = pane.getSeries()
            if (series.length > 0) {
              const seriesData = param.seriesData.get(series[0])
              if (seriesData && 'close' in (seriesData as object)) {
                onCrosshairMove(
                  String(param.time),
                  (seriesData as CandlestickData<Time>).close as number,
                )
              }
            }
          }
        }
      })
    }

    const resizeObserver = new ResizeObserver(handleResize)
    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.remove()
      chartRef.current = null
      candleSeriesRef.current = null
    }
  }, [height, handleResize])

  // Update data and overlays - recreate chart on data change
  useEffect(() => {
    const chart = chartRef.current
    if (!chart || !data.length) return

    const candleData: CandlestickData[] = data.map((d) => ({
      time: d.date as unknown as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }))

    const pane = chart.panes()[0]

    // Add candlestick series
    const candleSeries = pane.addSeries('Candlestick' as never, {
      upColor: '#10b981',
      downColor: '#f43f5e',
      borderUpColor: '#10b981',
      borderDownColor: '#f43f5e',
      wickUpColor: '#10b981',
      wickDownColor: '#f43f5e',
    })
    candleSeries.setData(candleData as never)
    candleSeriesRef.current = candleSeries as unknown as ISeriesApi<'Candlestick'>

    // Volume histogram
    if (volume) {
      const volumeData = data.map((d) => ({
        time: d.date as unknown as Time,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)',
      }))
      const volumeSeries = pane.addSeries('Histogram' as never, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.8, bottom: 0 },
      })
      volumeSeries.setData(volumeData as never)
    }

    // SMA overlays
    const closes = data.map((d) => d.close)

    if (sma20) {
      const sma = computeSMA(closes, 20)
      const smaData = sma
        .map((v, i) => (v !== null ? { time: data[i].date as unknown as Time, value: v } : null))
        .filter((d): d is { time: Time; value: number } => d !== null)
      const smaSeries = pane.addSeries('Line' as never, {
        color: '#8b5cf6',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      })
      smaSeries.setData(smaData as never)
    }

    if (sma50) {
      const sma = computeSMA(closes, 50)
      const smaData = sma
        .map((v, i) => (v !== null ? { time: data[i].date as unknown as Time, value: v } : null))
        .filter((d): d is { time: Time; value: number } => d !== null)
      const smaSeries = pane.addSeries('Line' as never, {
        color: '#f59e0b',
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      })
      smaSeries.setData(smaData as never)
    }

    if (sma200) {
      const sma = computeSMA(closes, 200)
      const smaData = sma
        .map((v, i) => (v !== null ? { time: data[i].date as unknown as Time, value: v } : null))
        .filter((d): d is { time: Time; value: number } => d !== null)
      const smaSeries = pane.addSeries('Line' as never, {
        color: '#f97316',
        lineWidth: 2,
        lineStyle: LineStyle.LargeDashed,
        lastValueVisible: false,
        priceLineVisible: false,
        crosshairMarkerVisible: false,
      })
      smaSeries.setData(smaData as never)
    }

    // Bollinger Bands
    if (bollingerBands) {
      const bands = computeBollingerBands(closes, 20, 2)
      const addBandLine = (values: (number | null)[], color: string, style: LineStyle) => {
        const lineData = values
          .map((v, i) => (v !== null ? { time: data[i].date as unknown as Time, value: v } : null))
          .filter((d): d is { time: Time; value: number } => d !== null)
        const series = pane.addSeries('Line' as never, {
          color,
          lineWidth: 1,
          lineStyle: style,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        })
        series.setData(lineData as never)
      }
      addBandLine(bands.upper, 'rgba(34, 211, 238, 0.4)', LineStyle.Dotted)
      addBandLine(bands.middle, 'rgba(34, 211, 238, 0.2)', LineStyle.Dotted)
      addBandLine(bands.lower, 'rgba(34, 211, 238, 0.4)', LineStyle.Dotted)
    }

    // Trade markers
    if (markers.length > 0) {
      createSeriesMarkers(candleSeries, markers.map((m) => ({
        time: m.time as unknown as Time,
        position: m.position,
        color: m.color,
        shape: m.shape,
        text: m.text,
        size: m.size ?? 1,
      })))
    }

    // Horizontal lines (stop-loss, take-profit)
    if (horizontalLines.length > 0) {
      for (const line of horizontalLines) {
        candleSeries.createPriceLine({
          price: line.price,
          color: line.color,
          lineWidth: (line.lineWidth ?? 1) as LineWidth,
          lineStyle: (line.lineStyle ?? LineStyle.Dashed) as LineStyle,
          axisLabelVisible: line.axisLabelVisible ?? true,
          title: line.title ?? '',
        })
      }
    }

    chart.timeScale().fitContent()
  }, [data, sma20, sma50, sma200, bollingerBands, volume, markers, horizontalLines])

  return (
    <div
      ref={containerRef}
      className={`rounded-lg overflow-hidden ${className}`}
      style={{ minHeight: height }}
    />
  )
}
