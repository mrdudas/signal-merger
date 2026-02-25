import { useEffect, useRef, useState, memo } from "react";

const FASTAPI_BASE = "http://127.0.0.1:3000";

interface WaveformData {
  x: number[];
  y_min: number[];
  y_max: number[];
  y_mean: number[];
  global_min: number;
  global_max: number;
  duration_s: number;
}

interface SignalWaveformProps {
  csvPath: string;           // local absolute path or http URL (goes to FastAPI)
  column: string;
  sampleRate?: number;
  width: number;             // scrubber pixel width
  height: number;            // scrubber pixel height
  color?: string;            // fill color (hex / css)
  points?: number;           // resolution hint (default = width / 2)
}

const cache = new Map<string, WaveformData>();

export const SignalWaveform = memo(
  ({ csvPath, column, sampleRate = 200, width, height, color = "#2dd4bf", points }: SignalWaveformProps) => {
    const [data, setData] = useState<WaveformData | null>(null);
    const [error, setError] = useState(false);
    const abortRef = useRef<AbortController | null>(null);

    useEffect(() => {
      if (!csvPath || !column) return;
      const key = `${csvPath}||${column}||${sampleRate}||${Math.round(width)}`;
      if (cache.has(key)) {
        setData(cache.get(key)!);
        return;
      }

      abortRef.current?.abort();
      abortRef.current = new AbortController();

      const numPoints = points ?? Math.max(50, Math.round(width / 2));

      fetch(`${FASTAPI_BASE}/api/signal/data`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ csv_path: csvPath, column, sample_rate: sampleRate, points: numPoints }),
        signal: abortRef.current.signal,
      })
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
        .then((d: WaveformData) => {
          cache.set(key, d);
          setData(d);
          setError(false);
        })
        .catch((e) => {
          if (e.name !== "AbortError") setError(true);
        });

      return () => abortRef.current?.abort();
    }, [csvPath, column, sampleRate, width, points]);

    if (error) {
      return (
        <div className="absolute inset-0 flex items-center justify-center opacity-40 pointer-events-none">
          <span className="text-[9px]">err</span>
        </div>
      );
    }
    if (!data) {
      return (
        <div className="absolute inset-0 flex items-center justify-center opacity-30 pointer-events-none">
          <span className="text-[9px]">…</span>
        </div>
      );
    }

    const { y_min, y_max, y_mean } = data;
    const range = (data.global_max - data.global_min) || 1;
    const pad = 4;
    const svgH = height - pad * 2;
    const svgW = width;
    const n = y_min.length;
    if (n < 2) return null;

    const toY = (v: number) =>
      pad + svgH - ((v - data.global_min) / range) * svgH;

    const step = svgW / (n - 1);

    // Build envelope polygon: top (y_max) left→right, then bottom (y_min) right→left
    const topPoints = y_max.map((v, i) => `${(i * step).toFixed(1)},${toY(v).toFixed(1)}`);
    const botPoints = [...y_min].reverse().map((v, i) =>
      `${((n - 1 - i) * step).toFixed(1)},${toY(v).toFixed(1)}`
    );
    const polygon = [...topPoints, ...botPoints].join(" ");

    // Mean line
    const meanPath = y_mean
      .map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${toY(v).toFixed(1)}`)
      .join(" ");

    return (
      <svg
        className="absolute inset-0 pointer-events-none"
        width={svgW}
        height={height}
        style={{ overflow: "hidden" }}
      >
        {/* Envelope fill */}
        <polygon
          points={polygon}
          fill={color}
          fillOpacity={0.25}
          stroke="none"
        />
        {/* Mean line */}
        <path
          d={meanPath}
          fill="none"
          stroke={color}
          strokeWidth={1}
          strokeOpacity={0.85}
        />
      </svg>
    );
  }
);

SignalWaveform.displayName = "SignalWaveform";
