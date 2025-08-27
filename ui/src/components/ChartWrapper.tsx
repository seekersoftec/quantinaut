"use client";

// src/components/ChartWrapper.tsx
import React, { useRef, useEffect } from 'react';
import { createChart, IChartApi, LineSeries } from 'lightweight-charts';

interface ChartWrapperProps {
  data: { time: string; value: number }[];
}

const ChartWrapper: React.FC<ChartWrapperProps> = ({ data }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const chart: IChartApi = createChart(containerRef.current, {
      width: 400,
      height: 300,
    });
    const series = chart.addSeries(LineSeries);
    series.setData(data);

    return () => {
      chart.remove(); // Clean up on unmount
    };
  }, [data]);

  return <div ref={containerRef} />;
};

export default ChartWrapper;
