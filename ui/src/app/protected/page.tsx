// src/pages/Protected.tsx
import React from 'react';
import LogoutButton from '@/components/login_buttons/LogoutButton';
import { auth } from '@/auth';
import ChartWrapper from '@/components/ChartWrapper';

export default async function Protected() {
  const session = await auth();

  const chartData = [
    { time: '2019-04-11', value: 80.01 },
    { time: '2019-04-12', value: 96.63 },
    { time: '2019-04-13', value: 76.64 },
    { time: '2019-04-14', value: 81.89 },
    { time: '2019-04-15', value: 74.43 },
    // More points...
  ];

  return (
    <main className="max-w-2xl min-h-screen flex flex-col items-center mx-auto">
      <div className="w-full flex justify-between my-10">
        <h1 className="text-2xl font-bold">Protected Page</h1>
        <LogoutButton />
      </div>
      <pre className="w-full bg-gray-200 p-4 rounded break-words whitespace-pre-wrap">
        {JSON.stringify(session, null, 2)}
      </pre>
      <div className="my-6">
        <ChartWrapper data={chartData} />
      </div>
    </main>
  );
}
