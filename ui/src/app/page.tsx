// app/page.tsx
import { Layout, Menu } from 'antd';
import 'antd/dist/antd.css';
import Image from "next/image";
import { redirect } from "next/navigation";

import { auth } from "@/auth";
import LoginButton from "@/components/login_buttons/LoginButton";
import LogoutButton from "@/components/login_buttons/LogoutButton";

export default async function Home() {
  const session = await auth();

  if (!session) {
    redirect("/auth/signin");
  }
  // if (session) {
  //   redirect("/");
  // } else {
  //   redirect("/auth/signin");
  // }

  return (
    <main className="flex min-h-screen flex-col items-center justify-around p-8 lg:p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-end font-mono text-sm lg:flex">
        <div className="flex h-48 w-full items-center justify-between bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:h-auto lg:bg-none">

            <Image
              src="/images/logo.png"
              alt="Quantinaut Logo"
              // className="dark:invert"
              width={100}
              height={24}
              priority
            />
          <div className="px-2"></div>
    
        </div>
      </div>

      <h1 className="text-3xl font-bold text-center mb-8 lg:mb-0">
        Quantinaut
      </h1>

      <div className="relative mb-8 flex place-items-center lg:my-0">
        <Image
          className="relative dark:drop-shadow-[0_0_0.3rem_#ffffff70] dark:invert"
          src="/images/icons/next.svg"
          alt="Next.js Logo"
          width={180}
          height={37}
          priority
        />
      </div>
      <div className="grid grid-cols-1 gap-y-4 md:grid-cols-2 md:gap-x-6">
        {session ? <LogoutButton /> : <LoginButton auth={session} />}
        <a href="/protected">Goto Protected Page</a>
      </div>
    </main>
  );
}


// 'use client';
// import React, { useEffect } from 'react';
// import Chart from '@/pages/chart'; // reuse your existing component
// import { useConfigsStore, useDataFrameStore } from '@/store';
// import { PubSub } from '@/events/pubsub';
// import * as Notifications from '@/utils/notifications';
// import { redirect } from 'next/navigation';
// import { auth } from '@/auth';

// export default async function Home() {
//   const session = await auth();

//   if (!session) {
//     redirect("/auth/signin");
//   }

//   useEffect(() => {
//     async function init() {
//       const pubsub = await PubSub.getInstance();
//       await useConfigsStore.getState().setDefault();
//       await useDataFrameStore.getState().restore();
//       Notifications.mount(pubsub);
//       pubsub.subscribe('DataFrame', (payload) => {
//         useDataFrameStore.getState().add([payload]);
//       });
//     }
//     init();
//   }, []);

//   return <Chart />;
// }