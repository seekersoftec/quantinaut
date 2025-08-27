import "@/app/globals.css";

import { AntdRegistry } from '@ant-design/nextjs-registry';
import { Inter } from "next/font/google";
import Provider from "@/components/Provider";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Quantinaut",
  description: "Quantinaut",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <Provider><AntdRegistry>{children}</AntdRegistry></Provider>
      </body>
    </html>
  );
}


