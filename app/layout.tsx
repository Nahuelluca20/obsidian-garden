import type React from "react";
import "./globals.css";
import { Inter } from "next/font/google";
import Sidebar from "@/components/sidebar";
import RightSidebar from "@/components/right-sidebar";
import MobileNav from "@/components/mobile-nav";
import { ThemeProvider } from "next-themes";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Digital Garden",
  description: "My personal knowledge space",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="es" suppressHydrationWarning>
      <body className={inter.className}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="flex flex-col md:flex-row min-h-screen bg-background">
            <Sidebar />
            <main className="flex-1 border-x border-border px-4 py-6 md:px-8 min-h-screen">
              <MobileNav />
              {children}
            </main>
            <RightSidebar />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
