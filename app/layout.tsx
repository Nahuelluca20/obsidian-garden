import type React from "react";
import "./globals.css";
import { GeistSans } from "geist/font/sans";
import { Navigation } from "@/components/navigation";
import { ThemeProvider } from "next-themes";

const geist = GeistSans.className;

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
      <body className={`${geist} antialiased bg-[#FFFCF4] dark:bg-zinc-900`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <div className="max-w-2xl mx-auto px-6 py-12">
            <Navigation />
            {children}
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
