import type { Metadata } from "next";
import { Fraunces, Outfit } from "next/font/google";
import type { ReactNode } from "react";
import "./globals.css";

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

const fraunces = Fraunces({
  variable: "--font-fraunces",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CardioLens",
  description:
    "Educational viewer for coronary artery disease and LAD, LCX, and RCA stenosis estimates, with a labeled 3D heart and SHAP explanations.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html
      lang="en"
      className={`${outfit.variable} ${fraunces.variable} dark h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
