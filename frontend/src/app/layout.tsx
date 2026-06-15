import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Smart ATS — Semantic Candidate Ranking",
  description: "AI-powered candidate ranking that understands capability, not just keywords.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-50">{children}</body>
    </html>
  );
}
