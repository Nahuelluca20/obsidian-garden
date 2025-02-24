"use client";
import prism from "prismjs";
import "prismjs/components/prism-python";
import { useEffect } from "react";

export function Fence({
  children,
  language,
}: {
  children: React.ReactNode;
  language: string;
}) {
  useEffect(() => {
    prism.highlightAll();
  }, []);

  return (
    <pre key={language}>
      <code className={`language-${language}`}>{children}</code>
    </pre>
  );
}
