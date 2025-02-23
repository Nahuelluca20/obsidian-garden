"use client";

import prism from "prismjs";
import { useEffect } from "react";
import "@/styles/prism.css";

export default function MdxComponent({
  contentHtml,
}: {
  contentHtml: string | TrustedHTML;
}) {
  useEffect(() => {
    prism.highlightAll();
  });

  return (
    <div
      className="prose dark:prose-invert prose-neutral max-w-none
        prose-a:text-zinc-900 prose-a:dark:text-zinc-100 prose-a:underline
        prose-p:text-zinc-800 prose-p:dark:text-zinc-200
        prose-headings:font-normal prose-headings:text-zinc-900 prose-headings:dark:text-zinc-100"
      dangerouslySetInnerHTML={{ __html: contentHtml }}
    />
  );
}
