"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";

interface TopicsFilterProps {
  topics: string[];
}

export function TopicsFilter({ topics }: Readonly<TopicsFilterProps>) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const topic = searchParams.get("topic");

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 max-w-xs">
        <h2 className="text-xl">Topics</h2>
        {topic && (
          <Button
            asChild
            variant="ghost"
            size="sm"
            className="text-zinc-500 hover:text-zinc-700 dark:text-zinc-400 dark:hover:text-zinc-200"
            aria-label="Clear filter"
          >
            <Link
              href={`${pathname}?topic=`}
              className="hover:underline text-zinc-900 dark:text-zinc-200"
            >
              Clear
            </Link>
          </Button>
        )}
      </div>
      <div className="flex flex-wrap gap-2">
        {topics.map((topic) => (
          <Link
            href={`${pathname}?topic=${topic}`}
            key={topic}
            className={`text-sm px-3 py-1 rounded-sm transition-colors ${
              searchParams.toString().includes(topic)
                ? "bg-black text-white dark:bg-white dark:text-black"
                : "bg-transparent text-black dark:text-white border border-black dark:border-white hover:bg-black/10 dark:hover:bg-white/10"
            }`}
          >
            {topic}
          </Link>
        ))}
      </div>
    </div>
  );
}
