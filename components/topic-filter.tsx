"use client";

import { useState } from "react";
import { Input } from "@/components/ui/input";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import { usePathname, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";

interface TopicsFilterProps {
  topics: string[];
}

export function TopicsFilter({ topics }: TopicsFilterProps) {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const topic = searchParams.get("topic");
  const [filter, setFilter] = useState("");

  const filteredTopics = topics.filter((topic) =>
    topic.toLowerCase().includes(filter.toLowerCase()),
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 max-w-xs">
        <Input
          type="search"
          placeholder="Filter topics..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="flex-grow bg-transparent border-zinc-200 dark:border-zinc-800"
        />
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
      <AnimatePresence>
        <motion.div className="flex flex-wrap gap-x-1 gap-y-2" initial={false}>
          {filteredTopics.map((topic, index) => (
            <motion.div
              key={topic}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2, delay: index * 0.05 }}
            >
              <Link
                href={`${pathname}?topic=${topic}`}
                className="hover:underline text-zinc-900 dark:text-zinc-200"
              >
                {topic}
                {index !== filteredTopics.length - 1 && (
                  <span className="text-zinc-400">,</span>
                )}
              </Link>
            </motion.div>
          ))}
        </motion.div>
      </AnimatePresence>

      {filteredTopics.length === 0 && (
        <p className="text-zinc-500 dark:text-zinc-400">No topics found</p>
      )}
    </div>
  );
}
