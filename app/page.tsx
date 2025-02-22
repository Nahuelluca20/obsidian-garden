import { BlogPosts } from "@/components/blog-posts";
import Link from "next/link";

export default function Home() {
  const topics = [
    "advice",
    "ai",
    "competition",
    "defaults",
    "design",
    "empathy",
    "evergreen",
    "friendship",
    "habits",
    "hardware",
    "humanism",
    "ideas",
    "learning",
    "leverage",
    "love",
    "manufacturing",
    "media",
    "minimalism",
    "money",
    "obsidian",
    "packaging",
    "perfectionism",
    "popular",
    "rituals",
    "startups",
    "synthography",
    "teaching",
    "tools",
    "writing",
  ];

  return (
    <div className="space-y-16">
      {/* <section>
        <h2 className="text-xl mb-8">Latest</h2>
        <article>
          <h3 className="text-xl font-medium mb-2">
            <Link
              href="/posts/self-guaranteeing-promises"
              className="hover:underline"
            >
              Self-guaranteeing promises
            </Link>
          </h3>
          <p className="text-zinc-600 dark:text-zinc-400 mb-2">
            December 3, 2024 · 1 minute read
          </p>
          <p className="text-zinc-600 dark:text-zinc-400">
            A self-guaranteeing promise does not require you to trust anyone.
            You can verify it yourself.{" "}
            <Link
              href="/posts/self-guaranteeing-promises"
              className="text-zinc-900 dark:text-zinc-200 hover:underline"
            >
              Keep reading →
            </Link>
          </p>
        </article>
      </section>

      <section>
        <h2 className="text-xl mb-8">Topics</h2>
        <div className="flex flex-wrap gap-x-1 gap-y-2">
          {topics.map((topic) => (
            <Link
              key={topic}
              href={`/topics/${topic}`}
              className="hover:underline text-zinc-900 dark:text-zinc-200"
            >
              {topic}
              {topic !== topics[topics.length - 1] && (
                <span className="text-zinc-400">,</span>
              )}
            </Link>
          ))}
        </div>
      </section> */}

      <section>
        <h2 className="text-xl mb-8">Writing</h2>
        <BlogPosts />
      </section>
    </div>
  );
}
