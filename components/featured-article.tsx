import Link from "next/link";
import { getFirstPost, getTags } from "@/utlis/blog-utils";

export function FeaturedArticle() {
  const post = getFirstPost();

  const topicKeys = getTags(post);

  return (
    <article className="mb-12">
      <Link href={`/post/${post.folderPath}/${post.slug}`}>
        <h2 className="text-5xl md:text-7xl font-black font-serif leading-none mb-8 hover:text-gray-800 dark:hover:text-gray-200 transition-colors">
          {post.metadata.title}
        </h2>
      </Link>
      <div className="flex flex-wrap gap-2 mb-4">
        {topicKeys.map((topic: string) => (
          <button
            key={topic}
            className="text-xs px-2 py-1 bg-transparent text-black dark:text-white border border-black dark:border-white hover:bg-black/10 dark:hover:bg-white/10 rounded-sm transition-colors"
          >
            {topic}
          </button>
        ))}
      </div>
      <div className="max-w-4xl">
        <p className="text-lg leading-relaxed mb-4">{post.metadata.summary}</p>
      </div>
    </article>
  );
}
