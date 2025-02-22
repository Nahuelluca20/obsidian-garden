import { getBlogPosts } from "@/utlis/blog-utils";
import Link from "next/link";

export function BlogPosts() {
  const allBlogs = getBlogPosts();

  return (
    <div className="space-y-3">
      {allBlogs.map((post) => (
        <div key={post.slug} className="flex gap-6">
          <span className="text-zinc-400 shrink-0 w-24">
            {post.metadata.publishedAt}
          </span>
          <Link
            href={`/posts/${post.slug}`}
            className="text-zinc-900 dark:text-zinc-200 hover:underline"
          >
            {post.metadata.title}
          </Link>
        </div>
      ))}
    </div>
  );
}
