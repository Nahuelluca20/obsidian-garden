import { getBlogPosts } from "@/utlis/blog-utils";
import Link from "next/link";

export function BlogPosts({ topic }: Readonly<{ topic: string }>) {
  const allBlogs = getBlogPosts();

  const sortedBlogs = allBlogs.toSorted(
    (a, b) =>
      new Date(b.metadata.publishedAt).getTime() -
      new Date(a.metadata.publishedAt).getTime()
  );

  console.log(sortedBlogs, topic);

  const filteredBlogs = topic
    ? sortedBlogs.filter((post) => `- ${topic}` in post.metadata)
    : sortedBlogs;

  return (
    <div className="space-y-3">
      {filteredBlogs.map((post) => (
        <div key={post.slug} className="flex gap-6">
          <span className="text-zinc-400 shrink-0 w-24">
            {post.metadata.publishedAt}
          </span>
          <Link
            href={`/post/${post.folderPath}/${post.slug}`}
            className="text-zinc-900 dark:text-zinc-200 hover:underline"
          >
            {post.metadata.title}
          </Link>
        </div>
      ))}
    </div>
  );
}
