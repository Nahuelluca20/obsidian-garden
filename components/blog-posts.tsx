import { getBlogPosts, getTags } from "@/utlis/blog-utils";
import { CalendarIcon } from "lucide-react";
import Link from "next/link";
import { Button } from "./ui/button";

export function BlogPosts({ topic }: Readonly<{ topic: string }>) {
  const allBlogs = getBlogPosts();

  const sortedBlogs = allBlogs.toSorted(
    (a, b) =>
      new Date(b.metadata.publishedAt).getTime() -
      new Date(a.metadata.publishedAt).getTime()
  );

  const filteredBlogs = topic
    ? sortedBlogs.filter((post) => `- ${topic}` in post.metadata)
    : sortedBlogs;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 border-t border-black dark:border-[#e8e6dd] pt-8">
        {filteredBlogs.map((post, index, arr) => (
          <div
            key={post.slug}
            className={`${
              index < arr.length - 1 && (index + 1) % 4 !== 0
                ? "border-r border-black dark:border-[#e8e6dd] pr-6"
                : ""
            }`}
          >
            <Link href={`/post/${post.folderPath}/${post.slug}`}>
              <h3 className="text-2xl font-serif mb-2 hover:text-gray-800 dark:hover:text-gray-200 transition-colors">
                {post.metadata.title}
              </h3>
            </Link>
            <div className="flex items-center text-xs text-muted-foreground mb-2">
              <CalendarIcon className="mr-1 h-3 w-3" />
              <span>{post.metadata.publishedAt}</span>
            </div>
            <div className="flex flex-wrap gap-1 mb-3">
              {getTags(post).map((topicBlog: string) => (
                <button
                  key={topicBlog}
                  className={`text-xs px-2 py-0.5 rounded-sm transition-colors ${
                    topic?.includes(topicBlog)
                      ? "bg-black text-white dark:bg-white dark:text-black"
                      : "bg-transparent text-black dark:text-white border border-black dark:border-white hover:bg-black/10 dark:hover:bg-white/10"
                  }`}
                >
                  {topicBlog}
                </button>
              ))}
            </div>
            <p className="text-lg leading-relaxed mb-4">
              {post.metadata.summary}
            </p>
            <div className="flex justify-end">
              <Button variant="link" className="text-xs p-0" asChild>
                <Link href={`/post/${post.folderPath}/${post.slug}`}>
                  Read more
                </Link>
              </Button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
