import { BlogPosts } from "@/components/blog-posts";
import { TopicsFilter } from "@/components/topic-filter";

export default async function Home({
  searchParams,
}: Readonly<{
  searchParams: Promise<{ [key: string]: string | string[] | undefined }>;
}>) {
  const topics = [
    "advice",
    "ai",
    "programming",
    "docker",
    "web",
    "friendship",
    "habits",
    "hardware",
    "ideas",
    "learning",
    "leverage",
    "love",
    "manufacturing",
    "media",
    "teaching",
    "tools",
    "writing",
  ];

  const { topic } = await searchParams;

  return (
    <div className="space-y-16">
      <section>
        <h2 className="text-xl mb-8">Topics</h2>
        <TopicsFilter topics={topics} />
      </section>

      <section>
        <h2 className="text-xl mb-8">Writing</h2>
        <BlogPosts topic={topic as string} />
      </section>
    </div>
  );
}
