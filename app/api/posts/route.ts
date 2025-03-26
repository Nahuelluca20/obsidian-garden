import { NextResponse } from 'next/server';
import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { format, parseISO } from "date-fns";

export async function GET() {
  const postsDirectory = path.join(process.cwd(), "content");

  function getAllMarkdownFiles(dir: string): string[] {
    let results: string[] = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        results = results.concat(getAllMarkdownFiles(fullPath));
      } else if (
        entry.isFile() &&
        [".md", ".mdx"].includes(path.extname(entry.name))
      ) {
        results.push(fullPath);
      }
    }
    return results;
  }

  const allFiles = getAllMarkdownFiles(postsDirectory);

  const posts = allFiles.map((filePath) => {
    const fileContents = fs.readFileSync(filePath, "utf8");
    const { data, content } = matter(fileContents);

    const relativePath = path.relative(postsDirectory, filePath);
    const segments = relativePath.split(path.sep);
    segments[segments.length - 1] = segments[segments.length - 1].replace(
      /\.mdx?$/,
      ""
    );

    const date = data.publishedAt
      ? format(parseISO(data.publishedAt), "MMMM d, yyyy")
      : "No date";
    const excerpt = content.substring(0, 150) + "...";

    return {
      slug: segments,
      title: data.title || "Untitled",
      date,
      excerpt,
      topics: data.topics || [],
      content: content.split("\n\n").filter(Boolean),
      titleStyle:
        "text-xl font-serif mb-2 hover:text-gray-800 dark:hover:text-gray-200 transition-colors",
    };
  });

  // Sort posts by date (newest first)
  posts.sort((a, b) => {
    if (!a.date || !b.date) return 0;
    return new Date(b.date).getTime() - new Date(a.date).getTime();
  });

  return NextResponse.json(posts);
}