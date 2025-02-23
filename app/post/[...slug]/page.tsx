import fs from "node:fs";
import path from "node:path";
import matter from "gray-matter";
import { remark } from "remark";
import html from "remark-html";
import { remarkObsidianImages } from "@/utlis/blog-utils";
import { format, parseISO } from "date-fns";
import MdxComponent from "@/components/mdx";

export async function generateStaticParams() {
  const rootDir = path.join(process.cwd(), "content");

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

  const allFiles = getAllMarkdownFiles(rootDir);

  return allFiles.map((filePath) => {
    const relativePath = path.relative(rootDir, filePath);
    const segments = relativePath.split(path.sep);

    segments[segments.length - 1] = segments[segments.length - 1].replace(
      /\.mdx?$/,
      "",
    );
    return { slug: segments };
  });
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const decodedSlug = slug.map((segment) => decodeURIComponent(segment));
  const postPath = `${path.join(process.cwd(), "content", ...decodedSlug)}.mdx`;

  const fileContent = fs.readFileSync(postPath, "utf-8");
  const { data } = matter(fileContent);
  return {
    title: data.title || "Untitled",
    description: data.summary || "",
  };
}

export default async function Post({
  params,
}: {
  params: Promise<{ slug: string[] }>;
}) {
  const { slug } = await params;
  const decodedSlug = slug.map((segment) => decodeURIComponent(segment));
  const postPath = `${path.join(process.cwd(), "content", ...decodedSlug)}.mdx`;
  const markdownWithMeta = fs.readFileSync(postPath, "utf-8");
  const { data: frontmatter, content } = matter(markdownWithMeta);

  const processedContent = await remark()
    .use(remarkObsidianImages)
    .use(html)
    .process(content);
  const contentHtml = processedContent.toString();

  const date = parseISO(frontmatter.publishedAt);
  const formatDate = format(date, "MMMM d, yyyy");

  return (
    <>
      <article className="max-w-4xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-normal mb-2">{frontmatter.title}</h1>
          <p className="text-zinc-600 dark:text-zinc-400">{formatDate}</p>
        </header>
        <MdxComponent contentHtml={contentHtml} />
      </article>
    </>
  );
}
