import fs from "node:fs";
import path from "node:path";
import type { Root } from "postcss";
import { visit } from "unist-util-visit";

type Metadata = {
  title: string;
  publishedAt: string;
  summary: string;
  image?: string;
  tags: string;
};

function parseFrontmatter(fileContent: string) {
  const frontmatterRegex = /---\s*([\s\S]*?)\s*---/;
  const match = frontmatterRegex.exec(fileContent);

  if (!match) {
    return {
      metadata: {} as Metadata,
      content: fileContent,
    };
  }

  const frontMatterBlock = match[1];
  const content = fileContent.replace(frontmatterRegex, "").trim();
  const frontMatterLines = frontMatterBlock.trim().split("\n");
  const metadata: Partial<Metadata> = {};

  for (let i = 0; i < frontMatterLines.length; i++) {
    const line = frontMatterLines[i];
    const [key, ...valueArr] = line.split(": ");
    let value = valueArr.join(": ").trim();
    value = value.replace(/^['"](.*)['"]$/, "$1"); // Elimina las comillas
    metadata[key.trim() as keyof Metadata] = value;
  }
  return { metadata: metadata as Metadata, content };
}

function getMDXFiles(dir: string): string[] {
  let results: string[] = [];
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      // Llamada recursiva
      results = results.concat(getMDXFiles(fullPath));
    } else if (entry.isFile() && path.extname(entry.name) === ".md") {
      results.push(fullPath);
    }
  }

  return results;
}

function readMDXFile(filePath: string) {
  const rawContent = fs.readFileSync(filePath, "utf-8");
  return parseFrontmatter(rawContent);
}

function getMDXData(rootDir: string) {
  // Obtenemos todas las rutas de archivos MD dentro de rootDir
  const mdxFiles = getMDXFiles(rootDir);

  return mdxFiles.map((absoluteFilePath) => {
    const { metadata, content } = readMDXFile(absoluteFilePath);
    const slug = path.basename(
      absoluteFilePath,
      path.extname(absoluteFilePath)
    );

    const relativePath = path.relative(rootDir, absoluteFilePath);

    const folderPath = path.dirname(relativePath);

    return {
      metadata,
      content,
      slug,
      folderPath,
    };
  });
}

export function getBlogPosts() {
  const contentDir = path.join(process.cwd(), "content");
  return getMDXData(contentDir);
}

export function formatDate(date: string, includeRelative = false) {
  const currentDate = new Date();
  if (!date.includes("T")) {
    date = `${date}T00:00:00`;
  }
  const targetDate = new Date(date);

  const yearsAgo = currentDate.getFullYear() - targetDate.getFullYear();
  const monthsAgo = currentDate.getMonth() - targetDate.getMonth();
  const daysAgo = currentDate.getDate() - targetDate.getDate();

  let formattedDate = "";

  if (yearsAgo > 0) {
    formattedDate = `${yearsAgo}y ago`;
  } else if (monthsAgo > 0) {
    formattedDate = `${monthsAgo}mo ago`;
  } else if (daysAgo > 0) {
    formattedDate = `${daysAgo}d ago`;
  } else {
    formattedDate = "Today";
  }

  const fullDate = targetDate.toLocaleString("en-us", {
    month: "long",
    day: "numeric",
    year: "numeric",
  });

  if (!includeRelative) {
    return fullDate;
  }

  return `${fullDate} (${formattedDate})`;
}

export function remarkObsidianImages() {
  return (tree: Root) => {
    visit(
      tree,
      "text",
      (node: { value?: string }, index: number | undefined, parent) => {
        if (!node.value) return;

        const imageRegex = /!\[\[([^\]]+)\]\]/g;
        let match: RegExpExecArray | null;
        const newNodes = [];
        let lastIndex = 0;

        while ((match = imageRegex.exec(node.value)) !== null) {
          const start = match.index;
          const end = imageRegex.lastIndex;

          if (start > lastIndex) {
            newNodes.push({
              type: "text",
              value: node.value.slice(lastIndex, start),
            });
          }

          const imageData = match[1].trim();
          const [filePath, altTextRaw] = imageData
            .split("|")
            .map((s) => s.trim());
          const altText = altTextRaw || filePath;

          newNodes.push({
            type: "image",
            url: `/attachments/${filePath}`,
            alt: altText,
          });

          lastIndex = end;
        }

        if (lastIndex < node.value.length) {
          newNodes.push({
            type: "text",
            value: node.value.slice(lastIndex),
          });
        }
        if (newNodes.length && parent && typeof index === "number") {
          // Type assertion to handle the 'never' type error
          (parent as { children: unknown[] }).children.splice(
            index,
            1,
            ...newNodes
          );
        }
      }
    );
  };
}
