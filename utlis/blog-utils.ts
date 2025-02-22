import fs from "node:fs";
import path from "node:path";

type Metadata = {
  title: string;
  publishedAt: string;
  summary: string;
  image?: string;
};

function parseFrontmatter(fileContent: string) {
  const frontmatterRegex = /---\s*([\s\S]*?)\s*---/;
  const match = frontmatterRegex.exec(fileContent);

  // Evitamos errores si no hay frontmatter
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

  frontMatterLines.forEach((line) => {
    const [key, ...valueArr] = line.split(": ");
    let value = valueArr.join(": ").trim();
    value = value.replace(/^['"](.*)['"]$/, "$1"); // Remove quotes
    metadata[key.trim() as keyof Metadata] = value;
  });

  return { metadata: metadata as Metadata, content };
}

/**
 * Recorre recursivamente el directorio `dir` y devuelve un array
 * con la ruta absoluta de todos los archivos .md que encuentre.
 */
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

    // Extraemos el "slug" (nombre de archivo sin extensión)
    const slug = path.basename(
      absoluteFilePath,
      path.extname(absoluteFilePath),
    );

    // Obtenemos la ruta relativa al directorio raíz para reflejar subcarpetas
    // Por ejemplo, si absoluteFilePath = /User/tu-app/content/LLM/Deep dive into LLMs/part1.md
    // y rootDir = /User/tu-app/content
    // relativePath será: LLM/Deep dive into LLMs/part1.md
    const relativePath = path.relative(rootDir, absoluteFilePath);

    // De esa ruta relativa, extraemos la carpeta sin el nombre del archivo
    // Por ejemplo, path.dirname("LLM/Deep dive into LLMs/part1.md") = "LLM/Deep dive into LLMs"
    const folderPath = path.dirname(relativePath);

    return {
      metadata,
      content,
      slug,
      folderPath,
      // También puedes guardar la ruta completa si la necesitas:
      // absolutePath: absoluteFilePath,
      // relativePath,
    };
  });
}

/**
 * Función principal que se encarga de devolver todos los posts,
 * incluidos los de subcarpetas.
 */
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
