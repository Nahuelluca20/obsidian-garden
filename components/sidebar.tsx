import { ThemeToggle } from "./theme-toggle";
import { Input } from "./ui/input";
import Link from "next/link";

export default function Sidebar() {
  const navigation = [
    { name: "Books", href: "/books", items: ["Books I Read"] },
    { name: "Docker", href: "/docker", items: ["Docker Helps"] },
    { name: "RAG", href: "/rag", items: ["Retrieval-Augmented Generation"] },
  ];

  return (
    <div className="hidden md:flex md:flex-col md:w-64 p-6 border-r border-border">
      <Link href="/" className="flex items-center gap-2 text-xl font-semibold">
        🚀 loadertsx
      </Link>

      <Input
        type="search"
        placeholder="Search"
        className="w-full bg-background mt-6"
      />

      <ThemeToggle className="mt-6" />

      <nav className="mt-6 space-y-1">
        {navigation.map((item) => (
          <div key={item.name} className="mb-2">
            <Link
              href={item.href}
              className="block px-2 py-1 text-sm rounded-md hover:bg-accent"
            >
              {item.name}
            </Link>
            {item.items.map((subItem) => (
              <Link
                key={subItem}
                href={`${item.href}/${subItem.toLowerCase().replace(/ /g, "-")}`}
                className="block px-4 py-1 text-sm text-muted-foreground hover:text-foreground"
              >
                {subItem}
              </Link>
            ))}
          </div>
        ))}
      </nav>
    </div>
  );
}
