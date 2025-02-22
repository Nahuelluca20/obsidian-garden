import Link from "next/link";
import { ThemeToggle } from "./theme-toggle";

export function Navigation() {
  return (
    <nav className="flex justify-between items-center mb-16">
      <Link
        href="/"
        className="text-xl text-emerald-700 dark:text-emerald-400 hover:underline"
      >
        Your Name
      </Link>
      <div className="flex items-center gap-6">
        <Link
          href="/about"
          className="text-zinc-600 dark:text-zinc-400 hover:underline"
        >
          About
        </Link>
        <ThemeToggle />
      </div>
    </nav>
  );
}
