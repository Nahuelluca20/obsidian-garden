export default function Home() {
  return (
    <div className="max-w-3xl mx-auto">
      <header className="flex items-center gap-4 mb-8">
        <div>
          <h1 className="text-xl md:text-2xl font-bold">🚀 loadertsx</h1>
          <p className="text-xs md:text-sm text-muted-foreground">
            Jan 01, 2025 · 1 min read
          </p>
        </div>
      </header>

      <article className="prose dark:prose-invert max-w-none">
        <p className="text-base md:text-lg mb-6">
          Hello, I am Loader. Frontend Engineer passionate about creating
          products. I'm also an artificial intelligence enthusiast and I'm
          learning about it. You can find what I'm learning about it on this
          website and on my{" "}
          <a href="#" className="font-medium no-underline hover:underline">
            blog
          </a>
          .
        </p>

        <p className="text-sm md:text-base text-muted-foreground mb-6">
          In this space I will write about things I learn and want to share.
        </p>

        <h2 className="text-lg md:text-xl font-semibold mt-8 mb-4">
          As a brief overview:
        </h2>

        <ul className="space-y-2 list-none pl-0">
          {[
            "Books I Read",
            "Docker Helps",
            "Retrieval-Augmented Generation",
          ].map((item) => (
            <li
              key={item}
              className="flex items-center gap-2 text-sm md:text-base"
            >
              <span className="w-1 h-1 bg-foreground rounded-full"></span>
              {item}
            </li>
          ))}
        </ul>
      </article>

      <footer className="mt-16 pt-8 border-t border-border text-xs md:text-sm text-muted-foreground">
        <p className="mb-2">Created with Quartz v4.4.0 © 2025</p>
        <a
          href="https://github.com"
          className="text-foreground hover:underline"
        >
          GitHub
        </a>
      </footer>
    </div>
  );
}
