import Link from "next/link";

export default function AboutPage() {
  return (
    <div className="space-y-16">
      <section>
        <h1 className="text-3xl mb-8">About me</h1>
        <div className="space-y-4 text-lg">
          <p>
            I'm Your Name. You may also know me as kepano, currently the CEO of{" "}
            <Link href="https://obsidian.md" className="underline">
              Obsidian
            </Link>
            . Previously I co-founded{" "}
            <Link href="#" className="underline">
              Lumi
            </Link>{" "}
            and{" "}
            <Link href="#" className="underline">
              Inkodye
            </Link>
            . A few links to explore:
          </p>
          <ul className="list-disc pl-6 space-y-2">
            <li>
              <Link href="/writing" className="underline">
                Writing
              </Link>
            </li>
            <li>
              <Link href="/projects" className="underline">
                Projects
              </Link>
            </li>
            <li>
              <Link href="/recipes" className="underline">
                Recipes
              </Link>
            </li>
            <li>
              <Link href="/now" className="underline">
                What I'm doing now
              </Link>
            </li>
          </ul>
        </div>
      </section>

      <section>
        <h2 className="text-2xl mb-8">Elsewhere</h2>
        <div className="space-y-4 text-lg">
          <p>
            You can{" "}
            <Link href="#" className="underline">
              receive updates from me via email
            </Link>{" "}
            or follow me:
          </p>
          <ul className="list-disc pl-6 space-y-2">
            <li>
              RSS feed:{" "}
              <Link href="#" className="underline">
                stephango.com/feed
              </Link>
            </li>
            <li>
              Twitter:{" "}
              <Link href="#" className="underline">
                @kepano ↗
              </Link>
            </li>
            <li>
              GitHub:{" "}
              <Link href="#" className="underline">
                @kepano ↗
              </Link>
            </li>
            <li>
              Mastodon:{" "}
              <Link href="#" className="underline">
                @kepano@mastodon.social ↗
              </Link>
            </li>
            <li>
              Bluesky:{" "}
              <Link href="#" className="underline">
                @stephango.com ↗
              </Link>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
