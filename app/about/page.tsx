import Link from "next/link";

export default function AboutPage() {
  return (
    <div className="space-y-16">
      <section>
        <h1 className="text-3xl mb-8">About me</h1>
        <div className="space-y-4 text-lg">
          <p>
            {`I'm Loader. This is my digital garden. I'm a software engineer and AI enthusiast. Here you found my notes, projects, and thoughts. `}
            A few links to explore:
          </p>
          <ul className="list-disc pl-6 space-y-2">
            <li>
              <Link
                href="https://nahuel-dev.pages.dev/"
                className="underline"
                target="__blank"
              >
                Blog
              </Link>
            </li>
            <li>
              <Link
                href="https://github.com/Nahuelluca20"
                target="__blank"
                className="underline"
              >
                Github
              </Link>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
}
