export function Callout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <blockquote className="bg-zinc-50 dark:bg-zinc-900 my-6 border-l-2 px-4 py-3">
      {children}
    </blockquote>
  );
}
