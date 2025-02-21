export default function RightSidebar() {
  return (
    <div className="md:w-64 p-6 border-t md:border-t-0 md:border-l border-border">
      <div className="mb-8">
        <h2 className="font-semibold text-sm text-muted-foreground uppercase tracking-wider mb-4">
          Graph View
        </h2>
        <div className="aspect-square rounded-md border border-border bg-accent/10">
          {/* Aquí iría el componente del grafo */}
        </div>
      </div>

      <div>
        <h2 className="font-semibold text-sm text-muted-foreground uppercase tracking-wider mb-4">
          Backlinks
        </h2>
        <p className="text-sm text-muted-foreground">No backlinks found</p>
      </div>
    </div>
  );
}
