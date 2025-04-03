import React from "react";

export function BulletListItem({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <li>
      <span className="mr-2 text-lg leading-10">•</span>
      {children}
    </li>
  );
}
