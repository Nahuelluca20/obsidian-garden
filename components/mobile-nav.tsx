"use client";

import { useState, useEffect } from "react";
import { Menu, X } from "lucide-react";
import { Button } from "./ui/button";
import { ThemeToggle } from "./theme-toggle";
import Link from "next/link";

export default function MobileNav() {
  const [isOpen, setIsOpen] = useState(false);

  const navigation = [
    { name: "Books", href: "/books", items: ["Books I Read"] },
    { name: "Docker", href: "/docker", items: ["Docker Helps"] },
    { name: "RAG", href: "/rag", items: ["Retrieval-Augmented Generation"] },
  ];

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }

    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  return (
    <div className="md:hidden">
      <div className="flex justify-between items-center mb-6">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsOpen(!isOpen)}
          aria-label="Toggle menu"
        >
          <Menu />
        </Button>
        <ThemeToggle />
      </div>
      {isOpen && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50">
          <div className="fixed inset-y-0 left-0 w-full max-w-xs bg-background p-6 shadow-lg transition-transform duration-300 ease-in-out transform translate-x-0">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-lg font-semibold">Menu</h2>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setIsOpen(false)}
                aria-label="Close menu"
              >
                <X />
              </Button>
            </div>
            <nav className="space-y-4">
              {navigation.map((item) => (
                <div key={item.name} className="mb-2">
                  <Link
                    href={item.href}
                    className="block px-2 py-1 text-lg rounded-md hover:bg-accent"
                    onClick={() => setIsOpen(false)}
                  >
                    {item.name}
                  </Link>
                  {item.items.map((subItem) => (
                    <Link
                      key={subItem}
                      href={`${item.href}/${subItem.toLowerCase().replace(/ /g, "-")}`}
                      className="block px-4 py-1 text-sm text-muted-foreground hover:text-foreground"
                      onClick={() => setIsOpen(false)}
                    >
                      {subItem}
                    </Link>
                  ))}
                </div>
              ))}
            </nav>
          </div>
        </div>
      )}
    </div>
  );
}
