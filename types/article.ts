import { Metadata } from "next";

export type Article = {
  metadata: Metadata;
  content: string;
  slug: string;
  folderPath: string;
};
