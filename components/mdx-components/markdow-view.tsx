import type { RenderableTreeNodes } from "@markdoc/markdoc";

import markdoc from "@markdoc/markdoc";
import * as React from "react";
import prism from "prismjs";
import { Fence } from "./fence";

type Props = {
  content: RenderableTreeNodes;
};

const { renderers } = markdoc;

function Callout({ children }: { children: React.ReactNode }) {
  return <div className="callout">{children}</div>;
}

export function MarkdownView({ content }: Props) {
  return (
    <>
      {renderers.react(content, React, {
        components: { Fence, Callout },
      })}
    </>
  );
}
