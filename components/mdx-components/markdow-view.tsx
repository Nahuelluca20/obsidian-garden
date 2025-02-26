import type { RenderableTreeNodes } from "@markdoc/markdoc";

import markdoc from "@markdoc/markdoc";
import * as React from "react";
import { Fence } from "./fence";

type Props = {
  content: RenderableTreeNodes;
};

const { renderers } = markdoc;

function Callout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <div className="callout">{children}</div>;
}

export function MarkdownView({ content }: Readonly<Props>) {
  return (
    <>
      {renderers.react(content, React, {
        components: { Fence, Callout },
      })}
    </>
  );
}
